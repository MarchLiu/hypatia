/**
 * dsh-hypatia-auto-memory — event-driven Hypatia memory for DSH.
 *
 * One host plugin whose collect fiber owns the durable core (state domain,
 * queue, hypatia client, writer, collector); optional child fibers attach the
 * rest, so a deployment missing `llm` still logs conversations and a
 * deployment missing `skills` still remembers:
 *
 *   collect  (inject sessions, storageDomain, subprocess)
 *     ├─ consolidate (inject llm, sessions)   — span summaries, cascade, work units
 *     ├─ recall      (inject agents)          — rules/taboos preload at session start
 *     ├─ housekeeping (inject sessionPersistence) — prune progress of vanished sessions
 *     ├─ memory-api  (inject webServer)       — the Memory tab + shelf listing routes
 *     ├─ on-demand   (inject commands)        — /hypatia-summarize, no model turn
 *     ├─ on-demand   (inject tools)           — the hypatia_summarize tool
 *     ├─ auto-approve (inject approval, tools) — the agent's own bash `hypatia` calls
 *     └─ skills      (inject skills)          — hypatia-memory, hypatia, hypatia-dream
 *
 * The last two fibers are here because this plugin REPLACES `dsh-hypatia`:
 * writing memory by asking the model to do it did not happen in practice, and
 * the parts of that plugin still worth having — approving the agent's hypatia
 * calls and shipping the CLI skills — only serve the retrieval half of this
 * one. Installing both is a misconfiguration; see README.
 *
 * @module dsh-hypatia-auto-memory
 */

import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { absolutizeDates, blocksToText, eventDate, redactSecrets } from './content-policy.js'
import { STARTUP_FIELDS, installConfig } from './config.js'
import { openState } from './state.js'
import { EMPTY_PROGRESS, advanceProgress } from './progress.js'
import { createStatus } from './status.js'
import { createHypatiaClient } from './hypatia-client.js'
import { createHypatiaCli } from './hypatia-cli.js'
import { createWriter } from './writer.js'
import { TaskDeferredError, createQueue } from './queue.js'
import {
  countLoggableMessages, createCollector, formatSpan, withdrawalWork, withdrawnOrdinals, withdrawnSeqs,
} from './collector.js'
import { createCascade } from './cascade.js'
import { createConsolidator, PLUGIN_NAME } from './consolidator.js'
import {
  SUMMARIZE_COMMAND,
  SUMMARIZE_TOOL,
  disabledOutcome,
  registerSummarizeCommand,
  registerSummarizeTool,
  unavailableOutcome,
} from './on-demand.js'
import { openModelLog } from './model-log.js'
import { createReasoningResolver } from './reasoning.js'
import { buildStatus } from './memory-status.js'
import { createMemoryApi } from './memory-api.js'
import { createRecall } from './recall.js'
import { createPersistedSessions } from './persisted-session.js'
import { createAutoApprove } from './auto-approve.js'
import { armSandboxWhitelistPrompt } from './sandbox-whitelist.js'
import { registerSkills } from './skills.js'
import { backfillConsolidation, normalizeTaskProjects, pruneVanishedSessions, reconcileProgress, sessionDirectory } from './housekeeping.js'
import { shelfTable } from './shelf.js'

export const name = 'dsh-hypatia-auto-memory'

// The Loader projects this schema into the settings tab (volatile fields) and
// validates every committed edit; see config.js.
export { Config } from './config.js'

const PACKAGE_ROOT = dirname(dirname(fileURLToPath(import.meta.url)))
const DEFAULT_SKILLS_DIR = join(PACKAGE_ROOT, 'skills')

/* -------------------------------------------------------------------------- */
/* Composition                                                                */
/* -------------------------------------------------------------------------- */

export function apply(ctx, config = {}) {
  const status = createStatus(ctx, name)
  // Created on the entry fiber's own context: `loader/volatile-update` is
  // instance-local, so the handle must subscribe here, not in a child fiber.
  const configHandle = installConfig(ctx, config, (message) => status.warn(message))

  let stopped = false
  ctx.effect(() => () => {
    stopped = true
  }, `${name}: stop collector restarts`)

  /** The startup-read values the running (or next) collector uses. */
  const pickStartup = (value) => Object.fromEntries(STARTUP_FIELDS.map((key) => [key, value[key]]))
  let collect
  let running
  const mount = () => {
    if (stopped) return
    running = pickStartup(configHandle.get())
    collect = ctx.plugin({
      name: `${name}/collect`,
      inject: ['sessions', 'storageDomain', 'subprocess'],
      apply: (collectCtx) => applyCollect(collectCtx, configHandle, status),
    })
  }

  // Mounted once every profile entry has settled rather than right away: dsh
  // imports a pre-0.1.7 `settings.yaml` at that point, as a live update, and
  // starting later narrows the window in which the collector runs on the
  // defaults before the imported values arrive. Whatever still lands after it
  // started is handled by the restart below.
  let chain = Promise.resolve()
    .then(() => ctx.root?.loader?.await?.())
    .catch(() => {})
    .then(mount)
    .catch((error) => status.error('collector mount failed', error))

  // The collector reads STARTUP_FIELDS once; a committed change to one of them
  // restarts it, so an edit on the settings tab takes effect on save.
  configHandle.onChange((value) => {
    if (running === undefined) return
    const changed = STARTUP_FIELDS.filter((key) => !Object.is(value[key], running[key]))
    if (changed.length === 0) return
    running = pickStartup(value)
    status.info(`${changed.join(', ')} changed; restarting the collector`)
    chain = chain.then(async () => {
      const previous = collect
      collect = undefined
      await previous?.dispose()
      mount()
    }).catch((error) => status.error('collector restart failed', error))
  })
}

function applyCollect(ctx, configHandle, status) {
  let disposed = false
  ctx.effect(() => () => {
    disposed = true
  }, `${name}: dispose flag`)

  void start().catch((error) => {
    status.error('startup failed; plugin will not collect until it restarts (a settings change or a profile reload)', error)
  })

  /**
   * The Memory tab's read side and the settings card's shelf listing, plus the
   * one write route an explicit "summarise now" rides.
   * Optional: it needs `webServer`, so a headless or TUI composition simply has
   * no Memory tab and nothing else changes. The route authenticates itself —
   * see memory-api.js.
   */
  function mountMemoryApi({ cli, read, shared, summarize }) {
    ctx.plugin({
      name: `${name}/memory-api`,
      inject: ['webServer'],
      apply: (c) => {
        const api = createMemoryApi({ cli, read, status, summarize })
        c.effect(() => {
          const unregister = c.webServer.register(api.route)
          shared.memoryApi = api
          return () => {
            unregister()
            // Only if this fiber's own API is still the live one: a reload may
            // have mounted a successor before this disposer runs.
            if (shared.memoryApi === api) shared.memoryApi = undefined
          }
        }, `${name}: memory API route`)
      },
    })
  }

  async function start() {
    if (configHandle.get().enabled === false) {
      status.info('disabled via hypatia-auto-memory.enabled=false')
      // The settings card still offers the shelf listing, so a user can pick
      // the shelf before turning the plugin back on. No state is opened: the
      // Memory tab reads as "nothing logged", which is the truth while off.
      mountMemoryApi({
        cli: createHypatiaCli(ctx, {
          get binaries() {
            return configHandle.get().binaries
          },
          shelf: configHandle.get().shelf,
        }),
        read: (sessionId) => buildStatus({ progressEntries: [], taskEntries: [], sessionId }),
        shared: {},
        // No state is being written while the plugin is off, so the route
        // answers with the truth rather than an empty span.
        summarize: disabledOutcome,
      })
      return
    }

    const openedState = await openState(ctx)
    if (disposed) {
      await openedState.domain.close().catch(() => {})
      return
    }

    // Fixed for this run; see shelf.js. Every table read below goes through a
    // view of this shelf's rows, so a switch never carries one shelf's
    // watermarks or queued work into another.
    const shelf = configHandle.get().shelf
    const state = {
      progress: shelfTable(openedState.progress, shelf),
      tasks: shelfTable(openedState.tasks, shelf),
    }
    // Which model a background attempt ran on is otherwise unknowable from
    // outside the process: the pick is made in-process at call time, the host
    // logger scrolls away, and the usage ledger cannot see plugin calls. Kept in
    // its own domain so a diagnostics row can never reject the open of the
    // watermarks (see model-log.js). Not shelf-scoped: the route list is config,
    // and the question is about this process, not this shelf.
    const openedModelLog = await openModelLog(ctx, { status })
    if (disposed) {
      // Same race as the state domain above: if the fiber went away while this
      // was opening, its disposer may never run, so close the handle here.
      await openedModelLog.domain?.close().catch(() => {})
      return
    }
    const modelLog = openedModelLog.modelLog

    const queue = createQueue({
      tasks: state.tasks,
      getConfig: () => configHandle.get().queue,
      status,
    })

    /**
     * Wrap a consolidation-side executor so its settle drops that session's
     * cached content. The memory API serves summaries and work units out of a
     * cache (they cost hypatia round trips), and consolidation and cascade are
     * exactly what change them — invalidating here is what makes the tab show
     * the new summary without waiting for the cache's own expiry. The API is
     * mounted by a child fiber below, so it is reached through `shared`,
     * late-bound; `finally` covers a failed run too, since it may have written
     * part of its output before failing.
     */
    const invalidatingContent = (executor) => async (task) => {
      try {
        return await executor(task)
      } finally {
        shared.memoryApi?.invalidate(task.sessionId)
      }
    }
    const cli = createHypatiaClient(ctx, {
      get binaries() {
        return configHandle.get().binaries
      },
      get transport() {
        return configHandle.get().transport
      },
      shelf,
    }, { status })
    // In-flight tasks drain (up to the queue's deadline) before the process
    // they write through is closed.
    ctx.effect(() => () => {
      void queue.dispose().finally(() => cli.dispose())
    }, `${name}: dispose queue and hypatia connection`)
    // The configured transport; a binary without `mcp` is only found out, and
    // warned about, at the first call.
    status.info(`writing to shelf "${shelf}" (transport: ${configHandle.get().transport})`)

    // Before the collector can queue anything and before any executor can run
    // a task: see housekeeping.js.
    await normalizeTaskProjects({ tasks: state.tasks, status })
    if (disposed) return

    // The one place a missing shelf is noticed before the first write fails on
    // it. The settings card's dropdown reads the same listing on demand from
    // the route family's `/shelves` endpoint (see memory-api.js); nothing polls
    // here anymore.
    void cli.listShelves().then((shelves) => {
      const found = shelves.find((entry) => entry.name === shelf)
      if (found === undefined) status.warn(`shelf "${shelf}" is not registered with hypatia; every write will fail until it is (hypatia connect <dir> --name ${shelf})`)
      else if (!found.connected) status.warn(`shelf "${shelf}" is registered but not connected; writes will fail until it is`)
    }, (error) => {
      status.warn(`shelf listing failed: ${error instanceof Error ? error.message : String(error)}`)
    })
    // Late-bound handles to the optional child fibers. Adjudication needs a
    // model route, which only the consolidate fiber has — without it the writer
    // stores a unit with no relationship, which is the correct degradation,
    // never a guessed one. The memory API is late-bound for the same reason: it
    // needs `webServer`, and a headless composition has none.
    const shared = { consolidator: undefined, memoryApi: undefined }

    /**
     * The one function all three explicit faces call: the slash command, the
     * tool, and the Memory tab's button. Late-bound like the rest of `shared`,
     * because the consolidator only exists once `llm` arrives; before that (or
     * in a composition without `llm` at all) the request is answered honestly
     * as unavailable rather than silently dropped.
     *
     * @param {string} sessionId
     * @returns {Promise<{sessionId: string, queued: boolean, reason: string, fromSeq: number, toSeq: number}>}
     */
    const summarizeNow = (sessionId) => shared.consolidator === undefined
      ? Promise.resolve(unavailableOutcome(sessionId))
      : shared.consolidator.consolidateNow(sessionId)

    // Mounted before reconcile and backfill, which can take long or fail: the
    // settings card's shelf listing rides this route family, and a broken
    // shelf is exactly when the user needs it.
    mountMemoryApi({
      cli,
      read: (sessionId) => buildStatus({
        progressEntries: state.progress.entries(),
        taskEntries: state.tasks.entries(),
        sessionId,
      }),
      shared,
      summarize: summarizeNow,
    })

    const writer = createWriter(cli, {
      status,
      adjudicate: (unit, candidates, attempt) => shared.consolidator === undefined
        ? Promise.resolve(undefined)
        : shared.consolidator.adjudicate(unit, candidates, attempt),
    })

    const collector = createCollector({
      ctx,
      queue,
      progress: state.progress,
      getConfig: configHandle.get,
      status,
      onTurnEnd: (sessionId, turn, triggerState) => shared.consolidator === undefined
        ? Promise.resolve({ resetTokens: false, advanceCheckpoint: false })
        : shared.consolidator.onTurnEnd(sessionId, turn, triggerState),
      onSessionEnd: (sessionId, session) => shared.consolidator === undefined
        ? Promise.resolve(false)
        : shared.consolidator.onSessionEnd(sessionId, session),
    })

    // Persist every open session's closing tail before the queue stops taking
    // work. Registered AFTER the queue/CLI disposer above on purpose: cordis
    // disposes a fiber's effects in reverse registration order, so this runs
    // first, while `queue.enqueue` is still a durable write rather than the
    // silent no-op it becomes once `dispose()` has set the flag. See
    // `finalizeLiveSessions` for why the model call is not waited for.
    ctx.effect(() => () => collector.finalizeLiveSessions(), `${name}: persist closing tails`)

    // Live store first, storage second. `session/created` — the signal that
    // wakes a deferred task — fires only where an agent runs, so a finished
    // session (and every subagent session) may never emit it again; storage
    // supplies the same events without publishing anything. Late-bound: a
    // composition without `sessionPersistence` keeps today's behavior, which is
    // to wait. See persisted-session.js.
    let loadPersisted = async () => undefined
    const resolveSession = async (sessionId) => collector.sessionFor(sessionId) ?? await loadPersisted(sessionId)

    ctx.plugin({
      name: `${name}/persisted-sessions`,
      inject: ['sessionPersistence'],
      apply: (c) => {
        const store = createPersistedSessions({ persistence: c.sessionPersistence, status })
        loadPersisted = store.get
        c.effect(() => () => {
          loadPersisted = async () => undefined
          store.clear()
        }, `${name}: release persisted sessions`)
      },
    })

    /** Log executor: re-read the span, format per protocol, write idempotently. */
    queue.registerExecutor('log-message', async (task) => {
      // Resolved through the collector, not `ctx.sessions`: a session closing
      // down has already left the store while its final span is still queued.
      const session = await resolveSession(task.sessionId)
      if (session === undefined) {
        throw new TaskDeferredError(`session ${task.sessionId} is not loaded; deferred until it is`)
      }
      const config = configHandle.get()
      const events = session.snapshotEvents(task.fromSeq, task.toSeq)
      // Entry names use a dense message ordinal, recomputed from the log prefix
      // rather than carried in the watermark: the log is immutable, so counting
      // is replay-stable by construction and cannot drift out of sync.
      const inherited = session.inheritedEventCount ?? 0
      const baseIndex = countLoggableMessages(session.snapshotEvents(inherited, task.fromSeq))
      // The marker window ends at `session.seq`, never at the task's own end: a
      // marker's seq is greater than every seq it shadows, so it can sit past the
      // range being logged — which is exactly the case where what it hid was
      // written by an earlier run (docs/rewind-memory-consistency.md §3 fact 5).
      // Everything else about the decision is collector.js's, where it is a pure
      // function over the log and therefore testable without a session.
      const { seqs: withdrawn, retract } = withdrawalWork({
        log: session.snapshotEvents(task.fromSeq, session.seq ?? 0),
        prefix: session.snapshotEvents(inherited, task.fromSeq),
        fromSeq: task.fromSeq,
        inherited,
      })
      const formatted = formatSpan(events, {
        now: new Date(),
        maxAssistantChars: config.collector.maxAssistantChars,
        maxUserChars: config.collector.maxUserChars,
        toolLedger: config.collector.toolLedger,
        baseIndex,
        before: task.fromSeq > 0 ? session.snapshotEvents(task.fromSeq - 1, task.fromSeq)[0] : undefined,
      })
      // A message the user took back is skipped, and one an earlier run wrote
      // before the marker landed is retracted below. Anchored on the log range,
      // never on "does the entry exist": a replayed span sees the same marker
      // again, which is what keeps a retracted message from being written back.
      const skipped = new Set()
      for (const item of formatted) {
        const isWithdrawn = withdrawn.has(item.seq)
        if (isWithdrawn) skipped.add(item.index)
        await writer.writeMessage({
          sessionId: task.sessionId,
          index: item.index,
          markdown: item.markdown,
          project: task.project,
          withdrawn: isWithdrawn,
        })
      }
      // Retracting is idempotent, and the attempts stop once the watermark passes
      // the marker's own seq, after which it leaves every later window too.
      for (const index of retract.values()) {
        skipped.add(index)
        await writer.retractMessage({ sessionId: task.sessionId, index })
      }
      const linkTo = baseIndex + formatted.length
      // `belongTo` edges only exist once the session node does. Most sessions
      // never produce one (it needs a host-supplied summary), so this stays a
      // watermark check rather than a lookup per span.
      const before = state.progress.get(task.sessionId) ?? EMPTY_PROGRESS
      if (before.hasSessionNode === 1 && linkTo > before.lastBelongToIndex) {
        await writer.writeSessionNode({
          sessionId: task.sessionId,
          markdown: '',
          project: task.project,
          // The cursor counts every loggable message, withdrawn ones included, so
          // the ordinals stay aligned with the log; only the edges are skipped.
          linkFrom: before.lastBelongToIndex,
          linkTo,
          skip: skipped,
        })
      }
      // Awaited: the task must not be reported done — and deleted — before the
      // watermark that makes its range unnecessary is durable.
      await advanceProgress(state.progress, task.sessionId, (current) => ({
        lastLoggedSeq: Math.max(current.lastLoggedSeq, task.toSeq),
        lastBelongToIndex: current.hasSessionNode === 1
          ? Math.max(current.lastBelongToIndex, linkTo)
          : current.lastBelongToIndex,
      }))
    })

    /**
     * Session-node executor: build `session-<id>` from the host summary at this
     * seq, then link every message logged so far to it.
     */
    queue.registerExecutor('session-node', async (task) => {
      const session = await resolveSession(task.sessionId)
      if (session === undefined) {
        throw new TaskDeferredError(`session ${task.sessionId} is not loaded; deferred until it is`)
      }
      const [event] = session.snapshotEvents(task.fromSeq, task.toSeq)
      if (event === undefined) return
      const summary = event.type === 'session/title'
        ? String(event.data?.title ?? '')
        : blocksToText(event.data?.summary ?? [])
      const markdown = absolutizeDates(redactSecrets(summary), eventDate(event, new Date())).trim()
      // The protocol forbids inventing a session summary: no text, no node.
      if (markdown === '') return

      const inherited = session.inheritedEventCount ?? 0
      const current = state.progress.get(task.sessionId) ?? EMPTY_PROGRESS
      const linkTo = countLoggableMessages(session.snapshotEvents(inherited, current.lastLoggedSeq))
      // This node links every message logged so far, so it owes the same skip set
      // the log executor used — without it this node's own range would settle
      // edges to messages that have no entry to link from. The markers are read
      // from the whole log (the node is written once per session), and a marker
      // anywhere in it can only hide earlier seqs: one slice, two passes.
      const log = session.snapshotEvents(inherited, session.seq ?? 0)
      const withdrawn = withdrawnSeqs(log)
      const skip = new Set(withdrawnOrdinals(log, withdrawn, inherited).values())
      await writer.writeSessionNode({
        sessionId: task.sessionId,
        markdown,
        project: task.project,
        linkFrom: current.lastBelongToIndex,
        linkTo,
        skip,
      })
      await advanceProgress(state.progress, task.sessionId, (progressRow) => ({
        hasSessionNode: 1,
        lastBelongToIndex: Math.max(progressRow.lastBelongToIndex, linkTo),
      }))
    })

    // Failed log-message records carry nothing the watermark does not: the range
    // was never marked logged, so the session's next flush covers it again.
    // Pruned at startup so the error message survives the run that produced it.
    const pruned = await queue.pruneFailed(['log-message'])
    if (pruned > 0) status.info(`pruned ${pruned} failed log-message task(s); the watermarks re-derive their ranges`)

    // A restart (a startup field changed) may have begun while the passes
    // above ran; stop before touching the shelf any further.
    if (disposed) return

    // Progress rows whose session left nothing in the shelf (a wiped or replaced
    // shelf) are reset BEFORE backfill, so a live session among them is logged
    // from the start in this very run. See housekeeping.js for why it only acts
    // on positive evidence.
    if (configHandle.get().housekeeping?.reconcileOnStartup !== false) {
      await reconcileProgress({ progress: state.progress, cli, status })
      if (disposed) return
    }

    await collector.backfillLiveSessions()
    status.info('collector running (backfill complete)')

    // Both passes need `sessionPersistence` for the full listing — live sessions
    // alone would make every unloaded session look gone, and carry the cwd a
    // scope is resolved from. A child fiber, registered only after the reconcile
    // pass above so the two never act on one row at once. Each pass keeps its
    // own switch: turning pruning off must not also stop consolidation backfill.
    const housekeeping = configHandle.get().housekeeping ?? {}
    if (housekeeping.pruneVanishedSessions !== false || housekeeping.backfillConsolidation !== false) {
      ctx.plugin({
        name: `${name}/housekeeping`,
        inject: ['sessionPersistence'],
        apply: (c) => {
          void (async () => {
            // Consumed in one tested place: `list()` yields snapshots wrapping a
            // `header`, and reading `id`/`cwd` off the snapshot left every
            // session unresolved — see sessionDirectory.
            const snapshots = await c.sessionPersistence.list()
            const { cwdById, knownSessionIds } = sessionDirectory(snapshots)
            if (housekeeping.pruneVanishedSessions !== false) {
              await pruneVanishedSessions({
                progress: state.progress,
                queue,
                knownSessionIds,
                isLive: (id) => ctx.sessions.get(id) !== undefined,
                status,
              })
            }
            // After pruning, so a vanished session's tail is never queued.
            if (housekeeping.backfillConsolidation !== false) {
              await backfillConsolidation({
                progress: state.progress,
                queue,
                cwdFor: (id) => cwdById.get(id),
                projectForCwd: collector.projectForCwd,
                perScopeLimit: housekeeping.consolidationBackfillPerScope,
                status,
              })
            }
          })().catch((error) => {
            status.warn(`startup housekeeping skipped: ${String(error)}`)
          })
        },
      })
    }

    // Optional child fibers — each waits only for the services it needs.
    ctx.plugin({
      name: `${name}/consolidate`,
      inject: ['llm', 'sessions'],
      apply: (c) => {
        // One capability cache and one set of "your configured effort was
        // dropped" warnings for both executors: extraction, adjudication and the
        // archive all resolve their thinking level through this.
        const reasoning = createReasoningResolver({ llm: c.llm, status })
        const consolidator = createConsolidator({
          queue,
          progress: state.progress,
          // Same resolver as the log executor: end-of-session consolidation runs
          // against a session the store has already released — and a startup
          // backfill runs against one it may never publish again.
          sessions: { get: resolveSession },
          llm: c.llm,
          cli,
          writer,
          getConfig: configHandle.get,
          status,
          modelLog,
          reasoning,
          projectFor: collector.projectFor,
        })
        queue.registerExecutor('consolidate', invalidatingContent(consolidator.execute))
        // The same priority selector and the same reasoning resolver as the
        // consolidator, but no shared cursor: each executor holds its own task's
        // `attempts`, so an archive retry degrades down the same list the
        // summary just degraded down.
        const cascade = createCascade({
          cli,
          llm: c.llm,
          selectRoute: consolidator.selectRoute,
          getConfig: configHandle.get,
          status,
          modelLog,
          reasoning,
        })
        queue.registerExecutor('cascade', invalidatingContent(cascade.execute))
        shared.consolidator = consolidator
      },
    })

    // The two Host-side faces of an explicit "summarise this session now".
    //
    // Separate fibers on purpose: `commands` is absent from a headless or TUI
    // composition while `tools` is present, and one fiber injecting both would
    // take the tool down with the missing command registry. Neither is mounted
    // under `auto-approve`, which exists only when `autoApprove !== false` and
    // has nothing to do with this — the tool is a first-class action, not a
    // shell call to approve.
    //
    // Both read `summarizeNow` at invocation time, so registering them before
    // the consolidate fiber settles is fine; a request that arrives before a
    // route exists is answered as unavailable instead of silently dropped.
    ctx.inject(['commands'], (commandCtx) => {
      try {
        registerSummarizeCommand(commandCtx, { run: summarizeNow, status })
      } catch (error) {
        status.warn(`/${SUMMARIZE_COMMAND} could not be registered: ${String(error)}`)
      }
    })
    ctx.inject(['tools'], (toolCtx) => {
      void registerSummarizeTool(toolCtx, { run: summarizeNow, status })
        .catch((error) => status.warn(`the ${SUMMARIZE_TOOL} tool could not be registered: ${String(error)}`))
    })

    ctx.plugin({
      name: `${name}/recall`,
      inject: ['agents'],
      apply: (c) => {
        createRecall({ ctx: c, cli, shelf, getConfig: configHandle.get, status, projectFor: collector.projectFor })
      },
    })

    // Auto-approve is for the agent's OWN bash `hypatia` calls — retrieval,
    // explicit remember/forget. This plugin's writes never pass through it:
    // they go over its own `hypatia mcp` connection (or argv arrays on the
    // CLI fallback), with no shell, no tool call and no approval in the path.
    if (configHandle.get().autoApprove !== false) {
      ctx.plugin({
        name: `${name}/auto-approve`,
        inject: ['approval', 'tools'],
        apply: (c) => {
          createAutoApprove(c, { getBinaries: () => configHandle.get().binaries, status })
        },
      })
    }

    // First-activation sandbox whitelist for `~/.hypatia` (see sandbox-whitelist.js).
    // Idempotent: silently skips when any patch layer already whitelists the
    // root, on win32 (the ACL backend has no additionalWritableRoots), or when
    // the user turned the prompt off with `sandboxWhitelistPrompt: false`.
    // When it does arm, the ask is a REAL approval prompt on the first tool
    // call of a session — never auto-answered; approving splices the root into
    // the patch layer that already carries the sandbox-policy entry.
    if (configHandle.get().sandboxWhitelistPrompt !== false) {
      ctx.plugin({
        name: `${name}/sandbox-whitelist`,
        inject: ['approval'],
        apply: (c) => {
          try {
            armSandboxWhitelistPrompt({
              ctx: c,
              requestApproval: (req) => c.approval.request(req),
              options: { status },
            })
          } catch (error) {
            status.warn(`sandbox whitelist setup failed: ${String(error)}`)
          }
        },
      })
    }

    if (configHandle.get().skills !== false) {
      ctx.plugin({
        name: `${name}/skills`,
        inject: ['skills'],
        apply: (c) => {
          void registerSkills(c, configHandle.get().skillsDir ?? DEFAULT_SKILLS_DIR, status, PLUGIN_NAME)
            .catch((error) => status.warn(`skill registration failed: ${String(error)}`))
        },
      })
    }
  }
}
