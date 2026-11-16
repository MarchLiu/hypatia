/**
 * Cordis Config schema and live configuration for dsh-hypatia-auto-memory.
 *
 * Since dsh 0.1.7 the settings domain is a projection over the Loader: a
 * plugin's settings page IS its cordis Config, stored in the profile patch and
 * edited through Settings → Plugins. The old `settings.installSection` /
 * `settings.register` Host API is gone. Fields meant to be editable live are
 * marked `.volatile()` — only volatile fields reach the settings forms at all,
 * and editing one updates the running reference (`loader/volatile-update`)
 * instead of remounting the plugin. The two packaging knobs (`skills`,
 * `skillsDir`) stay ordinary fields: changing them remounts the plugin, which
 * is exactly what a skills-directory change needs.
 *
 * @module dsh-hypatia-auto-memory/config
 */

import z from '@deepseek-ai/schemastery'

/** Profile entry id (and former settings namespace) this plugin answers to. */
export const SETTINGS_NAMESPACE = 'hypatia-auto-memory'

/** Composition defaults; every schema field mirrors one of these. */
export const DEFAULTS = {
  enabled: true,
  binaries: ['hypatia'],
  // How the plugin itself talks to hypatia: a private `hypatia mcp` process,
  // or one CLI command per call. A binary without `mcp` falls back to the CLI
  // on its own (see hypatia-client.js). Applies from the next call.
  transport: 'mcp',
  // Where every entry this plugin writes — and every lookup it makes — goes.
  // Read when the collector starts, like `enabled`; a committed change
  // restarts the collector on the new shelf (see STARTUP_FIELDS).
  shelf: 'default',
  // Approve the AGENT's own `hypatia …` bash calls (retrieval, explicit
  // remember/forget). This plugin's own writes never need it — they go over
  // its own connection, never through a tool call. Read when the collector
  // starts, like `enabled`.
  autoApprove: true,
  // Ask once (a real approval prompt) to add ~/.hypatia to the deployment's
  // sandbox-policy additionalWritableRoots on first activation. Set false to
  // never ask; approval itself writes the patch line, so saying yes once is
  // permanent. Read at activation, like autoApprove.
  sandboxWhitelistPrompt: true,
  collector: {
    enabled: true,
    maxAssistantChars: 8000,
    // User messages were never capped, so a pasted log went into hypatia whole.
    // Generous, because what the user typed is the part most worth keeping.
    maxUserChars: 32000,
    // Tool-call ledgers (name / count / success / duration) are intentionally
    // off by default: they still reveal which tools were used and when, and the
    // user has asked to avoid writing any tool-call-derived data to hypatia.
    // Set this to true to restore the previous behavior.
    toolLedger: false,
  },
  consolidation: {
    enabled: true,
    models: [],
    maxInputTokens: 16000,
    maxOutputTokens: 2000,
    timeoutMs: 120_000,
    checkEveryTurns: 5,
    minNewTokens: 3000,
    maxWorkUnitsPerRun: 3,
    // Work-unit relationship adjudication. `dedupMaxDistance` is a cosine
    // distance ceiling on candidates worth judging at all; anything further
    // away is not a relative, and asking about it only invites a wrong edge.
    adjudicate: true,
    dedupMaxDistance: 0.45,
    dedupCandidates: 5,
    // Hierarchical archive. 16:1 per tier is the protocol's constant; it is what
    // makes the summary layer log₁₆(n) deep instead of growing with the
    // conversation.
    cascade: {
      enabled: true,
      batchSize: 16,
    },
  },
  queue: {
    concurrency: 1,
    maxAttempts: 3,
    retryDelayMs: 5000,
    flushWindowMs: 120_000,
  },
  // Recall is a session seed, not a per-turn injector: the agent pulls the rest
  // through the bundled skill. `maxEntries` / `maxCharsPerEntry` belonged to the
  // removed per-turn path and would now be knobs that control nothing.
  recall: {
    enabled: true,
    preloadRulesTaboos: true,
  },
  // Startup housekeeping of the progress table (see housekeeping.js). Both act
  // only on positive evidence and skip on any doubt.
  housekeeping: {
    // Reset a row whose session has no msg-* entry left in the shelf (a wiped
    // or replaced shelf). A session whose messages were all deleted on purpose
    // looks the same and would be logged again — switch off if that matters.
    reconcileOnStartup: true,
    // Drop the row and tasks of a session DSH no longer has.
    pruneVanishedSessions: true,
    // Consolidate a logged tail the in-session trigger never reached. A process
    // restart takes the session-end path with it, so a short session would
    // otherwise stay logged-but-never-distilled. This pass ignores the live
    // trigger's `minNewTokens` floor on purpose — at boot there is no later turn
    // to wait for, and a tail skipped here is skipped for good — and bounds its
    // cost with `consolidationBackfillPerScope` instead.
    backfillConsolidation: true,
    // Sessions whose tail this one pass may queue, per project scope, per start.
    // The cap is what replaces the floor: a large backlog drains over several
    // starts, each spending at most this many model calls per scope.
    consolidationBackfillPerScope: 4,
  },
  // Cordis-level packaging knobs (not shown on the settings tab): whether to
  // register the bundled skills, and where they live.
  skills: true,
}

/**
 * Exact LLM route eligible for one consolidation attempt. Blank names are
 * refused by the schema, so the Loader rejects such an edit before committing
 * it and the settings tab reports the refusal — the pre-0.1.7 write hook did
 * the same.
 *
 * **The array is a priority order, highest priority first**, not a set: the
 * first attempt of a task takes the head and each retry degrades one entry down
 * (see `selectConsolidationRoute` in consolidator.js). Nothing here expresses
 * that — position does — so a reordering IS a behavior change, and the settings
 * card is the only editor that lets a user set it.
 *
 * `reasoningEffort` is the optional per-route thinking level. The vocabulary is
 * ADAPTER-OWNED and opaque (see reasoning.js), so this schema cannot validate a
 * value against it — it only refuses a blank. A value the route does not declare
 * is dropped at call time with a warning rather than sent to a guaranteed core
 * rejection. Absent means "follow the purpose policy", which is what every call
 * did before the field existed.
 */
const ConsolidationModelSchema = z.object({
  provider: z.string().pattern(/\S/),
  model: z.string().pattern(/\S/),
  // Absent by default (schemastery fields are nullable unless `.required()`),
  // which is what makes "no configured effort" the pre-existing behavior.
  reasoningEffort: z.string().pattern(/\S/),
})

/**
 * The plugin's cordis Config. Every behavior knob is volatile so the settings
 * tab can edit it live; `skills` / `skillsDir` are ordinary fields because
 * they select files on disk and a change must remount the plugin anyway.
 *
 * `shelf` is constrained by pattern rather than a runtime hook: the Loader
 * validates a volatile candidate before committing it, so a blank or padded
 * name never reaches the running value.
 *
 * `enabled`, `shelf` and `autoApprove` are volatile too, although the collector
 * reads them once when it starts: volatile is what puts a field on the
 * settings tab, and the entry restarts the collector itself when one of them
 * changes (see STARTUP_FIELDS and index.js).
 */
export const Config = z.object({
  enabled: z.boolean().default(DEFAULTS.enabled).volatile(),
  binaries: z.array(z.string()).default(DEFAULTS.binaries).volatile(),
  transport: z.union(['mcp', 'cli']).default(DEFAULTS.transport).volatile(),
  shelf: z.string().pattern(/^\S+$/).default(DEFAULTS.shelf).volatile(),
  autoApprove: z.boolean().default(DEFAULTS.autoApprove).volatile(),
  sandboxWhitelistPrompt: z.boolean().default(DEFAULTS.sandboxWhitelistPrompt).volatile(),
  collector: z.object({
    enabled: z.boolean().default(DEFAULTS.collector.enabled),
    maxAssistantChars: z.number().step(1).min(500).default(DEFAULTS.collector.maxAssistantChars),
    maxUserChars: z.number().step(1).min(500).default(DEFAULTS.collector.maxUserChars),
    toolLedger: z.boolean().default(DEFAULTS.collector.toolLedger),
  }).default(DEFAULTS.collector).volatile(),
  consolidation: z.object({
    enabled: z.boolean().default(DEFAULTS.consolidation.enabled),
    models: z.array(ConsolidationModelSchema).default(DEFAULTS.consolidation.models),
    maxInputTokens: z.number().step(1).min(1000).default(DEFAULTS.consolidation.maxInputTokens),
    maxOutputTokens: z.number().step(1).min(200).default(DEFAULTS.consolidation.maxOutputTokens),
    timeoutMs: z.number().step(1).min(1000).max(3_600_000).default(DEFAULTS.consolidation.timeoutMs),
    checkEveryTurns: z.number().step(1).min(1).default(DEFAULTS.consolidation.checkEveryTurns),
    minNewTokens: z.number().step(1).min(500).default(DEFAULTS.consolidation.minNewTokens),
    maxWorkUnitsPerRun: z.number().step(1).min(1).max(10).default(DEFAULTS.consolidation.maxWorkUnitsPerRun),
    adjudicate: z.boolean().default(DEFAULTS.consolidation.adjudicate),
    dedupMaxDistance: z.number().min(0).max(2).default(DEFAULTS.consolidation.dedupMaxDistance),
    dedupCandidates: z.number().step(1).min(1).max(20).default(DEFAULTS.consolidation.dedupCandidates),
    cascade: z.object({
      enabled: z.boolean().default(DEFAULTS.consolidation.cascade.enabled),
      batchSize: z.number().step(1).min(2).max(64).default(DEFAULTS.consolidation.cascade.batchSize),
    }).default(DEFAULTS.consolidation.cascade),
  }).default(DEFAULTS.consolidation).volatile(),
  queue: z.object({
    concurrency: z.number().step(1).min(1).max(8).default(DEFAULTS.queue.concurrency),
    maxAttempts: z.number().step(1).min(1).max(10).default(DEFAULTS.queue.maxAttempts),
    retryDelayMs: z.number().step(1).min(100).default(DEFAULTS.queue.retryDelayMs),
    flushWindowMs: z.number().step(1).min(0).max(600_000).default(DEFAULTS.queue.flushWindowMs),
  }).default(DEFAULTS.queue).volatile(),
  recall: z.object({
    enabled: z.boolean().default(DEFAULTS.recall.enabled),
    preloadRulesTaboos: z.boolean().default(DEFAULTS.recall.preloadRulesTaboos),
  }).default(DEFAULTS.recall).volatile(),
  housekeeping: z.object({
    reconcileOnStartup: z.boolean().default(DEFAULTS.housekeeping.reconcileOnStartup),
    pruneVanishedSessions: z.boolean().default(DEFAULTS.housekeeping.pruneVanishedSessions),
    backfillConsolidation: z.boolean().default(DEFAULTS.housekeeping.backfillConsolidation),
    consolidationBackfillPerScope: z.number().step(1).min(1).max(64)
      .default(DEFAULTS.housekeeping.consolidationBackfillPerScope),
  }).default(DEFAULTS.housekeeping).volatile(),
  skills: z.boolean().default(DEFAULTS.skills),
  skillsDir: z.string(),
})

/* -------------------------------------------------------------------------- */
/* Live reads                                                                 */
/* -------------------------------------------------------------------------- */

// The shared volatile-reference protocol (cosmokit), recognized across module
// copies without importing the package: a volatile field's parsed value is a
// frozen `{ get, [write] }` reference whose `get()` returns the latest
// immutable snapshot.
const VOLATILE_WRITE = Symbol.for('cosmokit.volatile.write')

function isVolatileRef(value) {
  return typeof value === 'object' && value !== null && VOLATILE_WRITE in value
}

/**
 * One plain-data snapshot of the resolved config: volatile references read
 * through to their current value, everything else passed through. Snapshots
 * are immutable, so the result is safe to hand to every consumer.
 */
export function snapshotConfig(config) {
  if (isVolatileRef(config)) return config.get()
  if (config === null || typeof config !== 'object') return config
  const out = Array.isArray(config) ? [] : {}
  for (const [key, value] of Object.entries(config)) {
    out[key] = isVolatileRef(value) ? value.get() : value
  }
  return out
}

/**
 * Cross-field checks the schema cannot express. The schema already refuses a
 * blank route; what is left is a duplicate, which the Loader has no hook to
 * refuse (it validates fields, not the list as a whole). A duplicate route is
 * not harmless any more: with the list read as a priority order, a repeat both
 * wastes a degradation step and claims a second priority slot for one route, so
 * the running side drops it and says so. Blank names are still dropped here as a
 * defence for values that bypassed the schema.
 *
 * @param {any} value - one plain config snapshot.
 * @param {(message: string) => void} warn - status logger.
 * @returns {any} the value, with unusable consolidation routes removed.
 */
export function sanitizeConfig(value, warn) {
  const routes = value?.consolidation?.models
  if (!Array.isArray(routes) || routes.length === 0) return value
  const seen = new Set()
  const kept = []
  for (const route of routes) {
    const provider = typeof route?.provider === 'string' ? route.provider.trim() : ''
    const model = typeof route?.model === 'string' ? route.model.trim() : ''
    if (provider === '' || model === '') {
      warn('dropping a consolidation model route with a blank provider or model')
      continue
    }
    const key = `${provider}\0${model}`
    if (seen.has(key)) {
      warn(`dropping duplicate consolidation model route ${provider}/${model}`)
      continue
    }
    seen.add(key)
    kept.push(route)
  }
  if (kept.length === routes.length) return value
  return { ...value, consolidation: { ...value.consolidation, models: kept } }
}

/**
 * Fields the collector reads once, when it starts. A committed change to any of
 * them restarts the collector (index.js) instead of waiting for a profile
 * reload — which is also what makes the one-time import of a pre-0.1.7
 * `settings.yaml` take effect: dsh imports it after every entry has started, as
 * a live update.
 */
export const STARTUP_FIELDS = ['enabled', 'shelf', 'autoApprove']

/**
 * Wrap the plugin's resolved cordis config in the live handle the rest of the
 * Host half reads. The snapshot is taken, sanitized and cached once per
 * committed change (volatile references are updated in place by the Loader,
 * which then emits `loader/volatile-update`), so `get()` is a field read on the
 * hot paths and `onChange` listeners see the very value `get()` returns.
 * Sanitizer warnings are reported once per committed value.
 *
 * Must be created on the plugin entry's own context: `loader/volatile-update`
 * is an instance-local event, so a child fiber's context never sees it.
 *
 * @param {import('@deepseek-ai/cordis').Context} ctx - the entry fiber's context.
 * @param {any} config - the resolved config `apply` received.
 * @param {(message: string) => void} [warn] - status logger for sanitizer drops.
 * @returns {{get: () => any, onChange: (cb: (value: any) => void) => () => void}}
 */
export function installConfig(ctx, config, warn = () => {}) {
  const listeners = new Set()
  const read = () => {
    const warned = new Set()
    return sanitizeConfig(snapshotConfig(config), (message) => {
      if (warned.has(message)) return
      warned.add(message)
      warn(message)
    })
  }
  let current = read()
  ctx.on('loader/volatile-update', () => {
    current = read()
    for (const cb of listeners) {
      try {
        cb(current)
      } catch {
        // A status listener must never break configuration commits.
      }
    }
  })
  return {
    get: () => current,
    onChange(cb) {
      listeners.add(cb)
      return () => listeners.delete(cb)
    },
  }
}
