/**
 * One-time sandbox whitelist authorization for `~/.hypatia`.
 *
 * Hypatia's CLI writes its store under `~/.hypatia`, but the deployment-level
 * sandbox-policy only allows that path when a user (or their setup recipe) has
 * added it to `additionalWritableRoots`. Without it every hypatia write stops
 * on an approval prompt — which is exactly the failure this plugin's
 * auto-approve cannot fix, because the write happens inside the sandboxed bash
 * tool, not at the approval waterfall.
 *
 * This module closes the gap at activation time:
 *
 *   1. Read every candidate patch file (user layer, then each profile's user
 *      patch layer) and check whether any `sandbox-policy` entry already lists
 *      the root. If it does, do nothing — the check is idempotent.
 *   2. Otherwise arm a ONE-SHOT hook on the first `tools/pre-execute` of any
 *      session. That event only fires inside an open turn, which is the
 *      lifecycle the approval service requires (`approval/request` outside a
 *      turn is rejected as crash-tail garbage, see dsh-user-approval). The ask
 *      is a REAL prompt: nothing auto-answers it — the waterfall forwards it to
 *      the client UI and blocks on the human.
 *   3. On `allowed-once`, splice the root into the patch file that already
 *      carries the `sandbox-policy` entry (or append a complete entry to the
 *      user layer when no entry exists). The splice is textual, not a YAML
 *      round-trip: cordis patch files are hand-maintained and a re-dump would
 *      destroy comments and reformat every other entry. `!!js` expression tags
 *      survive untouched because they are copied verbatim.
 *
 * The write itself is an ordinary host-side file write, so it needs no second
 * approval — the prompt above IS the informed consent, and the durable patch
 * line is its visible, revertible artifact.
 *
 * Windows is skipped entirely: the local sandbox's ACL backend does not
 * support `additionalWritableRoots` (dsh-sandbox-local), so a whitelist line
 * would be dead config there.
 *
 * Ported from dsh-hypatia-auto-memory 2.4.4 (same logic; only the module
 * identity and status plumbing differ, see index.js).
 *
 * @module dsh-hypatia/sandbox-whitelist
 */

import { homedir } from 'node:os'
import { join, resolve } from 'node:path'
import { readFileSync, writeFileSync, readdirSync, statSync } from 'node:fs'
import * as yaml from 'js-yaml'

/** Same scalar tag dialect cordis-plugin-include uses for patch files. */
const JsExpr = new yaml.Type('tag:yaml.org,2002:js', {
  kind: 'scalar',
  resolve: (data) => typeof data === 'string',
  construct: (data) => ({ __jsExpr: data }),
})
const PATCH_SCHEMA = yaml.JSON_SCHEMA.extend(JsExpr)

/** The root this plugin wants whitelisted. */
export function defaultHypatiaRoot(home = homedir()) {
  return join(home, '.hypatia')
}

/**
 * Path equality with normalization (and Windows case-folding). Symbolic links
 * are intentionally NOT resolved: patch files name real paths, and a link
 * check would force a filesystem probe on every activation.
 */
export function samePath(a, b) {
  if (typeof a !== 'string' || typeof b !== 'string') return false
  const left = resolve(a)
  const right = resolve(b)
  if (process.platform === 'win32') return left.toLowerCase() === right.toLowerCase()
  return left === right
}

/** Extract the plain-string roots of one parsed `sandbox-policy` entry. */
function entryRoots(entry) {
  const roots = entry?.id === 'sandbox-policy' ? entry.config?.additionalWritableRoots : undefined
  if (!Array.isArray(roots)) return []
  return roots.filter((root) => typeof root === 'string' && root.length > 0)
}

/**
 * Parse patch texts and return every `additionalWritableRoots` string declared
 * by a `sandbox-policy` entry. Unparsable texts are skipped (a broken user
 * patch must not crash activation); `!!js` expression values are skipped too —
 * only literal entries can be compared here, and an expression-valued root is
 * the deployment's own business.
 */
export function collectDeclaredRoots(patchTexts) {
  const roots = []
  for (const text of patchTexts) {
    let entries
    try {
      entries = yaml.load(text, { schema: PATCH_SCHEMA })
    } catch {
      continue
    }
    if (!Array.isArray(entries)) continue
    for (const entry of entries) roots.push(...entryRoots(entry))
  }
  return roots
}

/** True when any parsed patch already whitelists `root` (normalized compare). */
export function isWhitelisted(patchTexts, root) {
  return collectDeclaredRoots(patchTexts).some((declared) => samePath(declared, root))
}

/* -------------------------------------------------------------------------- */
/* Textual splice                                                              */
/* -------------------------------------------------------------------------- */

/**
 * Splice `root` into the `additionalWritableRoots` list of the FIRST
 * `sandbox-policy` entry in `text`. Returns the new text, or `null` when the
 * text has no `sandbox-policy` entry (caller appends a full block instead).
 *
 * The splice only ever ADDS one line (or a two-line key + item pair), so every
 * comment, fold and `!!js` expression in the file is preserved byte-for-byte.
 */
export function insertRootIntoPatchText(text, root) {
  const entryMatch = /^[ \t]*- id: sandbox-policy[ \t]*$/m.exec(text)
  if (entryMatch === null) return null
  const entryIndent = entryMatch[0].match(/^[ \t]*/)[0]
  const childIndent = `${entryIndent}  `
  const itemIndent = `${childIndent}  `

  // The entry block ends at the next line dedented to entryIndent or shallower.
  const start = entryMatch.index + entryMatch[0].length
  const rest = text.slice(start)
  const end = rest.search(new RegExp(`^[ \\t]{0,${entryIndent.length}}(\\S)`, 'm'))
  const block = end === -1 ? rest : rest.slice(0, end)

  const rootsLine = /^([ \t]*)additionalWritableRoots:[ \t]*$/m.exec(block)
  if (rootsLine !== null) {
    // Existing list: match the first item's indentation (fall back to config + 4).
    const itemMatch = /^[ \t]*-[ \t]/m.exec(block.slice(rootsLine.index + rootsLine[0].length))
    const itemPad = itemMatch !== null ? itemMatch[0].match(/^[ \t]*/)[0] : `${itemIndent}`
    const insertAt = start + rootsLine.index + rootsLine[0].length
    return `${text.slice(0, insertAt)}\n${itemPad}- ${root}${text.slice(insertAt)}`
  }

  // No list yet: insert the key aligned with the entry's OTHER config keys
  // (`mode`, `workspaceRoot`), i.e. one level deeper than `config:`, and the
  // list item one level deeper still. If the shape differs, fall back right
  // after the entry header — a sandbox-policy entry always has `config:`.
  const configChild = `${childIndent}  `
  const itemPad = `${configChild}  `
  // Anchor: the LAST config-child line (keys with inline values like
  // `mode: !!js …` count; `config:` itself does not — it lives one level up).
  const childLines = [...block.matchAll(new RegExp(`^${escapeRegExp(configChild)}.+$`, 'gm'))]
  const lastChild = childLines[childLines.length - 1]
  const configLine = new RegExp(`^${escapeRegExp(childIndent)}config:[ \\t]*$`, 'm').exec(block)
  const anchorMatch = lastChild ?? configLine
  const anchor = anchorMatch !== null ? start + anchorMatch.index + anchorMatch[0].length : start
  const keyPad = lastChild !== undefined ? configChild : childIndent
  return `${text.slice(0, anchor)}\n${keyPad}additionalWritableRoots:\n${itemPad}- ${root}${text.slice(anchor)}`
}

/** The complete user-layer `sandbox-policy` entry carrying exactly `roots`. */
export function sandboxPolicyBlock(roots) {
  const items = roots.map((root) => `      - ${root}`).join('\n')
  return [
    '- id: sandbox-policy',
    "  name: '@deepseek-ai/dsh-sandbox-policy'",
    '  config:',
    "    mode: !!js process.env.DSH_PERMISSION_MODE ?? 'workspace-write'",
    '    workspaceRoot: !!js process.cwd()',
    '    additionalWritableRoots:',
    items,
  ].join('\n')
}

/**
 * Append a complete `sandbox-policy` entry to a patch text, keeping blank-line
 * separation tidy.
 */
export function appendPolicyBlock(text, root) {
  const block = sandboxPolicyBlock([root])
  const trimmed = text.replace(/\s+$/, '')
  return `${trimmed}\n\n${block}\n`
}

function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/* -------------------------------------------------------------------------- */
/* File discovery + write                                                      */
/* -------------------------------------------------------------------------- */

/**
 * Candidate user-writable patch files, most preferred first: the bare user
 * layer, then every profile's user patch layer (sorted for determinism).
 * Missing files are skipped by the caller's read wrapper, not here.
 */
export function candidatePatchFiles(home) {
  const files = [join(home, 'cordis.patch.yml')]
  const profilesDir = join(home, 'profiles')
  let profiles = []
  try {
    profiles = readdirSorted(profilesDir)
  } catch {
    // No profiles directory: the bare user layer is the only candidate.
  }
  for (const profile of profiles) files.push(join(profilesDir, profile, 'cordis.patch.yml'))
  return files
}

function readdirSorted(dir) {
  return readdirSync(dir).filter((name) => {
    try { return statSync(join(dir, name)).isDirectory() } catch { return false }
  }).sort()
}

/**
 * Read every candidate patch file that exists and is non-empty.
 * @returns {{ path: string, text: string }[]}
 */
export function readPatchFiles(files) {
  const loaded = []
  for (const path of files) {
    let text
    try {
      text = readFileSync(path, 'utf8')
    } catch {
      continue
    }
    if (text.trim().length > 0) loaded.push({ path, text })
  }
  return loaded
}

/**
 * Decide where the root goes and produce the new file content.
 *
 * Preference: a file whose text already declares a `sandbox-policy` entry with
 * an `additionalWritableRoots` list (that line is the deployment's established
 * home for extra roots); then any file with a `sandbox-policy` entry at all;
 * finally the FIRST candidate (the bare user layer), which gets a fresh block.
 *
 * @returns {{ path: string, text: string } | null} null when nothing is writable.
 */
export function planPatchWrite(loadedFiles, root, files) {
  const withList = loadedFiles.find((file) => /additionalWritableRoots:/.test(slicePolicyBlock(file.text)))
  if (withList !== undefined) {
    return { path: withList.path, text: insertRootIntoPatchText(withList.text, root) }
  }
  const withEntry = loadedFiles.find((file) => insertRootIntoPatchText(file.text, root) !== null)
  if (withEntry !== undefined) {
    return { path: withEntry.path, text: insertRootIntoPatchText(withEntry.text, root) }
  }
  const fallback = loadedFiles[0]
  if (fallback !== undefined) return { path: fallback.path, text: appendPolicyBlock(fallback.text, root) }
  const fresh = (files ?? candidatePatchFiles(process.env.DSH_HOME ?? homedir()))[0]
  if (fresh === undefined) return null
  return { path: fresh, text: `${sandboxPolicyBlock([root])}\n` }
}

function slicePolicyBlock(text) {
  const match = /^[ \t]*- id: sandbox-policy[ \t]*$/m.exec(text)
  if (match === null) return ''
  const start = match.index + match[0].length
  const rest = text.slice(start)
  const entryIndent = match[0].match(/^[ \t]*/)[0].length
  const end = rest.search(new RegExp(`^[ \\t]{0,${entryIndent}}(\\S)`, 'm'))
  return end === -1 ? rest : rest.slice(0, end)
}

/**
 * Ask the human once, and on approval splice the root into the patch layer.
 *
 * @param {object} deps
 * @param {import('@deepseek-ai/cordis').Context} deps.ctx - a context that sees
 *   `tools/pre-execute` (any scope the agent events bubble through) and can
 *   reach the approval service.
 * @param {(req: object) => Promise<'allowed-once' | 'rejected' | 'cancelled' | 'unavailable'>} deps.requestApproval
 * @param {{ root?: string, home?: string, status?: { log?: (msg: string) => void, warn?: (msg: string) => void } }} deps.options
 * @returns {boolean} whether a prompt was armed (false = already whitelisted, nothing to do).
 */
export function armSandboxWhitelistPrompt({ ctx, requestApproval, options = {} }) {
  const root = options.root ?? defaultHypatiaRoot()
  const home = options.home ?? process.env.DSH_HOME ?? homedir()
  const status = options.status ?? {}

  if (process.platform === 'win32') {
    status.warn?.('sandbox whitelist: skipped on win32 (the local sandbox ACL backend does not support additionalWritableRoots)')
    return false
  }

  const loaded = readPatchFiles(candidatePatchFiles(home))
  if (isWhitelisted(loaded.map((file) => file.text), root)) return false

  // Exactly one ask per activation, no matter how many tool calls race in.
  let armed = false
  let off = null
  const onPreExecute = (exec) => {
    if (armed) return
    armed = true
    if (typeof off === 'function') off()
    void askAndWrite({ exec, requestApproval, root, home, status, patchFiles: candidatePatchFiles(home) })
      .catch((error) => status.warn?.(`sandbox whitelist: ${String(error)}`))
  }
  if (typeof ctx.on !== 'function') return false
  const disposer = ctx.on('tools/pre-execute', onPreExecute)
  off = typeof disposer === 'function' ? disposer : null
  return true
}

/**
 * The ask itself, split from arming so tests can drive it without cordis.
 */
export async function askAndWrite({ exec, requestApproval, root, home, status, patchFiles }) {
  const reason = [
    `Hypatia needs to write its memory store under ${root}.`,
    'Approving adds this ONE line to your DSH patch config (deployment-level: it whitelists the directory for ALL sessions, not just this one):',
    `  sandbox-policy → additionalWritableRoots: [${root}]`,
    `Revoke anytime by deleting the entry from ${home}/cordis.patch.yml or profiles/*/cordis.patch.yml.`,
    '',
    `Hypatia 需要写入其记忆存储目录 ${root}。批准后将在 DSH 补丁配置中加入一行 sandbox-policy additionalWritableRoots（部署级：对**所有会话**生效）。可随时从 ${home} 下的 cordis.patch.yml 删除该条目以撤销。`,
  ].join('\n')

  const outcome = await requestApproval({
    agent: exec?.agent,
    toolName: 'hypatia/sandbox-whitelist',
    reason,
    signal: exec?.signal,
  })
  if (outcome !== 'allowed-once') {
    status.log?.(`sandbox whitelist: not approved (${outcome}); will ask again on next activation`)
    return false
  }

  const files = patchFiles ?? candidatePatchFiles(home)
  const loaded = readPatchFiles(files)
  const plan = planPatchWrite(loaded, root, files)
  if (plan === null) throw new Error('no writable patch file found')
  writeFileSync(plan.path, plan.text, 'utf8')
  status.log?.(`sandbox whitelist: added ${root} to ${plan.path}; takes effect on next deployment start`)
  return true
}
