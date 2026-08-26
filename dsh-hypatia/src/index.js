/**
 * dsh-hypatia — Hypatia AI memory integration for DSH.
 *
 * One host plugin, two independent capabilities (each can be disabled via
 * config, and each waits only for the services it actually needs):
 *
 * 1. **Auto-approve** (`autoApprove`, default on): DSH's approval seam grants
 *    are one-shot (`allowed-once`) and `ApprovalRequest` deliberately carries
 *    NO tool arguments (only toolName, callId, reason). This part correlates
 *    two extension points by call id:
 *      - `tools/pre-execute` sees the full bash call (name + parsed
 *        arguments). A call whose command is a pure hypatia invocation is
 *        remembered by callId.
 *      - `approval/request` fires later, inside the same call's execution,
 *        when the bash tool asks to escalate the sandbox (hypatia writes to
 *        `~/.hypatia/`, outside any session workspace). A remembered callId
 *        is answered `allowed-once`; everything else falls through to the
 *        human answerer unchanged — all non-hypatia operations keep the
 *        normal interactive approval prompt.
 *
 *    "Pure" means: the invoked binary is `hypatia` (basename match, so
 *    absolute paths like `/usr/local/bin/hypatia` qualify), optionally
 *    preceded by `KEY=VALUE` env assignments, and the command contains no
 *    unquoted shell composition (`&&`, `;`, pipes, redirections, command
 *    substitution), so `hypatia ... && rm -rf x` still prompts. Quoted
 *    payloads (JSON data, JSE queries like '["$knowledge"]') are scanned
 *    quote-aware and never blocked.
 *
 * 2. **Bundled skills** (`skills`, default on): registers the packaged
 *    `hypatia` and `hypatia-memory` skills into `ctx.skills` as runtime
 *    skills, so installing this plugin alone gives every session the full
 *    Hypatia capability set — no separate skill installation. Runtime
 *    entries outrank user-level (`~/.dsh/skills`) copies of the same name,
 *    so the plugin's versions win when both exist.
 *
 * Config (all optional):
 *   binaries:    string[] — trusted CLI basenames, default ['hypatia']
 *   autoApprove: boolean  — default true
 *   skills:      boolean  — default true
 *   skillsDir:   string   — override the packaged skills/ directory
 *
 * @module dsh-hypatia
 */

import { existsSync, readFileSync, readdirSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const PACKAGE_ROOT = dirname(dirname(fileURLToPath(import.meta.url)))
const DEFAULT_BINARIES = ['hypatia']

export const name = 'dsh-hypatia'

/* ------------------------------------------------------------------------ */
/* Auto-approve                                                              */
/* ------------------------------------------------------------------------ */

/** True when the command's executable word (after env assignments) is trusted. */
function invokesTrustedBinary(command, binaries) {
  let rest = command.trimStart()
  // Skip leading KEY=VALUE environment assignments (e.g. HYPATIA_BIN=...).
  for (;;) {
    const assignment = /^[A-Za-z_][A-Za-z0-9_]*=\S+\s+/.exec(rest)
    if (!assignment) break
    rest = rest.slice(assignment[0].length).trimStart()
  }
  const word = /^([^\s"']+)/.exec(rest)?.[1]
  if (!word) return false
  return binaries.some((bin) => word === bin || word.endsWith(`/${bin}`))
}

/** True when the command contains shell composition OUTSIDE quotes. */
function hasShellComposition(command) {
  let single = false
  let double = false
  let escaped = false
  for (let index = 0; index < command.length; index += 1) {
    const char = command[index]
    if (escaped) {
      escaped = false
      continue
    }
    if (!single && char === '\\') {
      escaped = true
      continue
    }
    if (!double && char === "'") {
      single = !single
      continue
    }
    if (!single && char === '"') {
      double = !double
      continue
    }
    if (single || double) continue
    if (char === '&' || char === ';' || char === '|' || char === '<' || char === '>' || char === '`' || char === '\n') {
      return true
    }
    if (char === '$' && command[index + 1] === '(') return true
  }
  return false
}

function isPureTrustedCall(exec, binaries) {
  if (exec.name !== 'bash') return false
  const command = exec.arguments?.command
  if (typeof command !== 'string') return false
  return invokesTrustedBinary(command, binaries) && !hasShellComposition(command)
}

function applyAutoApprove(ctx, config) {
  const binaries = Array.isArray(config?.binaries) && config.binaries.length > 0 ? config.binaries : DEFAULT_BINARIES
  /** callIds of in-flight bash calls that qualify for automatic approval. */
  const pending = new Set()

  ctx.on('tools/pre-execute', (exec, next) => {
    try {
      if (isPureTrustedCall(exec, binaries)) pending.add(exec.callId)
    } catch {
      // A matcher bug must never block the tool pipeline; fall through to allow.
    }
    return next()
  })

  // Whether the call asked or not, drop the marker once the call settles.
  ctx.on('tools/result', (exec) => {
    pending.delete(exec.callId)
  })

  ctx.on('approval/request', (req, next) => {
    if (req.callId !== undefined && pending.delete(req.callId)) {
      return 'allowed-once'
    }
    return next()
  })
}

/* ------------------------------------------------------------------------ */
/* Bundled skills                                                            */
/* ------------------------------------------------------------------------ */

/**
 * Minimal YAML-frontmatter reader for flat `key: value` pairs (single-line
 * values only — the grammar every shipped SKILL.md uses). Double quotes
 * around a value are stripped.
 */
function parseFrontmatter(raw) {
  const match = /^---\r?\n([\s\S]*?)\r?\n---\s*/.exec(raw)
  if (!match) return { data: {}, content: raw }
  const data = {}
  for (const line of match[1].split(/\r?\n/)) {
    const kv = /^([A-Za-z0-9_-]+):\s*(.*)$/.exec(line)
    if (kv) data[kv[1]] = kv[2].replace(/^"(.*)"$/, '$1')
  }
  return { data, content: raw.slice(match[0].length) }
}

function warn(ctx, message, error) {
  try {
    ctx.logger('dsh-hypatia').warn(`${message}: ${error instanceof Error ? error.message : String(error)}`)
  } catch {
    // Logging must never take the plugin down.
  }
}

function applySkills(ctx, config) {
  const skillsDir = config?.skillsDir !== undefined ? resolve(config.skillsDir) : join(PACKAGE_ROOT, 'skills')
  if (!existsSync(skillsDir)) {
    warn(ctx, `skills directory not found: ${skillsDir}`, new Error('missing'))
    return
  }
  for (const entry of readdirSync(skillsDir, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue
    const dir = join(skillsDir, entry.name)
    const file = join(dir, 'SKILL.md')
    if (!existsSync(file)) continue
    try {
      const { data, content } = parseFrontmatter(readFileSync(file, 'utf8'))
      if (typeof data.name !== 'string' || data.name === '') throw new Error('frontmatter missing `name`')
      if (typeof data.description !== 'string' || data.description === '') throw new Error('frontmatter missing `description`')
      ctx.skills.register({
        name: data.name,
        description: data.description,
        content,
        path: file,
        source: 'bundled',
        provider: 'dsh-hypatia',
        resourceBase: { kind: 'directory', path: dir },
        invocation: {
          modelInvocable: true,
          userInvocable: data['user-invocable'] !== 'false',
        },
      })
    } catch (error) {
      warn(ctx, `failed to register bundled skill from ${file}`, error)
    }
  }
}

/* ------------------------------------------------------------------------ */
/* Composition                                                               */
/* ------------------------------------------------------------------------ */

export function apply(ctx, config = {}) {
  // Each half is its own sub-plugin with its own service requirements, so a
  // deployment without the skills registry still gets auto-approval (and
  // vice versa), and neither blocks plugin load waiting for an absent
  // service.
  if (config.autoApprove !== false) {
    ctx.plugin({ name: 'dsh-hypatia/auto-approve', inject: ['approval', 'tools'], apply: applyAutoApprove }, config)
  }
  if (config.skills !== false) {
    ctx.plugin({ name: 'dsh-hypatia/skills', inject: ['skills'], apply: applySkills }, config)
  }
}
