/**
 * Unit tests for dsh-hypatia/sandbox-whitelist.
 *
 * Pure-core coverage: detection (including `!!js`-tagged patch files), path
 * comparison, the textual splice (existing list, missing list, appended
 * block), write planning across candidate files, and the once-only ask flow.
 * All file I/O happens inside a temporary directory simulating a profile tree.
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  defaultHypatiaRoot,
  samePath,
  collectDeclaredRoots,
  isWhitelisted,
  insertRootIntoPatchText,
  appendPolicyBlock,
  planPatchWrite,
  candidatePatchFiles,
  readPatchFiles,
  armSandboxWhitelistPrompt,
  askAndWrite,
} from '../src/sandbox-whitelist.js'

const ENTRY_WITH_LIST = `# my hand-maintained patch file — keep comments!
- id: some-plugin
  name: some-package
  config:
    skin-custom-theme:
      applied: false
- id: sandbox-policy
  name: '@deepseek-ai/dsh-sandbox-policy'
  config:
    mode: !!js process.env.DSH_PERMISSION_MODE ?? 'workspace-write'
    workspaceRoot: !!js process.cwd()
    additionalWritableRoots:
      - /Users/someone/.other
`

const ENTRY_WITHOUT_LIST = `# user patch
- id: hypatia-auto-memory
  name: dsh-hypatia-auto-memory
  config:
    autoApprove: true
- id: sandbox-policy
  name: '@deepseek-ai/dsh-sandbox-policy'
  config:
    mode: !!js process.env.DSH_PERMISSION_MODE ?? 'workspace-write'
    workspaceRoot: !!js process.cwd()
`

const NO_ENTRY = `# plain patch, no sandbox-policy
- id: hypatia-auto-memory
  name: dsh-hypatia-auto-memory
  config:
    autoApprove: true
`

test('defaultHypatiaRoot joins ~/.hypatia', () => {
  assert.equal(defaultHypatiaRoot('/home/u'), join('/home/u', '.hypatia'))
})

test('samePath normalizes trailing separators and dot segments', () => {
  const root = defaultHypatiaRoot('/home/u')
  assert.ok(samePath(root, `${root}/`))
  assert.ok(samePath(root, `${root}/./x/..`))
  assert.ok(!samePath(root, '/home/u/.hypatia2'))
  assert.ok(!samePath(root, undefined))
})

test('collectDeclaredRoots reads real patch text with !!js tags and skips junk', () => {
  const roots = collectDeclaredRoots([ENTRY_WITH_LIST, '!!!not yaml', NO_ENTRY])
  assert.deepEqual(roots, ['/Users/someone/.other'])
})

test('isWhitelisted is true for the exact root, false for siblings', () => {
  const texts = [ENTRY_WITH_LIST]
  assert.ok(isWhitelisted(texts, '/Users/someone/.other'))
  assert.ok(!isWhitelisted(texts, '/Users/someone/.other2'))
})

test('insertRootIntoPatchText appends to an existing list, preserving bytes elsewhere', () => {
  const root = defaultHypatiaRoot('/Users/someone')
  const next = insertRootIntoPatchText(ENTRY_WITH_LIST, root)
  assert.ok(next !== null)
  assert.ok(next.includes('      - /Users/someone/.hypatia'))
  // Comments and the untouched entry survive verbatim.
  assert.ok(next.includes('# my hand-maintained patch file — keep comments!'))
  assert.ok(next.includes('- id: some-plugin'))
  // Round-trip parse: both roots now declared.
  assert.ok(isWhitelisted([next], root))
})

test('insertRootIntoPatchText adds the additionalWritableRoots key when absent', () => {
  const root = defaultHypatiaRoot('/Users/someone')
  const next = insertRootIntoPatchText(ENTRY_WITHOUT_LIST, root)
  assert.ok(next !== null)
  assert.ok(next.includes('    additionalWritableRoots:'))
  assert.ok(next.includes('      - /Users/someone/.hypatia'))
  assert.ok(next.includes("    mode: !!js process.env.DSH_PERMISSION_MODE ?? 'workspace-write'"))
  assert.ok(isWhitelisted([next], root))
  // The other entry was not touched.
  assert.ok(next.includes('- id: hypatia-auto-memory'))
})

test('insertRootIntoPatchText returns null without an entry', () => {
  assert.equal(insertRootIntoPatchText(NO_ENTRY, '/x'), null)
})

test('appendPolicyBlock creates a well-formed entry a parser accepts', () => {
  const root = defaultHypatiaRoot('/Users/someone')
  const next = appendPolicyBlock(NO_ENTRY, root)
  assert.ok(isWhitelisted([next], root))
  assert.ok(next.includes("    mode: !!js process.env.DSH_PERMISSION_MODE ?? 'workspace-write'"))
  assert.ok(next.includes('    workspaceRoot: !!js process.cwd()'))
})

test('candidatePatchFiles + readPatchFiles walk a simulated home', () => {
  const home = mkdtempSync(join(tmpdir(), 'hypatia-sbx-'))
  try {
    mkdirSync(join(home, 'profiles', 'web'), { recursive: true })
    mkdirSync(join(home, 'profiles', 'empty'))
    writeFileSync(join(home, 'profiles', 'web', 'cordis.patch.yml'), ENTRY_WITH_LIST)
    const files = candidatePatchFiles(home)
    assert.deepEqual(files, [
      join(home, 'cordis.patch.yml'),
      join(home, 'profiles', 'empty', 'cordis.patch.yml'),
      join(home, 'profiles', 'web', 'cordis.patch.yml'),
    ])
    const loaded = readPatchFiles(files)
    assert.equal(loaded.length, 1)
    assert.equal(loaded[0].path, join(home, 'profiles', 'web', 'cordis.patch.yml'))
  } finally {
    rmSync(home, { recursive: true, force: true })
  }
})

test('planPatchWrite prefers the file whose sandbox-policy already has a list', () => {
  const root = defaultHypatiaRoot('/Users/someone')
  const loaded = [
    { path: '/a/cordis.patch.yml', text: NO_ENTRY },
    { path: '/b/cordis.patch.yml', text: ENTRY_WITH_LIST },
  ]
  const plan = planPatchWrite(loaded, root)
  assert.equal(plan.path, '/b/cordis.patch.yml')
  assert.ok(isWhitelisted([plan.text], root))
})

test('planPatchWrite falls back to an entry without a list, then to a fresh block', () => {
  const root = defaultHypatiaRoot('/Users/someone')
  const plan = planPatchWrite([{ path: '/a/cordis.patch.yml', text: ENTRY_WITHOUT_LIST }], root)
  assert.equal(plan.path, '/a/cordis.patch.yml')
  assert.ok(isWhitelisted([plan.text], root))

  const fresh = planPatchWrite([{ path: '/a/cordis.patch.yml', text: NO_ENTRY }], root, ['/a/cordis.patch.yml'])
  assert.equal(fresh.path, '/a/cordis.patch.yml')
  assert.ok(isWhitelisted([fresh.text], root))
  assert.ok(fresh.text.includes('- id: sandbox-policy'))
})

test('armSandboxWhitelistPrompt: whitelisted deployments arm nothing', () => {
  const home = mkdtempSync(join(tmpdir(), 'hypatia-sbx-'))
  try {
    writeFileSync(join(home, 'cordis.patch.yml'), ENTRY_WITH_LIST)
    let asked = 0
    const armed = armSandboxWhitelistPrompt({
      ctx: { on: () => () => {} },
      requestApproval: async () => { asked += 1; return 'allowed-once' },
      options: { root: '/Users/someone/.other', home, status: {} },
    })
    assert.equal(armed, false)
    assert.equal(asked, 0)
  } finally {
    rmSync(home, { recursive: true, force: true })
  }
})

test('armSandboxWhitelistPrompt: asks exactly once on the first tool call', async () => {
  const home = mkdtempSync(join(tmpdir(), 'hypatia-sbx-'))
  try {
    writeFileSync(join(home, 'cordis.patch.yml'), NO_ENTRY)
    let handlers = []
    const ctx = { on: (event, handler) => { handlers.push(handler); return () => {} } }
    let asked = 0
    const armed = armSandboxWhitelistPrompt({
      ctx,
      requestApproval: async () => { asked += 1; return 'rejected' },
      options: { root: '/Users/someone/.hypatia', home, status: { log: () => {} } },
    })
    assert.equal(armed, true)
    assert.equal(handlers.length, 1)
    // Two racing tool calls: the ask still happens once (rejected → no write).
    handlers[0]({ agent: { session: {} } })
    handlers[0]({ agent: { session: {} } })
    await new Promise((resolve) => setImmediate(resolve))
    assert.equal(asked, 1)
    assert.equal(readFileSync(join(home, 'cordis.patch.yml'), 'utf8'), NO_ENTRY)
  } finally {
    rmSync(home, { recursive: true, force: true })
  }
})

test('armSandboxWhitelistPrompt: approval writes the patch file (end-to-end on a fake home)', async () => {
  const home = mkdtempSync(join(tmpdir(), 'hypatia-sbx-'))
  try {
    mkdirSync(join(home, 'profiles', 'web'), { recursive: true })
    writeFileSync(join(home, 'profiles', 'web', 'cordis.patch.yml'), ENTRY_WITHOUT_LIST)
    let handlers = []
    const ctx = { on: (event, handler) => { handlers.push(handler); return () => {} } }
    const root = defaultHypatiaRoot('/Users/someone')
    const armed = armSandboxWhitelistPrompt({
      ctx,
      requestApproval: async () => 'allowed-once',
      options: { root, home, status: { log: () => {} } },
    })
    assert.equal(armed, true)
    handlers[0]({ agent: { session: {} } })
    await new Promise((resolve) => setImmediate(resolve))
    const after = readFileSync(join(home, 'profiles', 'web', 'cordis.patch.yml'), 'utf8')
    assert.ok(after.includes('      - /Users/someone/.hypatia'))
    assert.ok(isWhitelisted([after], root))
    // Idempotent: a second activation now arms nothing.
    const armed2 = armSandboxWhitelistPrompt({
      ctx: { on: () => () => {} },
      requestApproval: async () => { throw new Error('must not ask') },
      options: { root, home, status: {} },
    })
    assert.equal(armed2, false)
  } finally {
    rmSync(home, { recursive: true, force: true })
  }
})

test('askAndWrite surfaces the dual-language, revocation-aware prompt text', async () => {
  const home = mkdtempSync(join(tmpdir(), 'hypatia-sbx-'))
  const patchFile = join(home, 'cordis.patch.yml')
  writeFileSync(patchFile, NO_ENTRY)
  try {
    const seen = []
    const outcome = await askAndWrite({
      exec: { agent: { session: {} } },
      requestApproval: async (req) => { seen.push(req); return 'allowed-once' },
      root: '/Users/someone/.hypatia',
      home,
      status: {},
    })
    assert.equal(outcome, true)
    const req = seen[0]
    assert.equal(req.toolName, 'hypatia/sandbox-whitelist')
    assert.ok(req.reason.includes('/Users/someone/.hypatia'))
    assert.ok(req.reason.includes('ALL sessions'))
    assert.ok(req.reason.includes('撤销'))
    // The write landed with a fresh, parseable sandbox-policy entry.
    assert.ok(isWhitelisted([readFileSync(patchFile, 'utf8')], '/Users/someone/.hypatia'))
  } finally {
    rmSync(home, { recursive: true, force: true })
  }
})

test('askAndWrite with no writable candidates reports failure honestly', async () => {
  await assert.rejects(
    () => askAndWrite({
      exec: { agent: { session: {} } },
      requestApproval: async () => 'allowed-once',
      root: '/Users/someone/.hypatia',
      home: '/Users/someone/.dsh',
      patchFiles: [],
      status: {},
    }),
    /no writable patch file/,
  )
})
