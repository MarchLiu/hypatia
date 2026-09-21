import assert from 'node:assert/strict'
import { test } from 'node:test'
import { hasShellComposition, invokesTrustedBinary } from '../src/index.js'

const BINARIES = ['hypatia']

// Regression: bash expands command substitution inside double quotes, so a
// double-quoted payload must count as composition and fall through to the
// human approval prompt (auto-approve bypass fixed here).
test('double-quoted $() is composition', () => {
  assert.equal(hasShellComposition('hypatia search "$(touch /tmp/pwn)"'), true)
  assert.equal(hasShellComposition('hypatia query "x$(rm -rf ~)y"'), true)
})

test('double-quoted backticks are composition', () => {
  assert.equal(hasShellComposition('hypatia query "x`touch /tmp/pwn`y"'), true)
})

test('single-quoted payloads stay literal (no composition)', () => {
  assert.equal(hasShellComposition(`hypatia search '["$knowledge"]'`), false)
  assert.equal(hasShellComposition(`hypatia query '$(not expanded)'`), false)
  assert.equal(hasShellComposition("hypatia query '`not expanded`'"), false)
})

test('unquoted composition is still composition', () => {
  assert.equal(hasShellComposition('hypatia search x && rm -rf /'), true)
  assert.equal(hasShellComposition('hypatia search x; rm -rf /'), true)
  assert.equal(hasShellComposition('hypatia search x | tee /tmp/x'), true)
  assert.equal(hasShellComposition('hypatia search `x`'), true)
  assert.equal(hasShellComposition('hypatia search $(x)'), true)
})

test('plain hypatia invocations are not composition', () => {
  assert.equal(hasShellComposition('hypatia query "find my notes"'), false)
  assert.equal(hasShellComposition("hypatia search 'knowledge graph'"), false)
})

test('trusted binary matching is unchanged', () => {
  assert.equal(invokesTrustedBinary('hypatia query x', BINARIES), true)
  assert.equal(invokesTrustedBinary('/usr/local/bin/hypatia query x', BINARIES), true)
  assert.equal(invokesTrustedBinary('HOME=/work hypatia query x', BINARIES), true)
  assert.equal(invokesTrustedBinary('evil query x', BINARIES), false)
  assert.equal(invokesTrustedBinary('evihypatia query x', BINARIES), false)
})
