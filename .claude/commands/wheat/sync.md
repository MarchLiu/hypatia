# /sync — Publish sprint artifacts to external platforms

You are publishing the current Wheat sprint's artifacts to external platforms. Read CLAUDE.md for sprint context and claims.json for the current state.

## Process

1. **Compile first**: Run `wheat compile --summary` to ensure compilation is fresh. If compilation is blocked, stop and tell the user to fix issues first.

2. **Detect available targets**: Check which MCP servers are available:

   - **Confluence** (via Atlassian MCP): Push the compiled brief as a Confluence page. Use `mcp__atlassian__confluence_create_page` or `mcp__atlassian__confluence_update_page`.
   - **Slack** (via Slack MCP): Post a sprint summary to a channel. Use `mcp__slack__send_message`.
   - **Notion** (via Notion MCP): Create a page with the brief content.
   - **Local file export**: Always available — write to `output/` directory.

3. **Confluence publish** (primary target):

   - Read `compilation.json` for the certified output
   - Read `output/brief.html` if it exists, otherwise generate the brief first via `/brief`
   - Convert the brief to Confluence Storage Format (XHTML subset):
     - Strip `<script>` and `<style>` tags
     - Convert inline styles to Confluence macros where possible
     - Wrap code blocks in `<ac:structured-macro ac:name="code">`
     - Preserve tables, headings, lists as-is (Confluence supports standard HTML)
   - Create or update the page:
     - Page title: `[Sprint] <question summary>` (truncated to 255 chars)
     - Space key: from user argument or CLAUDE.md connector config
     - Add labels: `wheat-sprint`, `<phase>`, claim type tags
   - Add a comment with sync metadata: timestamp, claims hash from compilation certificate, claim count

4. **Build sync manifest**: After publishing, append a sync record to `output/sync-log.json`:

   ```json
   {
     "timestamp": "<ISO>",
     "target": "confluence|slack|notion|file",
     "url": "<published URL or file path>",
     "claims_hash": "<from compilation.json>",
     "claims_count": <number>,
     "phase": "<current phase>"
   }
   ```

5. **Add a sync claim**: Append a claim recording the sync event:

   ```json
   {
     "id": "r<next>",
     "type": "factual",
     "topic": "sync",
     "content": "Sprint artifacts published to <target> at <url>. Claims hash: <hash>. <count> active claims at time of sync.",
     "source": {
       "origin": "mcp",
       "artifact": "<url>",
       "connector": "<target>"
     },
     "evidence": "documented",
     "status": "active",
     "phase_added": "<phase>",
     "timestamp": "<ISO>",
     "conflicts_with": [],
     "resolved_by": null,
     "tags": ["sync", "<target>"]
   }
   ```

6. **Compile again** after adding the sync claim.

## Arguments

- No argument: auto-detect available targets, prefer Confluence
- `confluence <space-key>`: publish to specific Confluence space
- `slack <channel>`: post summary to Slack channel
- `notion <page-id>`: publish to Notion page
- `file`: export to `output/` only (always works, no MCP needed)
- `--dry-run`: show what would be published without actually pushing

## Error handling

- If no MCP servers are configured, fall back to local file export and tell the user how to configure Atlassian MCP: `claude mcp add atlassian -- npx @anthropic-ai/atlassian-mcp`
- If Confluence API returns 403, suggest checking OAuth scopes
- If the page already exists, update it (don't create duplicates) — search by title first

## Next steps suggestions

After sync completes, suggest:

```
Next steps:
  /status              -- verify sync claim was recorded
  /sync slack <channel> -- cross-post to Slack
  /pull confluence      -- backfill claims from Confluence comments
  /brief               -- regenerate brief if claims changed
```
