# /pull — Backfill claims from external sources

You are pulling knowledge from external sources into the current Wheat sprint. Read CLAUDE.md for sprint context and claims.json for existing claims.

## Process

1. **Identify the source**: The user's argument tells you where to pull from. Supported sources:

   - **DeepWiki** (`deepwiki <github-org/repo>`): Pull architecture docs and dependency analysis from deepwiki.com
   - **Confluence** (`confluence <space-key>` or `confluence <page-url>`): Pull content from Confluence pages via Atlassian MCP
   - **Silo** (`silo <pack-name>`): Pull a knowledge pack via the silo MCP server
   - **GitHub** (`github <org/repo>`): Pull README, issues, discussions via GitHub MCP
   - **URL** (`url <url>`): Fetch and extract claims from any web page
   - No argument: scan CLAUDE.md connectors and pull from all configured sources

2. **Fetch content**: Use the appropriate MCP server or web fetch:

   - **DeepWiki**: Fetch `https://deepwiki.com/<org>/<repo>` — extract architecture overview, component descriptions, dependency graph, and key design decisions. DeepWiki auto-generates structured wiki docs from public GitHub repos. If the silo MCP is available, check `mcp__silo__silo_search` for cached DeepWiki content first.
   - **Confluence**: Use `mcp__atlassian__confluence_search` or `mcp__atlassian__confluence_get_page` to fetch page content
   - **Silo**: Use `mcp__silo__silo_pull` to retrieve the knowledge pack
   - **GitHub**: Use `mcp__github__search_repositories`, `mcp__github__get_file_contents`, `mcp__github__list_issues`
   - **URL**: Use web fetch to retrieve content

3. **Extract claims from content**: Parse the fetched content into typed claims:

   - Architecture descriptions → `factual` claims about system structure
   - Version numbers, metrics, stats → `factual` claims with evidence tier `documented` (for official sources) or `web` (for community content)
   - Known issues, limitations → `risk` claims
   - Best practices, recommendations → `recommendation` claims
   - Requirements, constraints mentioned → `constraint` claims
   - Estimates, projections → `estimate` claims

4. **Deduplicate against existing claims**: Before adding, check claims.json for:

   - Exact content matches (skip)
   - Semantic overlaps (flag as potential conflict with `conflicts_with`)
   - Claims that the new data supersedes (mark conflicts)

5. **Add claims**: Append new claims with IDs continuing the `r###` sequence. Set source appropriately:

   ```json
   {
     "source": {
       "origin": "mcp",
       "artifact": "<source URL or identifier>",
       "connector": "<deepwiki|confluence|silo|github|url>"
     },
     "evidence": "web|documented"
   }
   ```

   Use `"documented"` evidence for official docs, READMEs, and Confluence pages maintained by the project team. Use `"web"` for community-generated or AI-generated content (including DeepWiki).

6. **Compile**: Run `wheat compile --summary` after adding claims.

7. **Report**: Summarize what was pulled:
   - Number of new claims added
   - Number of duplicates skipped
   - Number of conflicts detected
   - Source attribution

## Arguments

- `deepwiki <org/repo>`: Pull from DeepWiki (e.g., `/pull deepwiki grainulation/wheat`)
- `confluence <space-key|page-url>`: Pull from Confluence
- `silo <pack-name>`: Pull from a Silo knowledge pack
- `github <org/repo>`: Pull from GitHub repo metadata
- `url <url>`: Pull from any web page
- No argument: pull from all connectors configured in CLAUDE.md
- `--max <n>`: Limit to n claims (default: 20)
- `--topic <slug>`: Scope pulled claims to a specific topic
- `--dry-run`: Show what would be added without modifying claims.json

## DeepWiki integration notes

DeepWiki (deepwiki.com) auto-generates structured documentation for any public GitHub repo. It provides:

- Architecture overviews with component descriptions
- Dependency graphs and data flow diagrams
- API documentation extracted from source code
- Design decision explanations grounded in actual code

To use: replace `github.com` with `deepwiki.com` in any repo URL. For self-hosted repos, DeepWiki-Open can be deployed internally.

When pulling from DeepWiki, prioritize:

1. Architecture claims (system structure, component relationships)
2. Dependency claims (what depends on what, version requirements)
3. API surface claims (public interfaces, protocols)
4. Design decision claims (why things are built a certain way)

## Next steps suggestions

After pull completes, suggest based on what was found:

```
Next steps:
  /research <topic>     -- deep dive on topics surfaced by pull
  /challenge <id>       -- verify pulled claims that seem surprising
  /witness <id> <url>   -- corroborate pulled claims from another source
  /blind-spot           -- check if pulled content reveals gaps
  /compile              -- recompile to see updated coverage
```
