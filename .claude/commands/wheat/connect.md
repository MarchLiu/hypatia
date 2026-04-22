# /connect — Link an external data source

You are connecting an external tool or data source to this Wheat sprint. Connected sources provide higher-quality evidence than web research alone.

## Connector Types

### GitHub Repository

```
/connect github <org/repo>
```

- Read the repo's README, key source files, architecture
- Extract claims about existing infrastructure, patterns, dependencies
- Evidence tier: `documented`
- Track as connector in claims.json source field

### Atlas File

```
/connect atlas <path-to-atlas.yaml>
```

- Read a RepoAtlas-style YAML file for multi-repo routing intelligence
- Extract claims about repo ownership, dependencies, infrastructure
- Evidence tier: `documented`

### Jira / Linear (via MCP)

```
/connect jira <project-key>
```

- Read relevant tickets, priorities, blockers
- Extract constraint and risk claims
- Evidence tier: `stated` (tickets are stakeholder input)

### Monitoring (Datadog, Grafana, etc.)

```
/connect monitoring <dashboard-name>
```

- Pull current metrics if accessible
- Evidence tier: `production` (highest tier)

### Confluence / Notion (via MCP)

```
/connect docs <space/page>
```

- Read existing documentation, ADRs, decision records
- Evidence tier: `documented`

## Process

1. **Parse the argument** to determine connector type and target.

2. **Attempt to access the source**: Use available MCP tools, file system access, or web fetch as appropriate. If the source isn't accessible, tell the user what's needed (MCP server config, file path, etc.)

3. **Extract initial claims**: Pull relevant information and create claims:

```json
{
  "id": "r0XX",
  "type": "factual|constraint",
  "topic": "<relevant topic>",
  "content": "<extracted finding>",
  "source": {
    "origin": "connector",
    "artifact": null,
    "connector": {
      "type": "<github|atlas|jira|monitoring|docs>",
      "target": "<org/repo or project-key or path>",
      "ref": "<specific file/ticket/page if applicable>",
      "fetched": "<ISO timestamp>"
    }
  },
  "evidence": "documented",
  "status": "active",
  "phase_added": "research",
  "timestamp": "<ISO timestamp>",
  "conflicts_with": [],
  "resolved_by": null,
  "tags": ["connector", "<type>"]
}
```

4. **Register the connector** in claims.json `meta.connectors`:

```json
{ "type": "github", "target": "org/repo", "connected": "<ISO timestamp>" }
```

5. **Update CLAUDE.md** Connectors section with the new connection.

6. **Run the compiler**:
   ```bash
   npx @grainulation/wheat compile --summary
   ```

## Git commit

Stage claims.json, CLAUDE.md, and any new files.

Commit: `wheat: /connect <type> <target> — added <claim IDs>`

## Tell the user

- Confirm what was connected and what was found
- List the claims extracted
- Suggest `/research` to dig deeper into findings, or `/status` to see the updated dashboard

$ARGUMENTS
