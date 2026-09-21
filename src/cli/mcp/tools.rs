//! MCP tools: thin wrappers over `Lab`, one per data-plane operation.

use std::collections::HashMap;

use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use super::{INVALID_PARAMS, RpcError};
use crate::lab::{Lab, uses_similar};
use crate::model::{Content, QueryResult, SearchOpts, StatementKey, Synonyms};
use crate::service::KnowledgePatch;

const DEFAULT_SHELF: &str = "default";
const DEFAULT_LIMIT: i64 = 100;
/// Small enough to finish well inside Codex's default 60 s tool timeout on a local model.
const BACKFILL_DEFAULT_LIMIT: usize = 64;
const BACKFILL_MAX_LIMIT: usize = 512;

type ToolResult = Result<Value, String>;

pub(super) fn definitions() -> Vec<Value> {
    let shelf = json!({ "type": "string", "description": "Shelf name; defaults to \"default\"" });
    let strings = |description: &str| json!({ "type": "array", "items": { "type": "string" }, "description": description });
    let read = json!({ "readOnlyHint": true, "openWorldHint": false });
    let write = |idempotent: bool| json!({ "readOnlyHint": false, "destructiveHint": false, "idempotentHint": idempotent, "openWorldHint": false });
    let delete = json!({ "readOnlyHint": false, "destructiveHint": true, "idempotentHint": false, "openWorldHint": false });
    let tags = strings("Tags; blank items are dropped");
    let synonyms = strings("Synonyms of the name; blank items are dropped");
    let figures = strings("Figure references such as archive://path/to/file");
    let scopes = strings(
        "Scopes; include \"\" for the global scope. An entry without \"\" is invisible to global lookups",
    );
    let knowledge_fields = |data: &str| {
        json!({
            "name": { "type": "string", "description": "Knowledge entry name" },
            "data": { "type": "string", "description": data },
            "tags": tags,
            "synonyms": synonyms,
            "figures": figures,
            "scopes": scopes,
            "shelf": shelf
        })
    };
    let triple = |extra: Value| {
        let mut properties = json!({
            "head": { "type": "string" },
            "relation": { "type": "string" },
            "tail": { "type": "string" },
            "shelf": shelf
        });
        if let (Some(p), Some(e)) = (properties.as_object_mut(), extra.as_object()) {
            p.extend(e.clone());
        }
        properties
    };
    vec![
        tool(
            "list_shelves",
            "List shelves",
            "List registered shelves and whether each is connected.",
            json!({}),
            &[],
            read.clone(),
        ),
        tool(
            "shelf_status",
            "Shelf status",
            "What works on a shelf (semantic search, embedder, anything needing attention) and its embedding debt. The same document as the hypatia://{shelf}/status resource, for hosts whose models cannot read resources.",
            json!({ "shelf": shelf }),
            &[],
            read.clone(),
        ),
        tool(
            "query",
            "Query with JSE",
            "Run a JSE query (JSON Search Expression) against knowledge or statements, e.g. [\"$knowledge\", [\"$has\", \"tags\", \"rule\"]]. Returns rows; semantic queries also return the embedding debt.",
            json!({ "jse": { "description": "The JSE expression, as JSON or as a JSON string" }, "shelf": shelf }),
            &["jse"],
            read.clone(),
        ),
        tool(
            "search",
            "Full-text search",
            "Full-text search over knowledge and statements.",
            json!({
                "query": { "type": "string" },
                "catalog": { "type": "string", "enum": ["knowledge", "statement"], "description": "Limit to one catalog" },
                "limit": { "type": "integer", "minimum": 1, "description": "Default 100" },
                "offset": { "type": "integer", "minimum": 0 },
                "shelf": shelf
            }),
            &["query"],
            read.clone(),
        ),
        tool(
            "similar",
            "Semantic search",
            "Find entries with similar meaning using vector embeddings. Fails with a next step when semantic search is off on the shelf. Also returns the embedding debt: pending entries are not found yet.",
            json!({
                "query": { "type": "string" },
                "target": { "type": "string", "enum": ["knowledge", "statement", "both"], "description": "Default both" },
                "limit": { "type": "integer", "minimum": 1, "description": "Default 100" },
                "shelf": shelf
            }),
            &["query"],
            read.clone(),
        ),
        tool(
            "session_current",
            "Unsummarized messages",
            "List message entries in a scope that no summary covers yet, oldest first.",
            json!({ "scope": { "type": "string", "description": "Scope to look in; defaults to the global scope" }, "shelf": shelf }),
            &[],
            read.clone(),
        ),
        tool(
            "scope_list",
            "List scopes",
            "List every scope in use on the shelf, with how many entries carry each. Read this before writing an entry: reusing an existing spelling is what keeps the entry findable, and a new one silently creates an island. The global scope is the empty string.",
            json!({ "shelf": shelf }),
            &[],
            read.clone(),
        ),
        tool(
            "scope_exists",
            "Check a scope",
            "Whether any entry already carries this scope. Cheaper than scope_list when you only want to confirm the spelling you are about to write.",
            json!({ "value": { "type": "string", "description": "Scope to look for; \"\" is the global scope" }, "shelf": shelf }),
            &["value"],
            read.clone(),
        ),
        tool(
            "tag_list",
            "List tags",
            "List every tag in use on the shelf, with how many entries carry each. Read this before writing an entry so it joins the vocabulary already in the shelf instead of starting a synonym of it.",
            json!({ "shelf": shelf }),
            &[],
            read.clone(),
        ),
        tool(
            "tag_exists",
            "Check a tag",
            "Whether any entry already carries this tag. Cheaper than tag_list when you only want to confirm the spelling you are about to write.",
            json!({ "value": { "type": "string", "description": "Tag to look for" }, "shelf": shelf }),
            &["value"],
            read.clone(),
        ),
        tool(
            "knowledge_create",
            "Create knowledge",
            "Create a knowledge entry. Fails if the name exists; use knowledge_update to change an entry.",
            knowledge_fields("Content data"),
            &["name"],
            write(false),
        ),
        tool(
            "knowledge_get",
            "Get knowledge",
            "Read one knowledge entry by name.",
            json!({ "name": { "type": "string" }, "shelf": shelf }),
            &["name"],
            read.clone(),
        ),
        tool(
            "knowledge_update",
            "Update knowledge",
            "Change only the fields you pass: an omitted field keeps its value, and an empty string or array clears it. `scopes` replaces the stored scopes, so include \"\" to stay global. Keeps created_at; the old vector is discarded and regenerated later. Reports changed: false, and writes nothing, when the content is already as given.",
            knowledge_fields("New content data"),
            &["name"],
            write(true),
        ),
        tool(
            "knowledge_delete",
            "Delete knowledge",
            "Delete a knowledge entry. Statements that mention it are kept.",
            json!({ "name": { "type": "string" }, "shelf": shelf }),
            &["name"],
            delete.clone(),
        ),
        tool(
            "statement_create",
            "Create statement",
            "Create a (head, relation, tail) statement. Idempotent: an existing triple is left unchanged and reported with created: false.",
            triple(json!({
                "data": { "type": "string", "description": "Content data" },
                "synonyms": {
                    "type": "object",
                    "description": "Synonyms per position",
                    "properties": { "head": strings("Synonyms of the head"), "relation": strings("Synonyms of the relation"), "tail": strings("Synonyms of the tail") },
                    "additionalProperties": false
                },
                "scopes": scopes
            })),
            &["head", "relation", "tail"],
            write(true),
        ),
        tool(
            "statement_delete",
            "Delete statement",
            "Delete a (head, relation, tail) statement.",
            triple(json!({})),
            &["head", "relation", "tail"],
            delete,
        ),
        tool(
            "archive_store",
            "Store archive file",
            "Copy a local file into the shelf's archives, record a knowledge entry describing it, and link it `is_a archive`.",
            json!({
                "file": { "type": "string", "description": "Path of a local file readable by the server" },
                "name": { "type": "string", "description": "Archive name: a relative path inside the shelf's archives; defaults to the file name" },
                "shelf": shelf
            }),
            &["file"],
            write(false),
        ),
        tool(
            "archive_get",
            "Get archive path",
            "Return the filesystem path of an archive file.",
            json!({ "name": { "type": "string" }, "shelf": shelf }),
            &["name"],
            read.clone(),
        ),
        tool(
            "archive_list",
            "List archive files",
            "List the archive files of a shelf.",
            json!({ "shelf": shelf }),
            &[],
            read,
        ),
        tool(
            "backfill",
            "Pay embedding debt",
            "Embed at most `limit` pending entries, newest first, and return what remains. Call again while a batch installs entries and debt remains; if a batch installs nothing, stop and run `hypatia backfill` in a terminal. Fails with the reason when no vector can be written, such as a missing model or vectors from another model.",
            json!({
                "limit": { "type": "integer", "minimum": 1, "maximum": BACKFILL_MAX_LIMIT, "description": "Default 64; larger batches may exceed the host's tool timeout" },
                "shelf": shelf
            }),
            &[],
            write(true),
        ),
    ]
}

fn tool(
    name: &str,
    title: &str,
    description: &str,
    properties: Value,
    required: &[&str],
    annotations: Value,
) -> Value {
    json!({
        "name": name,
        "title": title,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": false
        },
        "annotations": annotations
    })
}

pub(super) fn call(lab: &mut Lab, params: &Value) -> Result<Value, RpcError> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| RpcError::new(INVALID_PARAMS, "tools/call needs a tool name"))?;
    let args = match params.get("arguments") {
        None | Some(Value::Null) => json!({}),
        Some(args) => args.clone(),
    };
    let outcome = match name {
        "list_shelves" => list_shelves(lab, args),
        "shelf_status" => shelf_status(lab, args),
        "query" => query(lab, args),
        "search" => search(lab, args),
        "similar" => similar(lab, args),
        "session_current" => session_current(lab, args),
        "scope_list" => field_list(lab, "scopes", args),
        "scope_exists" => field_exists(lab, "scopes", args),
        "tag_list" => field_list(lab, "tags", args),
        "tag_exists" => field_exists(lab, "tags", args),
        "knowledge_create" => knowledge_create(lab, args),
        "knowledge_get" => knowledge_get(lab, args),
        "knowledge_update" => knowledge_update(lab, args),
        "knowledge_delete" => knowledge_delete(lab, args),
        "statement_create" => statement_create(lab, args),
        "statement_delete" => statement_delete(lab, args),
        "archive_store" => archive_store(lab, args),
        "archive_get" => archive_get(lab, args),
        "archive_list" => archive_list(lab, args),
        "backfill" => backfill(lab, args),
        _ => {
            return Err(RpcError::new(
                INVALID_PARAMS,
                format!("unknown tool: {name}"),
            ));
        }
    };
    Ok(match outcome {
        Ok(structured) => json!({
            "content": [{ "type": "text", "text": serde_json::to_string_pretty(&structured).unwrap_or_default() }],
            "structuredContent": structured,
            "isError": false
        }),
        // Business and argument errors go back to the model, which can correct itself.
        Err(message) => json!({
            "content": [{ "type": "text", "text": message }],
            "isError": true
        }),
    })
}

// ---- arguments ----

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NoArgs {}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ShelfArgs {
    shelf: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct QueryArgs {
    jse: Value,
    shelf: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchArgs {
    query: String,
    catalog: Option<String>,
    limit: Option<i64>,
    offset: Option<i64>,
    shelf: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SimilarArgs {
    query: String,
    target: Option<String>,
    limit: Option<i64>,
    shelf: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SessionArgs {
    scope: Option<String>,
    shelf: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FieldValueArgs {
    value: String,
    shelf: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct KnowledgeArgs {
    name: String,
    data: Option<String>,
    tags: Option<Vec<String>>,
    synonyms: Option<Vec<String>>,
    figures: Option<Vec<String>>,
    scopes: Option<Vec<String>>,
    shelf: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NameArgs {
    name: String,
    shelf: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StatementArgs {
    head: String,
    relation: String,
    tail: String,
    data: Option<String>,
    synonyms: Option<PositionalSynonyms>,
    scopes: Option<Vec<String>>,
    shelf: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PositionalSynonyms {
    head: Option<Vec<String>>,
    relation: Option<Vec<String>>,
    tail: Option<Vec<String>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TripleArgs {
    head: String,
    relation: String,
    tail: String,
    shelf: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArchiveStoreArgs {
    file: String,
    name: Option<String>,
    shelf: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BackfillArgs {
    limit: Option<usize>,
    shelf: Option<String>,
}

fn parse<T: DeserializeOwned>(args: Value) -> Result<T, String> {
    serde_json::from_value(args).map_err(|e| format!("invalid arguments: {e}"))
}

fn shelf_name(shelf: Option<String>) -> String {
    shelf.unwrap_or_else(|| DEFAULT_SHELF.to_string())
}

/// Every tool on a shelf first pays a debt that is overdue, as each CLI command does.
fn enter(lab: &mut Lab, shelf: &str) {
    let _ = lab.flush_if_overdue(shelf);
}

fn debt(lab: &Lab, shelf: &str) -> Value {
    lab.embedding_debt(shelf)
        .ok()
        .and_then(|debt| serde_json::to_value(debt).ok())
        .unwrap_or(Value::Null)
}

fn rows(result: QueryResult) -> Value {
    json!({ "rows": result.rows, "total_count": result.total_count })
}

/// Trimmed items with blanks dropped.
fn clean_list(items: Vec<String>) -> Vec<String> {
    items
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Trimmed scopes without duplicates; `""` stays, since it marks the global scope.
fn clean_scopes(items: Vec<String>) -> Vec<String> {
    let mut scopes: Vec<String> = Vec::new();
    for scope in items {
        let scope = scope.trim().to_string();
        if !scopes.contains(&scope) {
            scopes.push(scope);
        }
    }
    scopes
}

fn flat_synonyms(items: Vec<String>) -> Option<Synonyms> {
    let list = clean_list(items);
    if list.is_empty() {
        None
    } else {
        Some(Synonyms::Flat(list))
    }
}

/// Per-position synonyms, cleaned; `None` when every position is empty.
fn positional_synonyms(synonyms: PositionalSynonyms) -> Option<Synonyms> {
    let mut map = HashMap::new();
    for (position, items) in [
        ("head", synonyms.head),
        ("relation", synonyms.relation),
        ("tail", synonyms.tail),
    ] {
        let items = clean_list(items.unwrap_or_default());
        if !items.is_empty() {
            map.insert(position.to_string(), items);
        }
    }
    if map.is_empty() {
        None
    } else {
        Some(Synonyms::Positional(map))
    }
}

/// The schema's minimums, enforced: a negative limit would mean "everything" on some paths and
/// "nothing" on others.
fn paging(limit: Option<i64>, offset: Option<i64>) -> Result<(i64, i64), String> {
    let limit = limit.unwrap_or(DEFAULT_LIMIT);
    let offset = offset.unwrap_or(0);
    if limit < 1 {
        return Err(format!("limit must be at least 1, got {limit}"));
    }
    if offset < 0 {
        return Err(format!("offset must not be negative, got {offset}"));
    }
    Ok((limit, offset))
}

fn knowledge_json(k: &crate::model::Knowledge) -> Value {
    json!({ "name": k.name, "content": k.content, "created_at": k.created_at.to_string() })
}

// ---- tools ----

fn list_shelves(lab: &mut Lab, args: Value) -> ToolResult {
    parse::<NoArgs>(args)?;
    let shelves: Vec<Value> = lab
        .list_shelves()
        .into_iter()
        .map(
            |(name, path, connected)| json!({ "name": name, "path": path, "connected": connected }),
        )
        .collect();
    Ok(json!({ "shelves": shelves }))
}

fn shelf_status(lab: &mut Lab, args: Value) -> ToolResult {
    let args: ShelfArgs = parse(args)?;
    // Reports the debt as it stands, so no overdue flush runs first.
    super::resources::status_json(lab, &shelf_name(args.shelf)).map_err(|e| e.to_string())
}

fn query(lab: &mut Lab, args: Value) -> ToolResult {
    let args: QueryArgs = parse(args)?;
    let shelf = shelf_name(args.shelf);
    let jse = match args.jse {
        Value::String(text) => {
            serde_json::from_str(&text).map_err(|e| format!("invalid JSE JSON: {e}"))?
        }
        other => other,
    };
    enter(lab, &shelf);
    let semantic = uses_similar(&jse);
    let mut out = rows(lab.query(&shelf, &jse).map_err(|e| e.to_string())?);
    if semantic {
        out["embedding"] = debt(lab, &shelf);
    }
    Ok(out)
}

fn search(lab: &mut Lab, args: Value) -> ToolResult {
    let args: SearchArgs = parse(args)?;
    let (limit, offset) = paging(args.limit, args.offset)?;
    let shelf = shelf_name(args.shelf);
    enter(lab, &shelf);
    let opts = SearchOpts {
        catalog: args.catalog,
        limit,
        offset,
    };
    Ok(rows(
        lab.search(&shelf, &args.query, opts)
            .map_err(|e| e.to_string())?,
    ))
}

fn similar(lab: &mut Lab, args: Value) -> ToolResult {
    let args: SimilarArgs = parse(args)?;
    let (limit, _) = paging(args.limit, None)?;
    let shelf = shelf_name(args.shelf);
    enter(lab, &shelf);
    let target = args.target.unwrap_or_else(|| "both".to_string());
    let result = lab
        .similar(&shelf, &args.query, &target, limit)
        .map_err(|e| e.to_string())?;
    let mut out = rows(result);
    out["embedding"] = debt(lab, &shelf);
    Ok(out)
}

fn session_current(lab: &mut Lab, args: Value) -> ToolResult {
    let args: SessionArgs = parse(args)?;
    let shelf = shelf_name(args.shelf);
    enter(lab, &shelf);
    let jse = crate::cli::commands::session_current_query(args.scope.as_deref());
    Ok(rows(lab.query(&shelf, &jse).map_err(|e| e.to_string())?))
}

/// Enumeration reads the content index, which embedding never writes to, so these two
/// skip the overdue flush the other shelf tools run first.
fn field_list(lab: &mut Lab, field: &str, args: Value) -> ToolResult {
    let args: ShelfArgs = parse(args)?;
    let shelf = shelf_name(args.shelf);
    let values: Vec<Value> = lab
        .field_values(&shelf, field)
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|(value, entries)| json!({ "value": value, "entries": entries }))
        .collect();
    Ok(json!({ "field": field, "total_count": values.len(), "values": values }))
}

fn field_exists(lab: &mut Lab, field: &str, args: Value) -> ToolResult {
    let args: FieldValueArgs = parse(args)?;
    let shelf = shelf_name(args.shelf);
    let exists = lab
        .field_value_exists(&shelf, field, &args.value)
        .map_err(|e| e.to_string())?;
    Ok(json!({ "field": field, "value": args.value, "exists": exists }))
}

fn knowledge_create(lab: &mut Lab, args: Value) -> ToolResult {
    let args: KnowledgeArgs = parse(args)?;
    let shelf = shelf_name(args.shelf);
    enter(lab, &shelf);
    if lab
        .get_knowledge(&shelf, &args.name)
        .map_err(|e| e.to_string())?
        .is_some()
    {
        return Err(format!(
            "knowledge '{}' already exists; use knowledge_update to change it",
            args.name
        ));
    }
    let content = Content::new(args.data.unwrap_or_default())
        .with_tags(clean_list(args.tags.unwrap_or_default()))
        .with_synonyms(flat_synonyms(args.synonyms.unwrap_or_default()))
        .with_figures(clean_list(args.figures.unwrap_or_default()))
        .with_scopes(clean_scopes(args.scopes.unwrap_or_default()));
    let k = lab
        .create_knowledge(&shelf, &args.name, content)
        .map_err(|e| e.to_string())?;
    Ok(json!({ "name": k.name, "created": true, "embedding": debt(lab, &shelf) }))
}

fn knowledge_get(lab: &mut Lab, args: Value) -> ToolResult {
    let args: NameArgs = parse(args)?;
    let shelf = shelf_name(args.shelf);
    enter(lab, &shelf);
    match lab
        .get_knowledge(&shelf, &args.name)
        .map_err(|e| e.to_string())?
    {
        Some(k) => Ok(knowledge_json(&k)),
        None => Err(format!("not found: knowledge '{}'", args.name)),
    }
}

fn knowledge_update(lab: &mut Lab, args: Value) -> ToolResult {
    let args: KnowledgeArgs = parse(args)?;
    let shelf = shelf_name(args.shelf);
    let patch = KnowledgePatch {
        data: args.data,
        tags: args.tags.map(clean_list),
        synonyms: args.synonyms.map(flat_synonyms),
        figures: args.figures.map(clean_list),
        scopes: args.scopes.map(clean_scopes),
    };
    if patch.is_empty() {
        return Err(
            "nothing to update: pass at least one of data, tags, synonyms, figures, scopes".into(),
        );
    }
    enter(lab, &shelf);
    let updated = lab
        .patch_knowledge(&shelf, &args.name, &patch)
        .map_err(|e| e.to_string())?;
    Ok(json!({
        "name": updated.knowledge.name,
        "changed": updated.changed,
        "embedding": debt(lab, &shelf)
    }))
}

fn knowledge_delete(lab: &mut Lab, args: Value) -> ToolResult {
    let args: NameArgs = parse(args)?;
    let shelf = shelf_name(args.shelf);
    enter(lab, &shelf);
    lab.delete_knowledge(&shelf, &args.name)
        .map_err(|e| e.to_string())?;
    Ok(json!({ "name": args.name, "deleted": true }))
}

fn statement_create(lab: &mut Lab, args: Value) -> ToolResult {
    let args: StatementArgs = parse(args)?;
    let shelf = shelf_name(args.shelf);
    enter(lab, &shelf);
    let key = StatementKey::new(&args.head, &args.relation, &args.tail);
    let content = Content::new(args.data.unwrap_or_default())
        .with_synonyms(args.synonyms.and_then(positional_synonyms))
        .with_scopes(clean_scopes(args.scopes.unwrap_or_default()));
    let outcome = lab
        .create_statement(&shelf, &key, content, None, None)
        .map_err(|e| e.to_string())?;
    Ok(json!({
        "head": key.head,
        "relation": key.relation,
        "tail": key.tail,
        "created": outcome.created,
        "embedding": debt(lab, &shelf)
    }))
}

fn statement_delete(lab: &mut Lab, args: Value) -> ToolResult {
    let args: TripleArgs = parse(args)?;
    let shelf = shelf_name(args.shelf);
    enter(lab, &shelf);
    let key = StatementKey::new(&args.head, &args.relation, &args.tail);
    lab.delete_statement(&shelf, &key)
        .map_err(|e| e.to_string())?;
    Ok(json!({ "head": key.head, "relation": key.relation, "tail": key.tail, "deleted": true }))
}

fn archive_store(lab: &mut Lab, args: Value) -> ToolResult {
    let args: ArchiveStoreArgs = parse(args)?;
    let shelf = shelf_name(args.shelf);
    enter(lab, &shelf);
    let archived =
        crate::cli::archive::store(lab, &shelf, std::path::Path::new(&args.file), args.name)
            .map_err(|e| e.to_string())?;
    Ok(json!({
        "uri": archived.uri,
        "knowledge": archived.knowledge,
        "mime_type": archived.mime_type,
        "size_bytes": archived.size_bytes,
        "embedding": debt(lab, &shelf)
    }))
}

fn archive_get(lab: &mut Lab, args: Value) -> ToolResult {
    let args: NameArgs = parse(args)?;
    let shelf = shelf_name(args.shelf);
    enter(lab, &shelf);
    match lab.get_archive_path(&shelf, &args.name) {
        Some(path) => Ok(json!({ "name": args.name, "path": path })),
        None => Err(format!(
            "not found: archive '{}' in shelf '{shelf}'",
            args.name
        )),
    }
}

fn archive_list(lab: &mut Lab, args: Value) -> ToolResult {
    let args: ShelfArgs = parse(args)?;
    let shelf = shelf_name(args.shelf);
    enter(lab, &shelf);
    let files: Vec<String> = lab
        .list_archives(&shelf)
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|f| format!("archive://{f}"))
        .collect();
    Ok(json!({ "files": files }))
}

fn backfill(lab: &mut Lab, args: Value) -> ToolResult {
    let args: BackfillArgs = parse(args)?;
    let shelf = shelf_name(args.shelf);
    // `backfill` pays debt itself, so it skips the overdue flush the other tools run first.
    let limit = args.limit.unwrap_or(BACKFILL_DEFAULT_LIMIT);
    if !(1..=BACKFILL_MAX_LIMIT).contains(&limit) {
        return Err(format!(
            "limit must be between 1 and {BACKFILL_MAX_LIMIT}, got {limit}"
        ));
    }
    let stats = lab
        .backfill_batch(&shelf, limit)
        .map_err(|e| e.to_string())?;
    Ok(json!({
        "installed": stats.installed,
        "skipped": stats.skipped,
        "failed": stats.failed,
        "error": stats.error,
        "embedding": debt(lab, &shelf)
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_arguments_are_normalised() {
        assert_eq!(clean_list(vec![" a ".into(), "".into(), " ".into()]), ["a"]);
        assert_eq!(
            clean_scopes(vec!["p".into(), "".into(), " p ".into(), "".into()]),
            ["p", ""]
        );
        assert_eq!(flat_synonyms(vec![" ".into()]), None);
    }

    #[test]
    fn every_tool_declares_an_object_schema_and_no_admin_tool_is_exposed() {
        let names: Vec<String> = definitions()
            .iter()
            .map(|t| {
                assert_eq!(t["inputSchema"]["type"], "object", "{t}");
                t["name"].as_str().unwrap().to_string()
            })
            .collect();
        for admin in [
            "connect",
            "disconnect",
            "init",
            "model_install",
            "export",
            "import",
        ] {
            assert!(
                !names.iter().any(|n| n == admin),
                "{admin} must stay CLI-only"
            );
        }
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "duplicate tool names");
    }
}
