use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use crate::lab::{Lab, uses_similar};
use crate::model::{Content, QueryResult, SearchOpts, StatementKey, Synonyms};
use crate::service::KnowledgePatch;

#[derive(Parser)]
#[command(name = "hypatia", about = "AI-oriented memory management", version)]
pub struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Connect to a shelf directory
    Connect {
        /// Path to shelf directory
        path: PathBuf,
        /// Optional name for the shelf
        #[arg(short, long)]
        name: Option<String>,
    },
    /// Disconnect from a shelf
    Disconnect { name: String },
    /// List connected shelves
    List,
    /// Set up a shelf (the default one, or the directory given) and show what works on it
    Init {
        /// Shelf directory; the default shelf when omitted
        path: Option<PathBuf>,
        /// Name to register the directory under; without a directory, the registered shelf to report on
        #[arg(short, long)]
        name: Option<String>,
    },
    /// Execute a JSE query
    Query {
        /// JSE query as JSON string
        jse: String,
        /// Shelf to query
        #[arg(short, long, default_value = "default")]
        shelf: String,
    },
    /// Create a knowledge entry
    KnowledgeCreate {
        name: String,
        /// Content data
        #[arg(short, long, default_value = "")]
        data: String,
        /// Tags (comma-separated)
        #[arg(short, long, default_value = "")]
        tags: String,
        /// Synonyms (comma-separated)
        #[arg(long, default_value = "")]
        synonyms: String,
        /// Binary figure references (comma-separated, e.g. binary://euclid/fig1.png)
        #[arg(short, long, default_value = "")]
        figures: String,
        /// Scopes (comma-separated, e.g. "project-a," for global)
        #[arg(long, default_value = "")]
        scopes: String,
        /// Shelf name
        #[arg(short, long, default_value = "default")]
        shelf: String,
    },
    /// Update a knowledge entry: omitted fields keep their values, an empty value clears one
    KnowledgeUpdate {
        name: String,
        /// New content data
        #[arg(short, long)]
        data: Option<String>,
        /// New tags (comma-separated); "" clears them
        #[arg(short, long)]
        tags: Option<String>,
        /// New synonyms (comma-separated); "" clears them
        #[arg(long)]
        synonyms: Option<String>,
        /// New figure references (comma-separated); "" clears them
        #[arg(short, long)]
        figures: Option<String>,
        /// New scopes (comma-separated, trailing comma adds global); "" clears them
        #[arg(long)]
        scopes: Option<String>,
        /// Shelf name
        #[arg(short, long, default_value = "default")]
        shelf: String,
    },
    /// Get a knowledge entry
    KnowledgeGet {
        name: String,
        #[arg(short, long, default_value = "default")]
        shelf: String,
    },
    /// Delete a knowledge entry
    KnowledgeDelete {
        name: String,
        #[arg(short, long, default_value = "default")]
        shelf: String,
    },
    /// Delete a statement (triple)
    StatementDelete {
        head: String,
        relation: String,
        tail: String,
        #[arg(short, long, default_value = "default")]
        shelf: String,
    },
    /// Create a statement (triple); exits 0 without changes if it already exists
    StatementCreate {
        head: String,
        relation: String,
        tail: String,
        /// Content data
        #[arg(short, long, default_value = "")]
        data: String,
        /// Synonyms as JSON: {"head":["Bob"],"relation":["leads"],"tail":["DB"]}
        #[arg(long)]
        synonyms: Option<String>,
        /// Scopes (comma-separated, e.g. "project-a," for global)
        #[arg(long, default_value = "")]
        scopes: String,
        #[arg(short, long, default_value = "default")]
        shelf: String,
    },
    /// Search knowledge and statements
    Search {
        query: String,
        #[arg(short, long)]
        catalog: Option<String>,
        #[arg(long, default_value_t = 100)]
        limit: i64,
        #[arg(long, default_value_t = 0)]
        offset: i64,
        #[arg(short, long, default_value = "default")]
        shelf: String,
    },
    /// Find semantically similar entries using vector embeddings
    Similar {
        /// Query text to search for similar entries
        query: String,
        /// Search target: knowledge, statement, or both
        #[arg(short, long, default_value = "both")]
        target: String,
        /// Maximum number of results
        #[arg(long, default_value_t = 100)]
        limit: i64,
        /// Shelf to search
        #[arg(short, long, default_value = "default")]
        shelf: String,
    },
    /// Export a shelf to another directory
    Export { name: String, dest: PathBuf },
    /// Import an export into an empty configured shelf (does not switch backends)
    Import {
        source: PathBuf,
        #[arg(short, long, default_value = "default")]
        shelf: String,
        /// Omit exported vectors; use backfill with the target model afterwards
        #[arg(long)]
        reembed: bool,
    },
    /// Generate embeddings for existing entries that don't have vectors yet
    Backfill {
        /// Explicitly invalidate all vectors and regenerate with the configured model
        #[arg(long)]
        reembed: bool,
        /// Report the embedding debt as JSON instead of paying it
        #[arg(long, conflicts_with = "reembed")]
        status: bool,
        /// Shelf to backfill
        #[arg(short, long, default_value = "default")]
        shelf: String,
    },
    /// Store a file in the shelf archives and create a knowledge entry with metadata
    ArchiveStore {
        /// Path to the source file
        file: PathBuf,
        /// Destination path relative to archives/ (e.g. euclid/fig1.png)
        #[arg(short, long)]
        name: Option<String>,
        /// Shelf name
        #[arg(short, long, default_value = "default")]
        shelf: String,
    },
    /// Get an archive file path or copy it to a destination
    ArchiveGet {
        /// Archive file name (relative path in archives/)
        name: String,
        /// Output path (prints absolute path if omitted)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Shelf name
        #[arg(short, long, default_value = "default")]
        shelf: String,
    },
    /// List all archive files in the shelf
    ArchiveList {
        /// Shelf name
        #[arg(short, long, default_value = "default")]
        shelf: String,
    },
    /// List the scopes in use, or check one before writing with it
    #[command(subcommand)]
    Scope(FieldCommands),
    /// List the tags in use, or check one before writing with it
    #[command(subcommand)]
    Tag(FieldCommands),
    /// Manage embedding models
    #[command(subcommand)]
    Model(ModelCommands),
    /// Get unsummarized messages for the current session
    SessionCurrent {
        /// Filter by scope (project/user key)
        #[arg(long)]
        scope: Option<String>,
        /// Shelf to query
        #[arg(short, long, default_value = "default")]
        shelf: String,
    },
    /// Serve the knowledge graph to an agent over MCP (stdio)
    Mcp,
    /// Install or check the bundled agent skills
    #[command(subcommand)]
    Skill(super::skill::SkillCommands),
    /// Enter interactive REPL mode
    Repl,
}

/// `scope` and `tag` differ only in the content field they read.
#[derive(Subcommand)]
enum FieldCommands {
    /// List the distinct values in use, one per line
    List {
        /// Show how many entries carry each value
        #[arg(long)]
        count: bool,
        /// Emit [{"value":…,"entries":N}] instead; the global scope stays the empty string
        #[arg(long, conflicts_with = "count")]
        json: bool,
        /// Shelf to read
        #[arg(short, long, default_value = "default")]
        shelf: String,
    },
    /// Report whether a value is in use; exits 1 when it is not
    Exists {
        /// The value to look for; "" is the global scope
        name: String,
        /// Shelf to read
        #[arg(short, long, default_value = "default")]
        shelf: String,
    },
}

#[derive(Subcommand)]
enum ModelCommands {
    /// List available models in ~/.hypatia/models/
    List,
    /// Register a local model directory as a named model
    Register {
        /// Model name in Org/Name format (e.g. "BAAI/bge-m3")
        name: String,
        /// Path to the model directory containing ONNX and tokenizer files
        path: PathBuf,
    },
    /// Show details of a registered model
    Show {
        /// Model name or path
        name: String,
    },
    /// Download an ONNX model from Hugging Face into ~/.hypatia/models/ and use it on a shelf
    Install {
        /// Model name in Org/Name format (e.g. "BAAI/bge-m3")
        name: String,
        /// Shelf to use the model on; left unchanged if it holds vectors of another model
        #[arg(short, long, default_value = "default")]
        shelf: String,
        /// Branch, tag or commit to download
        #[arg(long, default_value = "main")]
        revision: String,
    },
}

impl Commands {
    /// The shelf whose overdue embedding debt a command pays before it runs. `backfill` and
    /// `import` handle vectors themselves, `export` copies the shelf as it is,
    /// `disconnect` and the archive file lookups never read entries, and `scope` and `tag`
    /// read the content index, which vectors never reach — loading a model to list four
    /// names would cost more than the listing.
    fn shelf(&self) -> Option<&str> {
        match self {
            Self::Query { shelf, .. }
            | Self::KnowledgeCreate { shelf, .. }
            | Self::KnowledgeUpdate { shelf, .. }
            | Self::KnowledgeGet { shelf, .. }
            | Self::KnowledgeDelete { shelf, .. }
            | Self::StatementDelete { shelf, .. }
            | Self::StatementCreate { shelf, .. }
            | Self::Search { shelf, .. }
            | Self::Similar { shelf, .. }
            | Self::ArchiveStore { shelf, .. }
            | Self::SessionCurrent { shelf, .. } => Some(shelf),
            Self::Connect { .. }
            | Self::Disconnect { .. }
            | Self::List
            | Self::Init { .. }
            | Self::Export { .. }
            | Self::Import { .. }
            | Self::Backfill { .. }
            | Self::ArchiveGet { .. }
            | Self::ArchiveList { .. }
            | Self::Scope(_)
            | Self::Tag(_)
            | Self::Model(_)
            | Self::Mcp
            | Self::Skill(_)
            | Self::Repl => None,
        }
    }
}

/// The process exit code, so `scope exists` and `tag exists` can answer "no" without
/// an error message. The shelves still close through `Lab`'s drop, which persists the
/// vector cache.
pub fn run() -> crate::error::Result<ExitCode> {
    let cli = Cli::parse();
    // Skill management touches no shelf: dispatch it before Lab::new(), which
    // opens every registered shelf and creates the default one on first run.
    if let Some(Commands::Skill(cmd)) = cli.command {
        super::skill::execute(cmd)?;
        return Ok(ExitCode::SUCCESS);
    }
    let mut lab = Lab::new()?;

    match cli.command {
        None | Some(Commands::Repl) => {
            let mut repl = super::repl::Repl::new(lab)?;
            repl.run()?;
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Mcp) => {
            super::mcp::serve(lab)?;
            Ok(ExitCode::SUCCESS)
        }
        Some(cmd) => execute_command(&mut lab, cmd),
    }
}

fn execute_command(lab: &mut Lab, cmd: Commands) -> crate::error::Result<ExitCode> {
    // Reject an update with nothing to change before paying any embedding debt.
    if let Commands::KnowledgeUpdate {
        data: None,
        tags: None,
        synonyms: None,
        figures: None,
        scopes: None,
        ..
    } = &cmd
    {
        return Err(crate::error::HypatiaError::Validation(
            "nothing to update: pass at least one of --data, --tags, --synonyms, --figures, \
             --scopes"
                .into(),
        ));
    }
    if let Some(shelf) = cmd.shelf() {
        // Best-effort: the command itself reports a shelf that is missing or broken.
        let _ = lab.flush_if_overdue(shelf);
    }
    match cmd {
        Commands::Connect { path, name } => {
            let shelf_name = lab.connect_shelf(&path, name.as_deref())?;
            println!("Shelf '{}' connected and registered.", shelf_name);
        }
        Commands::Disconnect { name } => {
            lab.disconnect_shelf(&name)?;
            println!("Shelf '{}' disconnected and unregistered.", name);
        }
        Commands::List => {
            let shelves = lab.list_shelves();
            if shelves.is_empty() {
                println!("No shelves registered.");
            } else {
                // Calculate column widths for alignment
                let max_name = shelves.iter().map(|(n, _, _)| n.len()).max().unwrap_or(0);
                for (name, path, connected) in &shelves {
                    let status = if *connected {
                        "[connected]"
                    } else {
                        "[disconnected]"
                    };
                    println!(
                        "  {:width$}  {}  {}",
                        name,
                        path.display(),
                        status,
                        width = max_name
                    );
                }
            }
        }
        Commands::Init { path, name } => super::init::run(lab, path.as_deref(), name.as_deref())?,
        Commands::Query { jse, shelf } => {
            let json: serde_json::Value = serde_json::from_str(&jse)
                .map_err(|e| crate::error::HypatiaError::Parse(format!("invalid JSON: {e}")))?;
            let semantic = uses_similar(&json);
            let result = lab.query(&shelf, &json)?;
            print_result(&result);
            if semantic {
                warn_if_incomplete(lab, &shelf, "both");
            }
        }
        Commands::KnowledgeCreate {
            name,
            data,
            tags,
            synonyms,
            figures,
            scopes,
            shelf,
        } => {
            let content = Content::new(&data)
                .with_tags(parse_tags(&tags))
                .with_synonyms(parse_flat_synonyms(&synonyms))
                .with_figures(parse_list(&figures))
                .with_scopes(parse_scopes(&scopes));
            let k = lab.create_knowledge(&shelf, &name, content)?;
            println!("Created knowledge: {}", k.name);
        }
        Commands::KnowledgeUpdate {
            name,
            data,
            tags,
            synonyms,
            figures,
            scopes,
            shelf,
        } => {
            let patch = KnowledgePatch {
                data,
                tags: tags.as_deref().map(parse_tags),
                synonyms: synonyms.as_deref().map(parse_flat_synonyms),
                figures: figures.as_deref().map(parse_list),
                scopes: scopes.as_deref().map(parse_scopes),
            };
            let updated = lab.patch_knowledge(&shelf, &name, &patch)?;
            if updated.changed {
                println!("Updated knowledge: {}", updated.knowledge.name);
            } else {
                println!("Knowledge unchanged: {}", updated.knowledge.name);
            }
        }
        Commands::KnowledgeGet { name, shelf } => match lab.get_knowledge(&shelf, &name)? {
            Some(k) => {
                let json = serde_json::to_string_pretty(&serde_json::json!({
                    "name": k.name,
                    "content": k.content,
                    "created_at": k.created_at.to_string(),
                }))?;
                println!("{json}");
            }
            None => println!("Knowledge '{}' not found.", name),
        },
        Commands::KnowledgeDelete { name, shelf } => {
            lab.delete_knowledge(&shelf, &name)?;
            println!("Deleted knowledge: {name}");
        }
        Commands::StatementDelete {
            head,
            relation,
            tail,
            shelf,
        } => {
            let key = StatementKey::new(&head, &relation, &tail);
            lab.delete_statement(&shelf, &key)?;
            println!("Deleted statement: ({}, {}, {})", head, relation, tail);
        }
        Commands::StatementCreate {
            head,
            relation,
            tail,
            data,
            synonyms,
            scopes,
            shelf,
        } => {
            let key = StatementKey::new(&head, &relation, &tail);
            let syn = match synonyms {
                Some(ref json_str) => {
                    let map: std::collections::HashMap<String, Vec<String>> =
                        serde_json::from_str(json_str).map_err(|e| {
                            crate::error::HypatiaError::Parse(format!("invalid synonyms JSON: {e}"))
                        })?;
                    Some(Synonyms::Positional(map))
                }
                None => None,
            };
            let scopes_vec = parse_scopes(&scopes);
            let content = Content::new(&data)
                .with_synonyms(syn)
                .with_scopes(scopes_vec);
            let outcome = lab.create_statement(&shelf, &key, content, None, None)?;
            // Exit 0 either way: an existing triple means the relationship is
            // already recorded, and its stored content is left unchanged.
            let verb = if outcome.created {
                "Created statement"
            } else {
                "Statement already exists"
            };
            let s = &outcome.statement;
            println!(
                "{verb}: ({}, {}, {})",
                s.key.head, s.key.relation, s.key.tail
            );
        }
        Commands::Search {
            query,
            catalog,
            limit,
            offset,
            shelf,
        } => {
            let opts = SearchOpts {
                catalog,
                limit,
                offset,
            };
            let result = lab.search(&shelf, &query, opts)?;
            print_result(&result);
        }
        Commands::Similar {
            query,
            target,
            limit,
            shelf,
        } => {
            let result = lab.similar(&shelf, &query, &target, limit)?;
            print_result(&result);
            warn_if_incomplete(lab, &shelf, &target);
        }
        Commands::Export { name, dest } => {
            lab.export_shelf(&name, &dest)?;
            println!("Exported shelf '{name}' to {}", dest.display());
        }
        Commands::Import {
            source,
            shelf,
            reembed,
        } => {
            lab.import_shelf(&shelf, &source, reembed)?;
            println!(
                "Imported {} into shelf '{shelf}'; connection configuration unchanged",
                source.display()
            );
        }
        Commands::Backfill {
            shelf,
            reembed,
            status,
        } => {
            if status {
                let debt = lab.embedding_debt(&shelf)?;
                println!("{}", serde_json::to_string_pretty(&debt)?);
                return Ok(ExitCode::SUCCESS);
            }
            let stats = lab.backfill_vectors_with_reembed(&shelf, reembed)?;
            println!(
                "Backfill complete: {} vectors created, {} skipped, {} errors",
                stats.created, stats.skipped, stats.errors
            );
            // Embedding is this command's only job: failed entries are a failure.
            if stats.errors > 0 {
                return Err(crate::error::HypatiaError::Embedding(format!(
                    "{} entries could not be embedded and stay pending",
                    stats.errors
                )));
            }
        }
        Commands::ArchiveStore { file, name, shelf } => {
            let archived = super::archive::store(lab, &shelf, &file, name)?;
            println!("Stored: {}", archived.uri);
            println!("Knowledge: {}", archived.knowledge);
            println!(
                "MIME: {}, Size: {} bytes",
                archived.mime_type, archived.size_bytes
            );
        }
        Commands::ArchiveGet {
            name,
            output,
            shelf,
        } => match lab.get_archive_path(&shelf, &name) {
            Some(path) => match output {
                Some(dest) => {
                    std::fs::copy(&path, &dest)?;
                    println!("Copied to: {}", dest.display());
                }
                None => {
                    println!("{}", path.display());
                }
            },
            None => println!("Archive '{}' not found in shelf '{}'.", name, shelf),
        },
        Commands::ArchiveList { shelf } => {
            let files = lab.list_archives(&shelf)?;
            if files.is_empty() {
                println!("No archive files in shelf '{}'.", shelf);
            } else {
                for f in &files {
                    println!("  archive://{}", f);
                }
                println!("  ({} files)", files.len());
            }
        }
        Commands::SessionCurrent { scope, shelf } => {
            let jse = session_current_query(scope.as_deref());
            let result = lab.query(&shelf, &jse)?;
            if result.rows.is_empty() {
                println!("No unsummarized messages.");
            } else {
                for row in &result.rows {
                    let name = row.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                    let created = row.get("created_at").and_then(|v| v.as_str()).unwrap_or("");
                    println!("  {name}  {created}");
                }
                println!("  ({} unsummarized messages)", result.rows.len());
            }
        }
        Commands::Scope(cmd) => return execute_field_command(lab, "scopes", cmd),
        Commands::Tag(cmd) => return execute_field_command(lab, "tags", cmd),
        Commands::Model(cmd) => execute_model_command(lab, cmd)?,
        Commands::Repl => unreachable!(),
        Commands::Mcp => {
            return Err(crate::error::HypatiaError::Validation(
                "`hypatia mcp` runs as a server of its own".into(),
            ));
        }
        // Normally dispatched before Lab::new(); still correct if routed here.
        Commands::Skill(cmd) => super::skill::execute(cmd)?,
    }
    Ok(ExitCode::SUCCESS)
}

/// A label for the terminal: the global scope is stored as an empty string, which
/// would otherwise print as a blank line. It is not the value — a caller that needs
/// the value back verbatim reads `--json`, or the MCP tools, which never relabel.
fn display_value(field: &str, value: &str) -> String {
    match (field, value) {
        ("scopes", "") => "(global)".to_string(),
        (_, "") => "(empty)".to_string(),
        _ => value.to_string(),
    }
}

fn execute_field_command(
    lab: &mut Lab,
    field: &str,
    cmd: FieldCommands,
) -> crate::error::Result<ExitCode> {
    // `scope` / `tag`, as the command the user typed, for messages and hints.
    let noun = field.trim_end_matches('s');
    match cmd {
        FieldCommands::List { count, json, shelf } => {
            let values = lab.field_values(&shelf, field)?;
            if json {
                let rows: Vec<serde_json::Value> = values
                    .iter()
                    .map(|(value, entries)| serde_json::json!({ "value": value, "entries": entries }))
                    .collect();
                println!("{}", serde_json::to_string_pretty(&rows)?);
                return Ok(ExitCode::SUCCESS);
            }
            if values.is_empty() {
                println!("No {field} in shelf '{shelf}'.");
                return Ok(ExitCode::SUCCESS);
            }
            let width = values
                .iter()
                .map(|(v, _)| display_value(field, v).chars().count())
                .max()
                .unwrap_or(0);
            for (value, entries) in &values {
                let shown = display_value(field, value);
                if count {
                    println!("  {shown:width$}  {entries}");
                } else {
                    println!("  {shown}");
                }
            }
            println!("  ({} {field})", values.len());
        }
        FieldCommands::Exists { name, shelf } => {
            // Quoted, so the global scope reads as "" and cannot be confused with a
            // scope whose name happens to be the label `list` prints for it.
            if lab.field_value_exists(&shelf, field, &name)? {
                println!("{name:?}");
            } else {
                // Not an error: asking is how a writer avoids inventing a spelling.
                eprintln!("no such {noun}: {name:?} (hypatia {noun} list -s {shelf})");
                return Ok(ExitCode::FAILURE);
            }
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn execute_model_command(lab: &mut Lab, cmd: ModelCommands) -> crate::error::Result<()> {
    match cmd {
        ModelCommands::Install {
            name,
            shelf,
            revision,
        } => super::model_install::run(lab, &name, &shelf, &revision)?,
        ModelCommands::List => {
            let models = crate::embedding::config::list_local_models();
            if models.is_empty() {
                println!("No models found in ~/.hypatia/models/");
                println!(
                    "Use 'hypatia model install BAAI/bge-m3' to download one, or 'hypatia model register <name> <path>' to register a local directory."
                );
            } else {
                for (name, path) in &models {
                    // Show symlink target if applicable
                    let resolved = std::fs::canonicalize(path).unwrap_or_else(|_| path.clone());
                    if resolved != *path {
                        println!("  {} -> {}", name, resolved.display());
                    } else {
                        println!("  {}  {}", name, path.display());
                    }
                }
                println!("  ({} models)", models.len());
            }
        }
        ModelCommands::Register { name, path } => {
            let abs_path = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
            match crate::embedding::config::register_model(&name, &abs_path) {
                Ok(target) => {
                    println!("Registered model '{}' -> {}", name, abs_path.display());
                    println!("  Location: {}", target.display());
                }
                Err(e) => {
                    return Err(crate::error::HypatiaError::Config(format!(
                        "failed to register model '{}': {}",
                        name, e
                    )));
                }
            }
        }
        ModelCommands::Show { name } => match crate::embedding::config::model_info(&name) {
            Ok(info) => {
                println!("Model: {}", info.name);
                println!("Directory: {}", info.directory.display());
                println!("ONNX: {}", info.model_path.display());
                println!("Tokenizer: {}", info.tokenizer_path.display());
                println!("Files:");
                for f in &info.files {
                    let size = if f.size_bytes >= 1_073_741_824 {
                        format!("{:.1} GB", f.size_bytes as f64 / 1_073_741_824.0)
                    } else if f.size_bytes >= 1_048_576 {
                        format!("{:.1} MB", f.size_bytes as f64 / 1_048_576.0)
                    } else if f.size_bytes >= 1024 {
                        format!("{:.1} KB", f.size_bytes as f64 / 1024.0)
                    } else {
                        format!("{} B", f.size_bytes)
                    };
                    println!("  {:30} {}", f.name, size);
                }
                let total = if info.total_size_bytes >= 1_073_741_824 {
                    format!("{:.1} GB", info.total_size_bytes as f64 / 1_073_741_824.0)
                } else {
                    format!("{:.1} MB", info.total_size_bytes as f64 / 1_048_576.0)
                };
                println!("Total: {}", total);
            }
            Err(e) => {
                return Err(crate::error::HypatiaError::Config(format!(
                    "model '{}' not found: {}",
                    name, e
                )));
            }
        },
    }
    Ok(())
}

/// Semantic results cannot include entries that have no vector yet; say so on stderr.
/// `target` is the searched catalog: "knowledge", "statement" or "both".
pub(crate) fn warn_if_incomplete(lab: &Lab, shelf: &str, target: &str) {
    let Ok(debt) = lab.embedding_debt(shelf) else {
        return;
    };
    let pending = match target {
        "knowledge" => debt.pending_knowledge,
        "statement" => debt.pending_statement,
        _ => debt.pending_knowledge + debt.pending_statement,
    };
    if pending > 0 {
        eprintln!(
            "note: {pending} entries are not embedded yet, so results may be incomplete; run `hypatia backfill -s {shelf}`"
        );
        if let Some(paused) = debt.paused {
            eprintln!("note: automatic embedding is paused: {}", paused.reason);
        }
    }
}

fn print_result(result: &QueryResult) {
    if result.rows.is_empty() {
        println!("No results found.");
    } else {
        match serde_json::to_string_pretty(&result.rows) {
            Ok(json) => println!("{json}"),
            Err(e) => eprintln!("Error formatting result: {e}"),
        }
    }
}

/// The JSE query behind `session-current`: messages in `scope` (default: the global scope)
/// that no summary covers yet.
pub(super) fn session_current_query(scope: Option<&str>) -> serde_json::Value {
    serde_json::json!([
        "$not-summaried",
        "message",
        ["$contains", "scopes", scope.unwrap_or("")]
    ])
}

/// Comma-separated tags, split exactly as `knowledge-create` always has.
fn parse_tags(raw: &str) -> Vec<String> {
    if raw.is_empty() {
        Vec::new()
    } else {
        raw.split(',').map(|s| s.trim().to_string()).collect()
    }
}

/// A comma-separated list with blank items dropped; `""` gives an empty list.
fn parse_list(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Comma-separated knowledge synonyms; `""` gives none.
fn parse_flat_synonyms(raw: &str) -> Option<Synonyms> {
    let list = parse_list(raw);
    if list.is_empty() {
        None
    } else {
        Some(Synonyms::Flat(list))
    }
}

/// Comma-separated scopes. A trailing comma adds the empty-string global scope, so `","`
/// is global only and `"p,"` is project plus global; `""` stores no scope at all.
fn parse_scopes(raw: &str) -> Vec<String> {
    let mut scopes = parse_list(raw);
    if raw.ends_with(',') && !scopes.contains(&String::new()) {
        scopes.push(String::new());
    }
    scopes
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;
    use std::collections::HashMap;

    #[test]
    fn scope_and_list_parsing_matches_the_documented_cli_behaviour() {
        assert!(parse_scopes("").is_empty());
        assert_eq!(parse_scopes(","), [""]);
        assert_eq!(parse_scopes("p"), ["p"]);
        assert_eq!(parse_scopes("p,"), ["p", ""]);
        assert_eq!(parse_scopes(" a , b "), ["a", "b"]);
        assert_eq!(parse_list("x,,y"), ["x", "y"]);
        assert_eq!(parse_tags("a,b"), ["a", "b"]);
        // Tags have never dropped blank items, unlike the other lists; keep create unchanged.
        assert_eq!(parse_tags(","), ["", ""]);
        assert_eq!(parse_tags("a,,b"), ["a", "", "b"]);
        assert_eq!(parse_tags(" "), [""]);
        assert!(parse_list(",").is_empty());
        // Only a comma at the very end marks global scope.
        assert_eq!(parse_scopes("p, "), ["p"]);
        assert_eq!(parse_flat_synonyms(""), None);
        assert_eq!(
            parse_flat_synonyms("q, r"),
            Some(Synonyms::Flat(vec!["q".into(), "r".into()]))
        );
    }

    #[test]
    fn session_current_query_defaults_to_the_global_scope() {
        let q = |scope| session_current_query(scope)[2][2].clone();
        assert_eq!(q(None), "");
        assert_eq!(q(Some("")), "");
        assert_eq!(q(Some("proj")), "proj");
    }

    #[test]
    fn mcp_touches_no_shelf_debt_itself() {
        let cli = Cli::try_parse_from(["hypatia", "mcp"]).unwrap();
        assert_eq!(cli.command.unwrap().shelf(), None);
    }

    #[test]
    fn knowledge_update_settles_the_debt_of_its_shelf() {
        let cli =
            Cli::try_parse_from(["hypatia", "knowledge-update", "k", "-d", "x", "-s", "work"])
                .unwrap();
        let cmd = cli.command.unwrap();
        assert_eq!(cmd.shelf(), Some("work"));
    }

    /// Check every subcommand for duplicate short flags.
    /// Catches issues like -s being used for both --shelf and --synonyms.
    #[test]
    fn no_duplicate_short_flags() {
        let cmd = Cli::command();
        check_subcommand(&cmd);

        for sub in cmd.get_subcommands() {
            check_subcommand(sub);
            for sub2 in sub.get_subcommands() {
                check_subcommand(sub2);
            }
        }
    }

    #[test]
    fn commands_on_a_shelf_settle_its_debt_except_backfill_and_import() {
        let shelf = |args: &[&str]| {
            let cli = Cli::try_parse_from(args).unwrap();
            cli.command.unwrap().shelf().map(str::to_string)
        };
        assert_eq!(
            shelf(&["hypatia", "similar", "x", "-s", "work"]).as_deref(),
            Some("work")
        );
        assert_eq!(
            shelf(&["hypatia", "knowledge-create", "k"]).as_deref(),
            Some("default")
        );
        assert_eq!(shelf(&["hypatia", "backfill", "-s", "work"]), None);
        assert_eq!(shelf(&["hypatia", "import", "/tmp/export"]), None);
        assert_eq!(shelf(&["hypatia", "list"]), None);
        assert_eq!(shelf(&["hypatia", "init", "/tmp/shelf"]), None);
        assert_eq!(shelf(&["hypatia", "archive-list", "-s", "work"]), None);
        // Enumeration reads the content index; a vector flush would add nothing.
        assert_eq!(shelf(&["hypatia", "scope", "list", "-s", "work"]), None);
        assert_eq!(shelf(&["hypatia", "tag", "exists", "rule"]), None);
    }

    #[test]
    fn scope_and_tag_take_the_same_subcommands_on_a_named_shelf() {
        let parsed = |args: &[&str]| match Cli::try_parse_from(args).unwrap().command.unwrap() {
            Commands::Scope(cmd) => ("scopes", cmd),
            Commands::Tag(cmd) => ("tags", cmd),
            other => panic!("not a field command: {:?}", other.shelf()),
        };
        assert!(matches!(
            parsed(&["hypatia", "scope", "list"]),
            ("scopes", FieldCommands::List { count: false, json: false, ref shelf }) if shelf == "default"
        ));
        assert!(matches!(
            parsed(&["hypatia", "tag", "list", "--count", "-s", "work"]),
            ("tags", FieldCommands::List { count: true, ref shelf, .. }) if shelf == "work"
        ));
        assert!(matches!(
            parsed(&["hypatia", "scope", "list", "--json"]),
            ("scopes", FieldCommands::List { json: true, .. })
        ));
        // --json is the exact-value form; pairing it with a column layout is a mistake.
        assert!(Cli::try_parse_from(["hypatia", "tag", "list", "--json", "--count"]).is_err());
        assert!(matches!(
            parsed(&["hypatia", "scope", "exists", ""]),
            ("scopes", FieldCommands::Exists { ref name, .. }) if name.is_empty()
        ));
        // `exists` needs its value: an accidental bare call must not answer "no".
        assert!(Cli::try_parse_from(["hypatia", "tag", "exists"]).is_err());
    }

    #[test]
    fn the_global_scope_prints_as_a_label_rather_than_a_blank_line() {
        assert_eq!(display_value("scopes", ""), "(global)");
        assert_eq!(display_value("scopes", "proj"), "proj");
        assert_eq!(display_value("tags", ""), "(empty)");
        assert_eq!(display_value("tags", "rule"), "rule");
    }

    fn check_subcommand(cmd: &clap::Command) {
        let mut seen: HashMap<char, String> = HashMap::new();
        for arg in cmd.get_arguments() {
            if let Some(short) = arg.get_short() {
                if let Some(prev) = seen.insert(short, arg.get_id().to_string()) {
                    panic!(
                        "Command '{}': short flag '-{}' used by both '{}' and '{}'",
                        cmd.get_name(),
                        short,
                        prev,
                        arg.get_id()
                    );
                }
            }
        }
    }
}
