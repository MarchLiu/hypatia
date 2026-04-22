use std::path::PathBuf;

use clap::{Parser, Subcommand};

use crate::lab::Lab;
use crate::model::{Content, QueryResult, SearchOpts, StatementKey, Synonyms};

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
    Disconnect {
        name: String,
    },
    /// List connected shelves
    List,
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
        subject: String,
        predicate: String,
        object: String,
        #[arg(short, long, default_value = "default")]
        shelf: String,
    },
    /// Create a statement (triple)
    StatementCreate {
        subject: String,
        predicate: String,
        object: String,
        /// Content data
        #[arg(short, long, default_value = "")]
        data: String,
        /// Synonyms as JSON: {"subject":["Bob"],"predicate":["leads"],"object":["DB"]}
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
    /// Export a shelf to another directory
    Export {
        name: String,
        dest: PathBuf,
    },
    /// Generate embeddings for existing entries that don't have vectors yet
    Backfill {
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
    /// Enter interactive REPL mode
    Repl,
}

pub fn run() -> crate::error::Result<()> {
    let cli = Cli::parse();
    let mut lab = Lab::new()?;

    match cli.command {
        None | Some(Commands::Repl) => {
            let mut repl = super::repl::Repl::new(lab)?;
            repl.run()
        }
        Some(cmd) => execute_command(&mut lab, cmd),
    }
}

fn execute_command(lab: &mut Lab, cmd: Commands) -> crate::error::Result<()> {
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
                    let status = if *connected { "[connected]" } else { "[disconnected]" };
                    println!("  {:width$}  {}  {}", name, path.display(), status, width = max_name);
                }
            }
        }
        Commands::Query { jse, shelf } => {
            let json: serde_json::Value = serde_json::from_str(&jse)
                .map_err(|e| crate::error::HypatiaError::Parse(format!("invalid JSON: {e}")))?;
            let result = lab.query(&shelf, &json)?;
            print_result(&result);
        }
        Commands::KnowledgeCreate { name, data, tags, synonyms, figures, scopes, shelf } => {
            let tags_vec: Vec<String> = if tags.is_empty() {
                Vec::new()
            } else {
                tags.split(',').map(|s| s.trim().to_string()).collect()
            };
            let syn = if synonyms.is_empty() {
                None
            } else {
                let list: Vec<String> = synonyms.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                if list.is_empty() { None } else { Some(Synonyms::Flat(list)) }
            };
            let figures_vec: Vec<String> = if figures.is_empty() {
                Vec::new()
            } else {
                figures.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
            };
            let scopes_vec: Vec<String> = if scopes.is_empty() {
                Vec::new()
            } else {
                scopes.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
            };
            // Treat trailing comma as implicit global scope
            let scopes_vec = if scopes.ends_with(',') && !scopes_vec.contains(&String::new()) {
                let mut v = scopes_vec;
                v.push(String::new());
                v
            } else {
                scopes_vec
            };
            let content = Content::new(&data)
                .with_tags(tags_vec)
                .with_synonyms(syn)
                .with_figures(figures_vec)
                .with_scopes(scopes_vec);
            let k = lab.create_knowledge(&shelf, &name, content)?;
            println!("Created knowledge: {}", k.name);
        }
        Commands::KnowledgeGet { name, shelf } => {
            match lab.get_knowledge(&shelf, &name)? {
                Some(k) => {
                    let json = serde_json::to_string_pretty(&serde_json::json!({
                        "name": k.name,
                        "content": k.content,
                        "created_at": k.created_at.to_string(),
                    }))?;
                    println!("{json}");
                }
                None => println!("Knowledge '{}' not found.", name),
            }
        }
        Commands::KnowledgeDelete { name, shelf } => {
            lab.delete_knowledge(&shelf, &name)?;
            println!("Deleted knowledge: {name}");
        }
        Commands::StatementDelete { subject, predicate, object, shelf } => {
            let key = StatementKey::new(&subject, &predicate, &object);
            lab.delete_statement(&shelf, &key)?;
            println!("Deleted statement: ({}, {}, {})", subject, predicate, object);
        }
        Commands::StatementCreate { subject, predicate, object, data, synonyms, scopes, shelf } => {
            let key = StatementKey::new(&subject, &predicate, &object);
            let syn = match synonyms {
                Some(ref json_str) => {
                    let map: std::collections::HashMap<String, Vec<String>> =
                        serde_json::from_str(json_str)
                        .map_err(|e| crate::error::HypatiaError::Parse(
                            format!("invalid synonyms JSON: {e}")
                        ))?;
                    Some(Synonyms::Positional(map))
                }
                None => None,
            };
            let scopes_vec: Vec<String> = if scopes.is_empty() {
                Vec::new()
            } else {
                scopes.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
            };
            let scopes_vec = if scopes.ends_with(',') && !scopes_vec.contains(&String::new()) {
                let mut v = scopes_vec;
                v.push(String::new());
                v
            } else {
                scopes_vec
            };
            let content = Content::new(&data).with_synonyms(syn).with_scopes(scopes_vec);
            let s = lab.create_statement(&shelf, &key, content, None, None)?;
            println!("Created statement: ({}, {}, {})", s.key.subject, s.key.predicate, s.key.object);
        }
        Commands::Search { query, catalog, limit, offset, shelf } => {
            let opts = SearchOpts {
                catalog,
                limit,
                offset,
            };
            let result = lab.search(&shelf, &query, opts)?;
            print_result(&result);
        }
        Commands::Export { name, dest } => {
            lab.export_shelf(&name, &dest)?;
            println!("Exported shelf '{name}' to {}", dest.display());
        }
        Commands::Backfill { shelf } => {
            let stats = lab.backfill_vectors(&shelf)?;
            println!("Backfill complete: {} vectors created, {} skipped, {} errors",
                stats.created, stats.skipped, stats.errors);
        }
        Commands::ArchiveStore { file, name, shelf } => {
            let file_name = file.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "unnamed".to_string());
            let dest_relative = name.unwrap_or(file_name);

            // Store the file
            let abs_path = lab.store_archive(&shelf, &file, &dest_relative)?;

            // Determine MIME type from extension
            let ext = std::path::Path::new(&dest_relative)
                .extension()
                .map(|e| e.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            let mime_type = match ext.as_str() {
                "png" => "image/png",
                "jpg" | "jpeg" => "image/jpeg",
                "gif" => "image/gif",
                "svg" => "image/svg+xml",
                "webp" => "image/webp",
                "pdf" => "application/pdf",
                "mp4" => "video/mp4",
                "mp3" => "audio/mpeg",
                "wav" => "audio/wav",
                _ => "application/octet-stream",
            };
            let category = if mime_type.starts_with("image/") {
                "image"
            } else if mime_type.starts_with("video/") {
                "video"
            } else if mime_type.starts_with("audio/") {
                "audio"
            } else {
                "file"
            };

            // Get file size
            let size_bytes = std::fs::metadata(&abs_path)?.len();

            // Create knowledge with metadata
            let meta_data = serde_json::json!({
                "filename": dest_relative,
                "size_bytes": size_bytes,
                "mime_type": mime_type
            }).to_string();

            let content = Content::new(&meta_data)
                .with_format(crate::model::Format::Json)
                .with_tags(vec![
                    "archive".to_string(),
                    category.to_string(),
                    ext.clone(),
                ])
                .with_figures(vec![format!("archive://{}", dest_relative)]);

            let k = lab.create_knowledge(&shelf, &dest_relative, content)?;

            // Create statement: <name> is_a archive
            let key = StatementKey::new(&dest_relative, "is_a", "archive");
            let stmt_content = Content::new("")
                .with_tags(vec!["archive".to_string()]);
            let _ = lab.create_statement(&shelf, &key, stmt_content, None, None);

            println!("Stored: archive://{}", dest_relative);
            println!("Knowledge: {}", k.name);
            println!("MIME: {}, Size: {} bytes", mime_type, size_bytes);
        }
        Commands::ArchiveGet { name, output, shelf } => {
            match lab.get_archive_path(&shelf, &name) {
                Some(path) => {
                    match output {
                        Some(dest) => {
                            std::fs::copy(&path, &dest)?;
                            println!("Copied to: {}", dest.display());
                        }
                        None => {
                            println!("{}", path.display());
                        }
                    }
                }
                None => println!("Archive '{}' not found in shelf '{}'.", name, shelf),
            }
        }
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
        Commands::Repl => unreachable!(),
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;
    use std::collections::HashMap;

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
