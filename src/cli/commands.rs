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
        #[arg(short, long, default_value = "")]
        synonyms: String,
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
        #[arg(short, long)]
        synonyms: Option<String>,
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
            println!("Connected to shelf: {shelf_name}");
        }
        Commands::Disconnect { name } => {
            lab.disconnect_shelf(&name)?;
            println!("Disconnected from shelf: {name}");
        }
        Commands::List => {
            let shelves = lab.list_shelves();
            if shelves.is_empty() {
                println!("No shelves connected.");
            } else {
                for name in &shelves {
                    println!("  {name}");
                }
            }
        }
        Commands::Query { jse, shelf } => {
            let json: serde_json::Value = serde_json::from_str(&jse)
                .map_err(|e| crate::error::HypatiaError::Parse(format!("invalid JSON: {e}")))?;
            let result = lab.query(&shelf, &json)?;
            print_result(&result);
        }
        Commands::KnowledgeCreate { name, data, tags, synonyms, shelf } => {
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
            let content = Content::new(&data).with_tags(tags_vec).with_synonyms(syn);
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
        Commands::StatementCreate { subject, predicate, object, data, synonyms, shelf } => {
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
            let content = Content::new(&data).with_synonyms(syn);
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
