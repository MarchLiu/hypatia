use crate::error::Result;
use crate::lab::Lab;

pub struct Repl {
    lab: Lab,
    editor: rustyline::DefaultEditor,
}

impl Repl {
    pub fn new(lab: Lab) -> Result<Self> {
        let editor = rustyline::DefaultEditor::new()
            .map_err(|e| crate::error::HypatiaError::Eval(format!("readline error: {e}")))?;
        Ok(Self { lab, editor })
    }

    pub fn run(&mut self) -> Result<()> {
        println!("hypatia 0.1.0 — AI-oriented memory management");
        println!("Type .help for commands, or enter JSE queries as JSON.");

        loop {
            let line = match self.editor.readline("hypatia> ") {
                Ok(line) => line,
                Err(rustyline::error::ReadlineError::Eof) => break,
                Err(rustyline::error::ReadlineError::Interrupted) => {
                    println!("(interrupted, type .quit to exit)");
                    continue;
                }
                Err(e) => {
                    eprintln!("Error: {e}");
                    break;
                }
            };

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            self.editor.add_history_entry(trimmed).ok();

            if trimmed.starts_with('.') {
                if let Err(e) = self.handle_dot_command(trimmed) {
                    eprintln!("Error: {e}");
                }
            } else if trimmed.starts_with('{') || trimmed.starts_with('[') {
                match self.handle_jse(trimmed) {
                    Ok(output) => println!("{output}"),
                    Err(e) => eprintln!("Error: {e}"),
                }
            } else {
                eprintln!("Unknown input. Start with {{ or [ for JSE queries, or . for commands.");
            }
        }

        Ok(())
    }

    fn handle_dot_command(&mut self, input: &str) -> Result<()> {
        let parts: Vec<&str> = input[1..].splitn(3, ' ').collect();
        let cmd = parts.first().copied().unwrap_or("");

        match cmd {
            "help" => {
                println!("Commands:");
                println!("  .connect <path> [name]   Connect to a shelf");
                println!("  .disconnect <name>       Disconnect from a shelf");
                println!("  .list                    List connected shelves");
                println!("  .export <name> <dest>    Export a shelf");
                println!("  .quit                    Exit REPL");
                println!();
                println!("JSE queries:");
                println!("  Enter JSON starting with {{ or [ to execute JSE queries.");
                println!("  Example: [\"$knowledge\", [\"$eq\", \"name\", \"test\"]]");
            }
            "connect" => {
                let path = parts.get(1).ok_or_else(|| {
                    crate::error::HypatiaError::Validation("usage: .connect <path> [name]".to_string())
                })?;
                let name = parts.get(2).map(|s| s.to_string());
                let shelf_name = self.lab.connect_shelf(
                    std::path::Path::new(path),
                    name.as_deref(),
                )?;
                println!("Connected to shelf: {shelf_name}");
            }
            "disconnect" => {
                let name = parts.get(1).ok_or_else(|| {
                    crate::error::HypatiaError::Validation("usage: .disconnect <name>".to_string())
                })?;
                self.lab.disconnect_shelf(name)?;
                println!("Disconnected from: {name}");
            }
            "list" => {
                let shelves = self.lab.list_shelves();
                if shelves.is_empty() {
                    println!("No shelves registered.");
                } else {
                    let max_name = shelves.iter().map(|(n, _, _)| n.len()).max().unwrap_or(0);
                    for (name, path, connected) in &shelves {
                        let status = if *connected { "[connected]" } else { "[disconnected]" };
                        println!("  {:width$}  {}  {}", name, path.display(), status, width = max_name);
                    }
                }
            }
            "export" => {
                let name = parts.get(1).ok_or_else(|| {
                    crate::error::HypatiaError::Validation("usage: .export <name> <dest>".to_string())
                })?;
                let dest = parts.get(2).ok_or_else(|| {
                    crate::error::HypatiaError::Validation("usage: .export <name> <dest>".to_string())
                })?;
                self.lab.export_shelf(name, std::path::Path::new(dest))?;
                println!("Exported shelf '{name}' to {dest}");
            }
            "quit" | "exit" => {
                println!("Goodbye.");
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown command: .{cmd}. Type .help for available commands.");
            }
        }
        Ok(())
    }

    fn handle_jse(&mut self, input: &str) -> Result<String> {
        let json: serde_json::Value = serde_json::from_str(input)
            .map_err(|e| crate::error::HypatiaError::Parse(format!("invalid JSON: {e}")))?;
        let result = self.lab.query("default", &json)?;
        if result.rows.is_empty() {
            Ok("No results found.".to_string())
        } else {
            Ok(serde_json::to_string_pretty(&result.rows)?)
        }
    }
}
