use std::process::ExitCode;

fn main() -> ExitCode {
    match hypatia::cli::run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}
