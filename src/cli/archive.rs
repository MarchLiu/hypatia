//! Storing a file as an archive entry, shared by the CLI and the MCP server.

use std::path::Path;

use crate::error::Result;
use crate::lab::Lab;
use crate::model::{Content, Format, StatementKey};

/// What `store` recorded.
pub(crate) struct Archived {
    pub uri: String,
    pub knowledge: String,
    pub mime_type: &'static str,
    pub size_bytes: u64,
}

/// Copies `file` into the shelf's archives under `name` (default: its file name), records a
/// JSON knowledge entry describing it, and links that entry `is_a archive`.
pub(crate) fn store(
    lab: &mut Lab,
    shelf: &str,
    file: &Path,
    name: Option<String>,
) -> Result<Archived> {
    let file_name = file
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "unnamed".to_string());
    let dest_relative = name.unwrap_or(file_name);

    let abs_path = lab.store_archive(shelf, file, &dest_relative)?;

    let ext = Path::new(&dest_relative)
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

    let size_bytes = std::fs::metadata(&abs_path)?.len();

    let meta_data = serde_json::json!({
        "filename": dest_relative,
        "size_bytes": size_bytes,
        "mime_type": mime_type
    })
    .to_string();
    let content = Content::new(&meta_data)
        .with_format(Format::Json)
        .with_tags(vec![
            "archive".to_string(),
            category.to_string(),
            ext.clone(),
        ])
        .with_figures(vec![format!("archive://{}", dest_relative)]);
    let k = lab.create_knowledge(shelf, &dest_relative, content)?;

    let key = StatementKey::new(&dest_relative, "is_a", "archive");
    let stmt_content = Content::new("").with_tags(vec!["archive".to_string()]);
    let _ = lab.create_statement(shelf, &key, stmt_content, None, None);

    Ok(Archived {
        uri: format!("archive://{dest_relative}"),
        knowledge: k.name,
        mime_type,
        size_bytes,
    })
}
