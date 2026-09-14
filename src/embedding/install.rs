//! `hypatia model install`: fetch an ONNX embedding model from the Hugging Face Hub into the
//! models directory, without Python's `hf`. Downloads resume after an interruption, large
//! files are checked against the Hub's sha256, and a file gets its final name only once
//! complete. A manifest beside the files records which version of each is in place.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

type Response = ureq::http::Response<ureq::Body>;

/// Records, beside the model files, the commit and the version of each file in place.
const MANIFEST: &str = ".hypatia-install.json";
/// Locked while an install writes into a model directory.
const LOCK: &str = ".hypatia-install.lock";
/// Pages of one listing followed at most.
const MAX_PAGES: usize = 100;

/// A Hugging Face Hub endpoint, with the token for gated or private repositories.
pub struct Hub {
    endpoint: String,
    token: Option<String>,
}

/// A file in a model repository.
#[derive(Debug, Clone, PartialEq)]
pub struct RepoFile {
    pub path: String,
    pub size: u64,
    /// Known for files stored with Git LFS, as every large model file is.
    pub sha256: Option<String>,
    /// Git's id for this version of the file.
    pub oid: String,
}

impl RepoFile {
    /// What identifies this version of the file: its sha256 when known, else its git id.
    pub fn version(&self) -> &str {
        self.sha256.as_deref().unwrap_or(&self.oid)
    }
}

/// A repository at one commit.
pub struct Resolved {
    /// The repository's name as the Hub spells it.
    pub repo: String,
    pub commit: String,
}

/// What an install did.
#[derive(Debug)]
pub struct Installed {
    /// The repository's name as the Hub spells it.
    pub repo: String,
    pub dir: PathBuf,
    /// A file of an earlier install was replaced by another version, so vectors embedded
    /// with the earlier files no longer match.
    pub replaced: bool,
}

#[derive(Deserialize)]
struct TreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    oid: String,
    lfs: Option<LfsPointer>,
}

#[derive(Deserialize)]
struct LfsPointer {
    oid: String,
}

#[derive(Deserialize)]
struct RepoRevision {
    sha: String,
    id: Option<String>,
}

#[derive(Default, Serialize, Deserialize)]
struct Manifest {
    commit: String,
    /// Repository path → version of that file in place.
    files: BTreeMap<String, String>,
}

impl Hub {
    /// The Hub named by `HF_ENDPOINT` (a mirror, say), or huggingface.co, with `HF_TOKEN`.
    pub fn from_env() -> Self {
        let endpoint = std::env::var("HF_ENDPOINT")
            .ok()
            .filter(|endpoint| !endpoint.trim().is_empty())
            .unwrap_or_else(|| "https://huggingface.co".into());
        let token = std::env::var("HF_TOKEN")
            .ok()
            .filter(|token| !token.trim().is_empty());
        if token.is_some() && endpoint.trim().starts_with("http://") {
            eprintln!("warning: HF_TOKEN is sent unencrypted, because HF_ENDPOINT uses http");
        }
        Self::new(&endpoint, token)
    }

    pub fn new(endpoint: &str, token: Option<String>) -> Self {
        Self {
            endpoint: endpoint.trim().trim_end_matches('/').to_string(),
            token,
        }
    }

    /// A GET request, optionally for the bytes from `from` on. No overall timeout: a model
    /// file takes as long as it takes, and an interrupted download resumes.
    fn get(&self, url: &str, from: Option<u64>) -> Result<Response, String> {
        // Byte offsets must match the file, so nothing may compress it on the way.
        let mut request = ureq::get(url).header("Accept-Encoding", "identity");
        if let Some(token) = &self.token {
            request = request.header("Authorization", &format!("Bearer {token}"));
        }
        if let Some(from) = from {
            request = request.header("Range", &format!("bytes={from}-"));
        }
        request
            .config()
            .timeout_connect(Some(Duration::from_secs(30)))
            .timeout_recv_response(Some(Duration::from_secs(60)))
            .http_status_as_error(false)
            // The Hub redirects small files within itself, where the token is still needed;
            // it must never follow a redirect to another host, such as the CDN.
            .redirect_auth_headers(ureq::config::RedirectAuthHeaders::SameHost)
            .build()
            .call()
            .map_err(|e| format!("cannot reach the Hub: {e}"))
    }

    /// What a refusal means for the user, from its status and the Hub's error code.
    fn refusal(&self, response: &Response, what: &str) -> String {
        let code = response
            .headers()
            .get("x-error-code")
            .and_then(|code| code.to_str().ok());
        self.explain(response.status().as_u16(), code, what)
    }

    fn explain(&self, status: u16, code: Option<&str>, what: &str) -> String {
        match (status, code) {
            (_, Some("GatedRepo")) => format!(
                "{what}: the repository is gated; accept its terms on the Hub, then set HF_TOKEN to a token of that account"
            ),
            (404, _) | (_, Some("RepoNotFound" | "RevisionNotFound" | "EntryNotFound")) => {
                format!("{what}: not found")
            }
            // The Hub answers 401, not 404, for a repository that does not exist.
            (401 | 403, _) if self.token.is_none() => format!(
                "{what}: not found, or gated or private ({status}); check the name, or set HF_TOKEN"
            ),
            (401 | 403, _) => format!(
                "{what}: not found, or not readable with HF_TOKEN ({status}); check the name and the token"
            ),
            _ => format!("{what}: the Hub answered {status}"),
        }
    }

    /// The commit `revision` names in `repo`, so every file comes from the same one, with
    /// the repository's own spelling of its name.
    pub fn resolve(&self, repo: &str, revision: &str) -> Result<Resolved, String> {
        let url = format!(
            "{}/api/models/{repo}/revision/{}",
            self.endpoint,
            encode(revision)
        );
        let what = format!("{repo} at {revision}");
        let mut response = self.get(&url, None)?;
        if response.status().as_u16() >= 400 {
            return Err(self.refusal(&response, &what));
        }
        let answer: RepoRevision = response
            .body_mut()
            .read_json()
            .map_err(|e| format!("{what}: unexpected answer from the Hub: {e}"))?;
        if answer.sha.len() != 40 || !answer.sha.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(format!("{what}: unexpected commit id from the Hub"));
        }
        // Only a difference in case is taken from the Hub: it may not rename the install.
        let repo = answer
            .id
            .filter(|id| id.eq_ignore_ascii_case(repo) && validate_repo(id).is_ok())
            .unwrap_or_else(|| repo.to_string());
        Ok(Resolved {
            repo,
            commit: answer.sha,
        })
    }

    /// The files directly in `dir` (the root when empty) at `commit`; a missing directory
    /// has none.
    pub fn list(&self, repo: &str, commit: &str, dir: &str) -> Result<Vec<RepoFile>, String> {
        let what = format!("files of {repo}");
        let mut url = format!("{}/api/models/{repo}/tree/{commit}/{dir}", self.endpoint)
            .trim_end_matches('/')
            .to_string();
        let mut files = Vec::new();
        for _ in 0..MAX_PAGES {
            let mut response = self.get(&url, None)?;
            let status = response.status().as_u16();
            if status == 404 && !dir.is_empty() && files.is_empty() {
                return Ok(files);
            }
            if status >= 400 {
                return Err(self.refusal(&response, &what));
            }
            let next = self
                .next_page(&response)
                .map_err(|e| format!("{what}: {e}"))?;
            let entries: Vec<TreeEntry> = response
                .body_mut()
                .with_config()
                .limit(16 * 1024 * 1024)
                .read_json()
                .map_err(|e| format!("{what}: unexpected answer from the Hub: {e}"))?;
            files.extend(
                entries
                    .into_iter()
                    .filter(|entry| entry.kind == "file")
                    .map(|entry| RepoFile {
                        path: entry.path,
                        size: entry.size,
                        sha256: entry.lfs.map(|lfs| lfs.oid),
                        oid: entry.oid,
                    }),
            );
            match next {
                Some(next) => url = next,
                None => return Ok(files),
            }
        }
        Err(format!("{what}: more than {MAX_PAGES} pages"))
    }

    /// The next page of a long listing, from the `Link` header. Only pages on this Hub are
    /// followed, since the token goes with every request; a page elsewhere is an error, as
    /// stopping would pass part of the listing off as all of it.
    fn next_page(&self, response: &Response) -> Result<Option<String>, String> {
        let Some(link) = response
            .headers()
            .get("link")
            .and_then(|link| link.to_str().ok())
        else {
            return Ok(None);
        };
        let Some(next) = link.split(',').find(|part| part.contains("rel=\"next\"")) else {
            return Ok(None);
        };
        let target = next
            .split_once('<')
            .and_then(|(_, rest)| rest.split_once('>'))
            .map_or("", |(target, _)| target);
        let url = if target.starts_with('/') {
            format!("{}{target}", self.endpoint)
        } else {
            target.to_string()
        };
        if url.starts_with(&format!("{}/", self.endpoint)) {
            Ok(Some(url))
        } else {
            Err(format!(
                "the Hub continues the listing at another address ({url}), which is not followed"
            ))
        }
    }

    /// Downloads `file` into `dir` under its repository path. A partial download of the same
    /// version left by an earlier run is resumed, and the file gets its final name only
    /// once its size and checksum match. `progress` hears how many bytes are there so far.
    pub fn download(
        &self,
        repo: &str,
        commit: &str,
        file: &RepoFile,
        dir: &Path,
        progress: &mut dyn FnMut(u64),
    ) -> Result<(), String> {
        no_links(dir, Path::new(&file.path))?;
        let dest = dir.join(&file.path);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
        }
        let part = partial(&dest, file.version());
        if std::fs::symlink_metadata(&part).is_ok_and(|meta| meta.file_type().is_symlink()) {
            return Err(format!(
                "{} is a symbolic link; remove it and run the command again",
                part.display()
            ));
        }
        let mut have = std::fs::metadata(&part).map_or(0, |meta| meta.len());
        if have > file.size {
            have = 0;
        }
        let mut hash = Sha256::new();
        if have > 0 && file.sha256.is_some() {
            // The bytes already here count toward the checksum.
            hash_file(&part, &mut hash)
                .map_err(|e| format!("cannot read {}: {e}", part.display()))?;
        }
        if have < file.size {
            let url = format!("{}/{repo}/resolve/{commit}/{}", self.endpoint, file.path);
            let mut from = (have > 0).then_some(have);
            let mut response = loop {
                let response = self.get(&url, from)?;
                match response.status().as_u16() {
                    206 if from.is_some() && content_range_start(&response) == from => {
                        break response;
                    }
                    200 => break response,
                    // Resuming did not work out: start over.
                    206 | 416 if from.is_some() => from = None,
                    _ => return Err(self.refusal(&response, &file.path)),
                }
            };
            if response.status().as_u16() != 206 {
                have = 0;
                hash = Sha256::new();
            }
            let out = if have == 0 {
                std::fs::File::create(&part)
            } else {
                std::fs::OpenOptions::new().append(true).open(&part)
            };
            let mut out = out.map_err(|e| format!("cannot write {}: {e}", part.display()))?;
            let mut body = response.body_mut().as_reader();
            let mut buffer = vec![0u8; 1 << 20];
            loop {
                let read = body.read(&mut buffer).map_err(|e| {
                    format!(
                        "{}: download interrupted ({e}); run the command again to resume",
                        file.path
                    )
                })?;
                if read == 0 {
                    break;
                }
                out.write_all(&buffer[..read])
                    .map_err(|e| format!("cannot write {}: {e}", part.display()))?;
                hash.update(&buffer[..read]);
                have += read as u64;
                progress(have);
            }
            out.sync_all()
                .map_err(|e| format!("cannot write {}: {e}", part.display()))?;
        }
        if have != file.size {
            return Err(format!(
                "{}: received {have} of {} bytes; run the command again to resume",
                file.path, file.size
            ));
        }
        if let Some(expected) = &file.sha256 {
            let actual = hex(&hash.finalize());
            if !actual.eq_ignore_ascii_case(expected) {
                let _ = std::fs::remove_file(&part);
                return Err(format!(
                    "{}: checksum mismatch (expected {expected}, got {actual}); the download was discarded, run the command again",
                    file.path
                ));
            }
        }
        std::fs::rename(&part, &dest)
            .map_err(|e| format!("cannot move {} into place: {e}", dest.display()))
    }
}

/// Picks the files a local embedding model needs from a repository's root and `onnx/`
/// listings: `model.onnx` (preferring `onnx/`), its external data beside it (possibly split
/// into numbered parts), and the tokenizer beside the model, else at the root. The graph
/// comes last, so an interrupted install never looks like a usable model.
pub fn select(files: &[RepoFile]) -> Result<Vec<RepoFile>, String> {
    let find = |path: &str| files.iter().find(|file| file.path == path);
    let Some((dir, model)) = ["onnx/", ""]
        .into_iter()
        .find_map(|dir| find(&format!("{dir}model.onnx")).map(|model| (dir, model)))
    else {
        return Err(
            "no model.onnx at the root or in onnx/: hypatia runs ONNX models, and this repository has no ONNX export".into(),
        );
    };
    let tokenizer = find(&format!("{dir}tokenizer.json"))
        .or_else(|| find("tokenizer.json"))
        .ok_or_else(|| format!("no tokenizer.json beside {} or at the root", model.path))?;
    let is_data = |name: &str| {
        matches!(name, "model.onnx_data" | "model.onnx.data")
            || name
                .strip_prefix("model.onnx_data_")
                .is_some_and(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()))
    };
    let mut selected: Vec<RepoFile> = files
        .iter()
        .filter(|file| {
            file.path
                .strip_prefix(dir)
                .is_some_and(|name| !name.contains('/') && is_data(name))
        })
        .cloned()
        .collect();
    selected.sort_by(|a, b| a.path.cmp(&b.path));
    selected.push(tokenizer.clone());
    selected.push(model.clone());
    Ok(selected)
}

/// Installs `repo` at `revision` into `models_dir/<repo>`, with progress on stderr.
pub fn install(
    hub: &Hub,
    repo: &str,
    revision: &str,
    models_dir: &Path,
) -> Result<Installed, String> {
    validate_repo(repo)?;
    let Resolved { repo, commit } = hub.resolve(repo, revision)?;
    no_links(models_dir, Path::new(&repo))?;
    let dir = models_dir.join(&repo);
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let _lock = lock(&dir, &repo)?;
    let mut files = hub.list(&repo, &commit, "")?;
    files.extend(hub.list(&repo, &commit, "onnx")?);
    let selected = select(&files)?;
    if let Some(odd) = selected.iter().find(|file| {
        file.version().is_empty() || !file.version().bytes().all(|b| b.is_ascii_hexdigit())
    }) {
        return Err(format!("{}: unexpected file id from the Hub", odd.path));
    }

    let mut manifest = read_manifest(&dir);
    let mut replaced = false;
    let mut missing = Vec::new();
    for file in &selected {
        let path = dir.join(&file.path);
        let recorded = manifest.files.get(&file.path).map(String::as_str);
        let in_place =
            std::fs::metadata(&path).is_ok_and(|meta| meta.is_file() && meta.len() == file.size);
        if in_place && recorded == Some(file.version()) {
            continue;
        }
        // A file in place from before the manifest counts once its checksum matches.
        if recorded.is_none() && verified(&path, file) {
            manifest
                .files
                .insert(file.path.clone(), file.version().to_string());
            continue;
        }
        // Only another version changes the model; a missing or damaged file is fetched again.
        replaced |= recorded.is_some_and(|version| version != file.version());
        missing.push(file);
    }
    manifest.commit = commit.clone();
    if !missing.is_empty() {
        let total: u64 = missing.iter().map(|file| file.size).sum();
        eprintln!(
            "Installing {repo} ({}, {} files) into {}",
            human_size(total),
            missing.len(),
            dir.display()
        );
        for file in missing {
            let mut meter = Meter::new(&file.path, file.size);
            hub.download(&repo, &commit, file, &dir, &mut |have| meter.show(have))?;
            meter.finish();
            // Recorded as each file lands, so an interrupted install keeps what it verified.
            manifest
                .files
                .insert(file.path.clone(), file.version().to_string());
            write_manifest(&dir, &manifest)?;
        }
    }
    // Files of an earlier version that this one no longer uses could still be found first
    // (a tokenizer at the root, say), and partial downloads of other versions only take space.
    for path in manifest.files.keys() {
        let ours = Path::new(path)
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)));
        if ours
            && !selected.iter().any(|file| &file.path == path)
            && no_links(&dir, Path::new(path)).is_ok()
        {
            let _ = std::fs::remove_file(dir.join(path));
        }
    }
    for file in &selected {
        remove_stale_parts(&dir.join(&file.path), file.version());
    }
    manifest
        .files
        .retain(|path, _| selected.iter().any(|file| &file.path == path));
    write_manifest(&dir, &manifest)?;
    Ok(Installed {
        repo,
        dir,
        replaced,
    })
}

/// `Org/Name`, as the Hub names repositories, and nothing that could leave the models
/// directory.
pub(crate) fn validate_repo(repo: &str) -> Result<(), String> {
    let valid = |part: &str| {
        !part.is_empty()
            && part != "."
            && part != ".."
            && part
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    };
    match repo.split_once('/') {
        Some((org, name)) if valid(org) && valid(name) => Ok(()),
        _ => Err(format!(
            "'{repo}' is not a Hugging Face model name like BAAI/bge-m3"
        )),
    }
}

/// Refuses symbolic links along `relative` below `base`: an install writes only into
/// directories of its own, and a model registered with `model register` is a link.
fn no_links(base: &Path, relative: &Path) -> Result<(), String> {
    let mut path = base.to_path_buf();
    for component in relative.components() {
        path.push(component);
        if std::fs::symlink_metadata(&path).is_ok_and(|meta| meta.file_type().is_symlink()) {
            return Err(format!(
                "{} is a symbolic link (a registered model?); `hypatia model install` does not write through links",
                path.display()
            ));
        }
    }
    Ok(())
}

/// Keeps a second install of the same model from writing at the same time. The operating
/// system releases the lock when the process ends, however it ends.
fn lock(dir: &Path, repo: &str) -> Result<std::fs::File, String> {
    let path = dir.join(LOCK);
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&path)
        .map_err(|e| format!("cannot lock {}: {e}", path.display()))?;
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(std::fs::TryLockError::WouldBlock) => {
            Err(format!("another install of {repo} is running"))
        }
        // Some network and virtual filesystems cannot lock at all.
        Err(std::fs::TryLockError::Error(e)) if e.kind() == std::io::ErrorKind::Unsupported => {
            eprintln!(
                "warning: {} cannot be locked here ({e}); do not run two installs of {repo} at once",
                path.display()
            );
            Ok(file)
        }
        Err(std::fs::TryLockError::Error(e)) => Err(format!("cannot lock {}: {e}", path.display())),
    }
}

/// Deletes partial downloads of other versions of `dest`.
fn remove_stale_parts(dest: &Path, version: &str) {
    let (Some(dir), Some(name)) = (
        dest.parent(),
        dest.file_name().and_then(|name| name.to_str()),
    ) else {
        return;
    };
    let current = partial(dest, version);
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let stale = entry.file_name().to_str().is_some_and(|other| {
            other
                .strip_prefix(&format!("{name}."))
                .and_then(|rest| rest.strip_suffix(".part"))
                .is_some_and(|tag| {
                    !tag.is_empty() && tag.len() <= 16 && tag.bytes().all(|b| b.is_ascii_hexdigit())
                })
        });
        if stale && entry.path() != current {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

fn read_manifest(dir: &Path) -> Manifest {
    std::fs::read(dir.join(MANIFEST))
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

fn write_manifest(dir: &Path, manifest: &Manifest) -> Result<(), String> {
    let path = dir.join(MANIFEST);
    let temporary = dir.join(format!("{MANIFEST}.tmp"));
    let json = serde_json::to_vec_pretty(manifest).map_err(|e| e.to_string())?;
    std::fs::write(&temporary, json)
        .and_then(|()| std::fs::rename(&temporary, &path))
        .map_err(|e| format!("cannot write {}: {e}", path.display()))
}

/// Whether a file already in place is `file`'s version, by its checksum.
fn verified(path: &Path, file: &RepoFile) -> bool {
    let Some(expected) = &file.sha256 else {
        return false;
    };
    if !std::fs::metadata(path).is_ok_and(|meta| meta.is_file() && meta.len() == file.size) {
        return false;
    }
    let mut hash = Sha256::new();
    hash_file(path, &mut hash).is_ok() && hex(&hash.finalize()).eq_ignore_ascii_case(expected)
}

/// Where a download collects until it is verified, named after the version it is of.
fn partial(dest: &Path, version: &str) -> PathBuf {
    let tag: String = version
        .chars()
        .filter(char::is_ascii_hexdigit)
        .take(16)
        .collect();
    let mut name = dest.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".{tag}.part"));
    dest.with_file_name(name)
}

/// Where a partial response starts, from `Content-Range: bytes <start>-<end>/<size>`.
fn content_range_start(response: &Response) -> Option<u64> {
    response
        .headers()
        .get("content-range")?
        .to_str()
        .ok()?
        .strip_prefix("bytes ")?
        .split('-')
        .next()?
        .trim()
        .parse()
        .ok()
}

/// Percent-encodes everything but unreserved characters, for one URL path segment.
fn encode(text: &str) -> String {
    text.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}

fn hash_file(path: &Path, hash: &mut Sha256) -> std::io::Result<()> {
    let mut file = std::fs::File::open(path)?;
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            return Ok(());
        }
        hash.update(&buffer[..read]);
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

/// Download progress on stderr: redrawn in place on a terminal, a line per tenth elsewhere.
struct Meter<'a> {
    name: &'a str,
    size: u64,
    terminal: bool,
    drawn: Option<Instant>,
    tenths: u64,
}

impl<'a> Meter<'a> {
    fn new(name: &'a str, size: u64) -> Self {
        Self {
            name,
            size,
            terminal: std::io::stderr().is_terminal(),
            drawn: None,
            tenths: 0,
        }
    }

    fn show(&mut self, have: u64) {
        let percent = if self.size == 0 {
            100
        } else {
            have * 100 / self.size
        };
        if self.terminal {
            let recent = self
                .drawn
                .is_some_and(|drawn| drawn.elapsed() < Duration::from_millis(200));
            if recent && have < self.size {
                return;
            }
            self.drawn = Some(Instant::now());
            // Clear the rest of the line: a shorter figure must not leave old digits behind.
            eprint!(
                "\r  {}  {} / {}  {percent:>3}%\x1b[K",
                self.name,
                human_size(have),
                human_size(self.size)
            );
        } else if percent / 10 > self.tenths || self.drawn.is_none() {
            self.drawn = Some(Instant::now());
            self.tenths = percent / 10;
            eprintln!("  {}  {percent}%", self.name);
        }
    }

    fn finish(&mut self) {
        if self.terminal {
            eprintln!();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::{BufRead, BufReader};
    use std::net::{TcpListener, TcpStream};
    use std::sync::{Arc, Mutex};

    const COMMIT: &str = "c0ffeec0ffeec0ffeec0ffeec0ffeec0ffeec0ff";

    fn file(path: &str, size: u64, lfs: bool) -> RepoFile {
        RepoFile {
            path: path.into(),
            size,
            sha256: lfs.then(|| "0".repeat(64)),
            oid: "1".repeat(40),
        }
    }

    fn paths(files: &[RepoFile]) -> Vec<&str> {
        files.iter().map(|file| file.path.as_str()).collect()
    }

    #[test]
    fn picks_the_onnx_export_its_data_and_the_tokenizer_beside_it() {
        // BAAI/bge-m3: a different tokenizer at the root than beside the ONNX graph.
        let files = [
            file("tokenizer.json", 17, true),
            file("pytorch_model.bin", 99, true),
            file("onnx/model.onnx", 7, true),
            file("onnx/model.onnx_data", 2266, true),
            file("onnx/tokenizer.json", 17, true),
            file("onnx/config.json", 1, false),
            file("onnx/Constant_7_attr__value", 65, false),
        ];
        assert_eq!(
            paths(&select(&files).unwrap()),
            [
                "onnx/model.onnx_data",
                "onnx/tokenizer.json",
                "onnx/model.onnx"
            ]
        );
        // jinaai/jina-embeddings-v5: the tokenizer only at the root.
        let files = [
            file("tokenizer.json", 11, true),
            file("onnx/model.onnx", 1, true),
            file("onnx/model.onnx_data", 2384, true),
            file("model.safetensors", 1192, true),
        ];
        assert_eq!(
            paths(&select(&files).unwrap()),
            ["onnx/model.onnx_data", "tokenizer.json", "onnx/model.onnx"]
        );
        // An export too large for one data file, split into numbered parts.
        let files = [
            file("onnx/model.onnx", 1, true),
            file("onnx/model.onnx_data_2", 5, true),
            file("onnx/model.onnx_data", 5, true),
            file("onnx/model.onnx_data_1", 5, true),
            file("onnx/model.onnx_data_backup", 5, true),
            file("model.onnx_data_1", 5, true),
            file("tokenizer.json", 1, false),
        ];
        assert_eq!(
            paths(&select(&files).unwrap()),
            [
                "onnx/model.onnx_data",
                "onnx/model.onnx_data_1",
                "onnx/model.onnx_data_2",
                "tokenizer.json",
                "onnx/model.onnx"
            ]
        );
        // A small model at the root, without external data.
        let files = [
            file("model.onnx", 5, true),
            file("tokenizer.json", 1, false),
        ];
        assert_eq!(
            paths(&select(&files).unwrap()),
            ["tokenizer.json", "model.onnx"]
        );
        let unexported = [
            file("model.safetensors", 1, true),
            file("tokenizer.json", 1, false),
        ];
        assert!(select(&unexported).unwrap_err().contains("model.onnx"));
    }

    #[test]
    fn repository_names_cannot_leave_the_models_directory() {
        for bad in [
            "bge-m3",
            "../etc",
            "BAAI/../../x",
            "BAAI/bge m3",
            "/abs/path",
            "BAAI/",
            "BAAI/..",
        ] {
            assert!(validate_repo(bad).is_err(), "{bad}");
        }
        for good in [
            "BAAI/bge-m3",
            "jinaai/jina-embeddings-v5-text-nano-text-matching",
        ] {
            assert!(validate_repo(good).is_ok(), "{good}");
        }
        assert_eq!(encode("refs/pr/1 #?%"), "refs%2Fpr%2F1%20%23%3F%25");
    }

    #[test]
    fn refusals_say_what_to_do() {
        let anonymous = Hub::new("https://hub.test/", None);
        assert_eq!(anonymous.endpoint, "https://hub.test");
        assert!(
            anonymous
                .explain(401, None, "x")
                .contains("check the name, or set HF_TOKEN")
        );
        assert!(
            anonymous
                .explain(401, Some("GatedRepo"), "x")
                .contains("gated")
        );
        assert_eq!(
            anonymous.explain(401, Some("RepoNotFound"), "x"),
            "x: not found"
        );
        let signed_in = Hub::new("https://hub.test", Some("token".into()));
        assert!(
            signed_in
                .explain(403, None, "x")
                .contains("not readable with HF_TOKEN")
        );
    }

    /// One request a mock Hub received.
    #[derive(Clone, Debug)]
    struct Seen {
        path: String,
        range: Option<String>,
        authorization: Option<String>,
    }

    /// How a mock Hub behaves; tests may change it between installs.
    #[derive(Clone)]
    struct Mock {
        files: Vec<(&'static str, Vec<u8>, bool)>,
        /// Flip the first byte of this file in what is sent.
        corrupt: Option<&'static str>,
        /// Answer a range request for this file from its start instead.
        misplaced_range: Option<&'static str>,
        /// Serve files only with this bearer token, as for a gated repository.
        token: Option<&'static str>,
        /// Continue the root listing on another host.
        foreign_next: bool,
    }

    fn mock(files: Vec<(&'static str, Vec<u8>, bool)>) -> Arc<Mutex<Mock>> {
        Arc::new(Mutex::new(Mock {
            files,
            corrupt: None,
            misplaced_range: None,
            token: None,
            foreign_next: false,
        }))
    }

    fn model_files() -> Vec<(&'static str, Vec<u8>, bool)> {
        vec![
            ("onnx/model.onnx", vec![7u8; 3000], true),
            (
                "onnx/model.onnx_data",
                (0..10_000u32).map(|i| (i % 251) as u8).collect(),
                true,
            ),
            ("onnx/tokenizer.json", br#"{"model":{}}"#.to_vec(), false),
            ("README.md", b"readme".to_vec(), false),
        ]
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        hex(&Sha256::digest(bytes))
    }

    /// Serves `org/model` at `COMMIT` the way the Hub does: a paged root listing, downloads
    /// through a redirect on the same host, and Range support.
    fn serve_hub(hub: Arc<Mutex<Mock>>) -> (String, Arc<Mutex<Vec<Seen>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let seen: Arc<Mutex<Vec<Seen>>> = Arc::default();
        let log = seen.clone();
        let url = base.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { return };
                let Some(request) = read_head(&stream) else {
                    continue;
                };
                log.lock().unwrap().push(request.clone());
                let behaviour = hub.lock().unwrap().clone();
                let (status, headers, body) = answer(&behaviour, &base, &request);
                let mut head = format!(
                    "HTTP/1.1 {status} Test\r\nContent-Length: {}\r\nConnection: close\r\n",
                    body.len()
                );
                for (name, value) in headers {
                    head.push_str(&format!("{name}: {value}\r\n"));
                }
                head.push_str("\r\n");
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(&body);
            }
        });
        (url, seen)
    }

    fn read_head(stream: &TcpStream) -> Option<Seen> {
        let mut reader = BufReader::new(stream.try_clone().ok()?);
        let mut line = String::new();
        reader.read_line(&mut line).ok()?;
        let path = line.split_whitespace().nth(1)?.to_string();
        let mut seen = Seen {
            path,
            range: None,
            authorization: None,
        };
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).ok()?;
            let line = line.trim_end();
            if line.is_empty() {
                return Some(seen);
            }
            let Some((name, value)) = line.split_once(':') else {
                continue;
            };
            if name.eq_ignore_ascii_case("range") {
                seen.range = Some(value.trim().to_string());
            } else if name.eq_ignore_ascii_case("authorization") {
                seen.authorization = Some(value.trim().to_string());
            }
        }
    }

    type Answer = (u16, Vec<(String, String)>, Vec<u8>);

    fn answer(hub: &Mock, base: &str, request: &Seen) -> Answer {
        let entry = |(name, bytes, lfs): &(&str, Vec<u8>, bool)| {
            let digest = sha256_hex(bytes);
            let mut entry =
                json!({"type": "file", "path": name, "size": bytes.len(), "oid": &digest[..40]});
            if *lfs {
                entry["lfs"] = json!({"oid": digest});
            }
            entry
        };
        let in_dir = |dir: &str| -> Vec<serde_json::Value> {
            hub.files
                .iter()
                .filter(|(name, _, _)| Path::new(name).parent().unwrap().to_str() == Some(dir))
                .map(entry)
                .collect()
        };
        let json_body = |value: serde_json::Value| -> Answer {
            (200, Vec::new(), serde_json::to_vec(&value).unwrap())
        };
        let tree = format!("/api/models/org/model/tree/{COMMIT}");
        let resolve = format!("/org/model/resolve/{COMMIT}/");
        let cache = "/api/resolve-cache/";
        let path = request.path.as_str();
        if matches!(
            path,
            "/api/models/org/model/revision/main" | "/api/models/ORG/Model/revision/main"
        ) {
            return json_body(json!({"sha": COMMIT, "id": "org/model"}));
        }
        if path == tree {
            // An empty first page that links to the rest.
            let host = if hub.foreign_next {
                "http://elsewhere.test"
            } else {
                base
            };
            let link = format!("<{host}{tree}?cursor=2>; rel=\"next\"");
            return (200, vec![("Link".into(), link)], b"[]".to_vec());
        }
        if path == format!("{tree}?cursor=2") {
            let mut root = in_dir("");
            root.push(json!({"type": "directory", "path": "onnx"}));
            return json_body(serde_json::Value::Array(root));
        }
        if path == format!("{tree}/onnx") {
            return json_body(serde_json::Value::Array(in_dir("onnx")));
        }
        if let Some(name) = path.strip_prefix(&resolve) {
            return (
                307,
                vec![("Location".into(), format!("{cache}{name}"))],
                Vec::new(),
            );
        }
        let Some(name) = path.strip_prefix(cache) else {
            return (404, Vec::new(), Vec::new());
        };
        if let Some(token) = hub.token
            && request.authorization.as_deref() != Some(format!("Bearer {token}").as_str())
        {
            return (
                401,
                vec![("X-Error-Code".into(), "GatedRepo".into())],
                Vec::new(),
            );
        }
        let Some((_, bytes, _)) = hub.files.iter().find(|(file, _, _)| *file == name) else {
            return (404, Vec::new(), Vec::new());
        };
        let mut bytes = bytes.clone();
        if hub.corrupt == Some(name) {
            bytes[0] ^= 1;
        }
        let size = bytes.len();
        let from = request
            .range
            .as_deref()
            .and_then(|range| range.strip_prefix("bytes="))
            .and_then(|range| range.strip_suffix('-'))
            .and_then(|from| from.parse::<usize>().ok())
            .filter(|&from| from < size);
        let content_range = |from: usize| {
            (
                "Content-Range".to_string(),
                format!("bytes {from}-{}/{size}", size - 1),
            )
        };
        match from {
            Some(_) if hub.misplaced_range == Some(name) => (206, vec![content_range(0)], bytes),
            Some(from) => (206, vec![content_range(from)], bytes[from..].to_vec()),
            None => (200, Vec::new(), bytes),
        }
    }

    fn downloads_since(seen: &Arc<Mutex<Vec<Seen>>>, since: usize) -> Vec<String> {
        seen.lock().unwrap()[since..]
            .iter()
            .filter_map(|request| request.path.strip_prefix("/api/resolve-cache/"))
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn installs_what_is_needed_resuming_and_skipping_what_is_in_place() {
        let files = model_files();
        let hub = mock(files.clone());
        let (url, seen) = serve_hub(hub.clone());
        let models = tempfile::tempdir().unwrap();
        let dir = models.path().join("org/model");
        // A download of this very version, interrupted halfway by an earlier run.
        let data = &files[1].1;
        let part = dir.join(format!(
            "onnx/model.onnx_data.{}.part",
            &sha256_hex(data)[..16]
        ));
        std::fs::create_dir_all(part.parent().unwrap()).unwrap();
        std::fs::write(&part, &data[..4000]).unwrap();

        let hub_client = Hub::new(&url, None);
        let installed = install(&hub_client, "ORG/Model", "main", models.path()).unwrap();
        assert_eq!(
            (installed.repo.as_str(), installed.replaced),
            ("org/model", false)
        );
        assert_eq!(installed.dir, dir);
        for (name, bytes, _) in &files[..3] {
            assert_eq!(&std::fs::read(dir.join(name)).unwrap(), bytes, "{name}");
        }
        assert!(!part.exists() && !dir.join("README.md").exists());
        let requests = seen.lock().unwrap().clone();
        assert!(requests.iter().any(|request| {
            request.path.ends_with("resolve-cache/onnx/model.onnx_data")
                && request.range.as_deref() == Some("bytes=4000-")
        }));
        // The graph, which makes the directory a model, came last.
        assert_eq!(
            downloads_since(&seen, 0).last().map(String::as_str),
            Some("onnx/model.onnx")
        );

        // A second run downloads nothing.
        let before = seen.lock().unwrap().len();
        let again = install(&hub_client, "org/model", "main", models.path()).unwrap();
        assert!(!again.replaced);
        assert!(downloads_since(&seen, before).is_empty());

        // A new version of one file replaces just that file, and says so.
        hub.lock().unwrap().files[2].1 = br#"{"model":{"version":2}}"#.to_vec();
        let before = seen.lock().unwrap().len();
        assert!(
            install(&hub_client, "org/model", "main", models.path())
                .unwrap()
                .replaced
        );
        assert_eq!(downloads_since(&seen, before), ["onnx/tokenizer.json"]);
        assert_eq!(
            std::fs::read(dir.join("onnx/tokenizer.json")).unwrap(),
            br#"{"model":{"version":2}}"#
        );
    }

    #[test]
    fn large_files_in_place_without_a_manifest_are_verified_not_downloaded() {
        let files = model_files();
        let (url, seen) = serve_hub(mock(files.clone()));
        let models = tempfile::tempdir().unwrap();
        let dir = models.path().join("org/model");
        std::fs::create_dir_all(dir.join("onnx")).unwrap();
        for (name, bytes, _) in &files[..2] {
            std::fs::write(dir.join(name), bytes).unwrap();
        }
        install(&Hub::new(&url, None), "org/model", "main", models.path()).unwrap();
        assert_eq!(downloads_since(&seen, 0), ["onnx/tokenizer.json"]);
    }

    #[test]
    fn a_resumed_download_that_starts_elsewhere_starts_over() {
        let files = model_files();
        let hub = mock(files.clone());
        hub.lock().unwrap().misplaced_range = Some("onnx/model.onnx_data");
        let (url, seen) = serve_hub(hub);
        let models = tempfile::tempdir().unwrap();
        let data = &files[1].1;
        let part = models.path().join(format!(
            "org/model/onnx/model.onnx_data.{}.part",
            &sha256_hex(data)[..16]
        ));
        std::fs::create_dir_all(part.parent().unwrap()).unwrap();
        std::fs::write(&part, &data[..4000]).unwrap();
        install(&Hub::new(&url, None), "org/model", "main", models.path()).unwrap();
        assert_eq!(
            &std::fs::read(models.path().join("org/model/onnx/model.onnx_data")).unwrap(),
            data
        );
        let ranges: Vec<Option<String>> = seen
            .lock()
            .unwrap()
            .iter()
            .filter(|request| request.path.ends_with("resolve-cache/onnx/model.onnx_data"))
            .map(|request| request.range.clone())
            .collect();
        assert_eq!(ranges, [Some("bytes=4000-".to_string()), None]);
    }

    #[test]
    fn missing_or_damaged_files_are_fetched_again_without_calling_it_a_new_model() {
        let (url, seen) = serve_hub(mock(model_files()));
        let models = tempfile::tempdir().unwrap();
        let hub = Hub::new(&url, None);
        install(&hub, "org/model", "main", models.path()).unwrap();
        let dir = models.path().join("org/model");
        std::fs::remove_file(dir.join("onnx/tokenizer.json")).unwrap();
        std::fs::write(dir.join("onnx/model.onnx"), b"truncated").unwrap();
        let before = seen.lock().unwrap().len();
        assert!(
            !install(&hub, "org/model", "main", models.path())
                .unwrap()
                .replaced
        );
        assert_eq!(
            downloads_since(&seen, before),
            ["onnx/tokenizer.json", "onnx/model.onnx"]
        );
        assert_eq!(
            std::fs::read(dir.join("onnx/model.onnx")).unwrap(),
            vec![7u8; 3000]
        );
    }

    #[test]
    fn superseded_files_and_stale_partial_downloads_are_removed() {
        let hub = mock(model_files());
        let (url, _seen) = serve_hub(hub.clone());
        let models = tempfile::tempdir().unwrap();
        let client = Hub::new(&url, None);
        install(&client, "org/model", "main", models.path()).unwrap();
        let dir = models.path().join("org/model");
        let stale = dir.join("onnx/model.onnx_data.deadbeefdeadbeef.part");
        std::fs::write(&stale, b"old").unwrap();
        // The next version keeps its tokenizer at the root instead.
        hub.lock().unwrap().files[2].0 = "tokenizer.json";
        install(&client, "org/model", "main", models.path()).unwrap();
        assert!(dir.join("tokenizer.json").exists());
        assert!(!dir.join("onnx/tokenizer.json").exists());
        assert!(!stale.exists());
    }

    #[test]
    fn a_listing_continued_on_another_host_is_an_error() {
        let hub = mock(model_files());
        hub.lock().unwrap().foreign_next = true;
        let (url, _seen) = serve_hub(hub);
        let models = tempfile::tempdir().unwrap();
        let err = install(&Hub::new(&url, None), "org/model", "main", models.path()).unwrap_err();
        assert!(err.contains("not followed"), "{err}");
    }

    #[test]
    fn the_token_follows_redirects_on_the_same_host() {
        let hub = mock(model_files());
        hub.lock().unwrap().token = Some("secret");
        let (url, _seen) = serve_hub(hub);
        let models = tempfile::tempdir().unwrap();
        let err = install(&Hub::new(&url, None), "org/model", "main", models.path()).unwrap_err();
        assert!(err.contains("gated"), "{err}");
        let signed_in = Hub::new(&url, Some("secret".into()));
        install(&signed_in, "org/model", "main", models.path()).unwrap();
    }

    #[test]
    fn a_corrupt_download_is_discarded() {
        let hub = mock(vec![
            ("onnx/model.onnx", vec![1; 100], true),
            ("onnx/tokenizer.json", b"{}".to_vec(), false),
        ]);
        hub.lock().unwrap().corrupt = Some("onnx/model.onnx");
        let (url, _seen) = serve_hub(hub);
        let models = tempfile::tempdir().unwrap();
        let err = install(&Hub::new(&url, None), "org/model", "main", models.path()).unwrap_err();
        assert!(err.contains("checksum mismatch"), "{err}");
        let leftovers: Vec<_> = std::fs::read_dir(models.path().join("org/model/onnx"))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .filter(|name| name.starts_with("model.onnx"))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    #[test]
    fn an_unknown_repository_is_reported() {
        let (url, _seen) = serve_hub(mock(Vec::new()));
        let models = tempfile::tempdir().unwrap();
        let err = install(&Hub::new(&url, None), "org/missing", "main", models.path()).unwrap_err();
        assert!(err.contains("not found"), "{err}");
    }

    #[test]
    fn installs_stay_out_of_links_and_of_each_other() {
        let (url, _seen) = serve_hub(mock(model_files()));
        let models = tempfile::tempdir().unwrap();
        let hub = Hub::new(&url, None);
        #[cfg(unix)]
        {
            let elsewhere = tempfile::tempdir().unwrap();
            std::os::unix::fs::symlink(elsewhere.path(), models.path().join("org")).unwrap();
            let err = install(&hub, "org/model", "main", models.path()).unwrap_err();
            assert!(err.contains("symbolic link"), "{err}");
            std::fs::remove_file(models.path().join("org")).unwrap();
        }
        let dir = models.path().join("org/model");
        std::fs::create_dir_all(&dir).unwrap();
        let held = lock(&dir, "org/model").unwrap();
        let err = install(&hub, "org/model", "main", models.path()).unwrap_err();
        assert!(err.contains("another install"), "{err}");
        drop(held);
        install(&hub, "org/model", "main", models.path()).unwrap();
    }
}
