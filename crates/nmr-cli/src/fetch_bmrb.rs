//! Clean-room BMRB NMR-STAR and raw time-domain (FID) archive fetcher (`nmrassign fetch-bmrb`).
//!
//! Provenance: Original clean-room implementation of `docs/specs/11-fid-processing.md` section 5.1
//! with zero external Cargo dependencies, citing:
//! - NIST FIPS PUB 180-4 (*Secure Hash Standard*, National Institute of Standards and Technology,
//!   August 2015, sections 4.2.2, 5.1.1, 5.3.3, and 6.2) for streaming SHA-256 message digest computation.
//! - Biological Magnetic Resonance Data Bank (BMRB) public archive and REST API v2 layout:
//!   Hoch, J. C. et al., *Nucleic Acids Res.* **2023**, 51, D368–D376;
//!   Ulrich, E. L. et al., *Nucleic Acids Res.* **2008**, 36, D402–D408.
//! - POSIX.1-1988 `ustar` / Info-ZIP archive member path validation (rejecting leading `/`, `\`,
//!   drive prefixes, NUL bytes, `..` traversal components, and escaping symbolic/hard links prior to extraction).

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Required User-Agent header sent on all BMRB HTTP/HTTPS requests.
pub const BMRB_USER_AGENT: &str = "nmrassign-fetch";

/// Minimum delay between consecutive HTTP requests (`>= 1000 ms`, enforcing `<= 1 req/s`).
pub const MIN_HTTP_INTERVAL_MS: u64 = 1050;

/// Default safety ceiling (`5 GiB`) when `--max-bytes` is not overridden.
pub const DEFAULT_MAX_DOWNLOAD_BYTES: u64 = 5 * 1024 * 1024 * 1024;

const DEFAULT_BMRB_FTP_BASE: &str = "https://bmrb.io/ftp/pub/bmrb";
const DEFAULT_BMRB_API_BASE: &str = "https://api.bmrb.io/v2";

// ============================================================================
// Clean-Room FIPS PUB 180-4 SHA-256 Implementation
// ============================================================================

const SHA256_H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// Streaming FIPS 180-4 SHA-256 state.
#[derive(Clone)]
pub struct Sha256 {
    state: [u32; 8],
    block: [u8; 64],
    block_len: usize,
    total_bytes: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    pub fn new() -> Self {
        Self {
            state: SHA256_H0,
            block: [0u8; 64],
            block_len: 0,
            total_bytes: 0,
        }
    }

    fn compress_block(state: &mut [u32; 8], chunk: &[u8; 64]) {
        let mut w = [0u32; 64];
        for (t, slot) in w.iter_mut().take(16).enumerate() {
            let b = t * 4;
            *slot = u32::from_be_bytes([chunk[b], chunk[b + 1], chunk[b + 2], chunk[b + 3]]);
        }
        for t in 16..64 {
            let s0 = w[t - 15].rotate_right(7) ^ w[t - 15].rotate_right(18) ^ (w[t - 15] >> 3);
            let s1 = w[t - 2].rotate_right(17) ^ w[t - 2].rotate_right(19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16]
                .wrapping_add(s0)
                .wrapping_add(w[t - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
        for t in 0..64 {
            let sigma1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(sigma1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[t])
                .wrapping_add(w[t]);
            let sigma0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = sigma0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        for (st, add) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *st = st.wrapping_add(add);
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.total_bytes = self.total_bytes.wrapping_add(data.len() as u64);
        if self.block_len > 0 {
            let take = (64 - self.block_len).min(data.len());
            self.block[self.block_len..self.block_len + take].copy_from_slice(&data[..take]);
            self.block_len += take;
            data = &data[take..];
            if self.block_len == 64 {
                Self::compress_block(&mut self.state, &self.block);
                self.block_len = 0;
            }
        }
        while data.len() >= 64 {
            let chunk: &[u8; 64] = data[..64].try_into().unwrap();
            Self::compress_block(&mut self.state, chunk);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.block[..data.len()].copy_from_slice(data);
            self.block_len = data.len();
        }
    }

    pub fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total_bytes.wrapping_mul(8);
        self.block[self.block_len] = 0x80;
        self.block_len += 1;
        if self.block_len > 56 {
            self.block[self.block_len..64].fill(0);
            Self::compress_block(&mut self.state, &self.block);
            self.block_len = 0;
        }
        self.block[self.block_len..56].fill(0);
        self.block[56..64].copy_from_slice(&bit_len.to_be_bytes());
        Self::compress_block(&mut self.state, &self.block);

        let mut out = [0u8; 32];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }
}

pub fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    bytes_to_hex(&h.finalize())
}

pub fn sha256_file_and_size(path: &Path) -> Result<(String, u64), String> {
    let mut f = File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    let mut total = 0u64;
    loop {
        let n = f
            .read(&mut buf)
            .map_err(|e| format!("failed reading {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total += n as u64;
    }
    Ok((bytes_to_hex(&hasher.finalize()), total))
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

// ============================================================================
// UTC Timestamp Formatting (Howard Hinnant Civil-from-Days, Zero Dependencies)
// ============================================================================

pub fn format_utc_timestamp(unix_secs: u64) -> String {
    let days = (unix_secs / 86_400) as i64;
    let rem = unix_secs % 86_400;
    let hour = rem / 3600;
    let minute = (rem % 3600) / 60;
    let second = rem % 60;

    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };

    format!("{year:04}-{m:02}-{d:02}T{hour:02}:{minute:02}:{second:02}Z")
}

fn now_utc_iso8601() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format_utc_timestamp(secs)
}

// ============================================================================
// Rate-Limited Curl Client (`User-Agent: nmrassign-fetch`, `<= 1 req/s`)
// ============================================================================

pub struct RateLimitedCurl {
    min_interval: Duration,
    last_request: Option<Instant>,
}

impl RateLimitedCurl {
    pub fn new(min_interval: Duration) -> Self {
        Self {
            min_interval,
            last_request: None,
        }
    }

    fn throttle(&mut self) {
        if let Some(prev) = self.last_request {
            let elapsed = prev.elapsed();
            if elapsed < self.min_interval {
                thread::sleep(self.min_interval - elapsed);
            }
        }
        self.last_request = Some(Instant::now());
    }

    /// Fetch text/bytes from `url` using `curl -fsSL -A nmrassign-fetch`.
    pub fn get_bytes(&mut self, url: &str) -> Result<Vec<u8>, String> {
        self.throttle();
        let out = Command::new("curl")
            .args(["-fsSL", "-A", BMRB_USER_AGENT, url])
            .output()
            .map_err(|e| format!("failed to spawn curl for {url}: {e}"))?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            return Err(format!(
                "curl GET failed for {url} (status {}): {}",
                out.status,
                stderr.trim()
            ));
        }
        Ok(out.stdout)
    }

    /// Query HTTP `HEAD` (`curl -fsSLI -A nmrassign-fetch`) and return optional `Content-Length`.
    pub fn head_content_length(&mut self, url: &str) -> Result<Option<u64>, String> {
        if let Some(local_path) = url.strip_prefix("file://") {
            let p = Path::new(local_path);
            let meta =
                fs::metadata(p).map_err(|e| format!("HEAD local file {url} not found: {e}"))?;
            return Ok(Some(meta.len()));
        }
        self.throttle();
        let out = Command::new("curl")
            .args(["-fsSLI", "-A", BMRB_USER_AGENT, url])
            .output()
            .map_err(|e| format!("failed to spawn curl HEAD for {url}: {e}"))?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            return Err(format!(
                "curl HEAD failed for {url} (status {}): {}",
                out.status,
                stderr.trim()
            ));
        }
        let headers = String::from_utf8_lossy(&out.stdout);
        Ok(parse_last_content_length(&headers))
    }

    /// Stream `url` directly to `dest_file` via `curl -fsSL -A nmrassign-fetch --max-filesize <limit> -o <dest>`.
    pub fn download_to_file(
        &mut self,
        url: &str,
        dest_file: &Path,
        max_remaining_bytes: Option<u64>,
    ) -> Result<(), String> {
        if let Some(parent) = dest_file.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("failed creating {}: {e}", parent.display()))?;
        }
        self.throttle();
        let mut cmd = Command::new("curl");
        cmd.args(["-fsSL", "-A", BMRB_USER_AGENT]);
        if let Some(limit) = max_remaining_bytes {
            cmd.args(["--max-filesize", &limit.to_string()]);
        }
        cmd.arg("-o").arg(dest_file).arg(url);
        let out = cmd
            .output()
            .map_err(|e| format!("failed to spawn curl download for {url}: {e}"))?;
        if !out.status.success() {
            let _ = fs::remove_file(dest_file);
            let stderr = String::from_utf8_lossy(&out.stderr);
            return Err(format!(
                "curl download failed for {url} (status {}): {}",
                out.status,
                stderr.trim()
            ));
        }
        if let Some(limit) = max_remaining_bytes {
            let actual = fs::metadata(dest_file).map(|m| m.len()).unwrap_or(u64::MAX);
            if actual > limit {
                let _ = fs::remove_file(dest_file);
                return Err(format!(
                    "downloaded file from {url} ({actual} bytes) exceeds remaining --max-bytes budget ({limit} bytes)"
                ));
            }
        }
        Ok(())
    }
}

pub fn parse_last_content_length(raw_headers: &str) -> Option<u64> {
    let mut found = None;
    for line in raw_headers.lines() {
        if let Some((name, val)) = line.split_once(':') {
            if name.trim().eq_ignore_ascii_case("content-length") {
                if let Ok(n) = val.trim().parse::<u64>() {
                    found = Some(n);
                }
            }
        }
    }
    found
}

// ============================================================================
// HTML Directory Listing & URL Resolution Helpers
// ============================================================================

const ARCHIVE_SUFFIXES: &[&str] = &[
    ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".tar", ".zip",
];

pub fn is_supported_archive_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    ARCHIVE_SUFFIXES.iter().any(|ext| lower.ends_with(ext))
}

/// Returns `true` when an archive filename in a multi-archive directory clearly belongs to a
/// sidechain / NOESY / TOCSY / COSY / `13C`-HSQC / relaxation experiment rather than `15N`-HSQC or
/// 3D triple-resonance backbone experiments (e.g. BMRB `51351` multi-`.tgz` directory).
pub fn is_non_backbone_archive_filename(name: &str) -> bool {
    let (stem, _) = archive_stem_key(name);
    let alnum: String = stem
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase();

    const BACKBONE_TOKENS: &[&str] = &[
        "nhsqc", "fhsqc", "15nhsqc", "n15hsqc", "hnco", "hnca", "hncacb", "cbcanh", "cbcaconh",
        "cbcacoca", "hncoca", "hncaco", "trosy",
    ];
    if BACKBONE_TOKENS.iter().any(|tok| alnum.contains(tok)) {
        return false;
    }

    const NON_BACKBONE_TOKENS: &[&str] = &[
        "noesy", "roesy", "tocsy", "dipsi", "hcch", "cosy", "hbha", "hccconh", "ccconh", "hmqcj",
        "hmbc", "chsqc", "13chsqc", "c13hsqc", "arohsqc", "hetnoe", "t1rho", "cpmg", "cest",
    ];
    NON_BACKBONE_TOKENS.iter().any(|tok| alnum.contains(tok))
}

fn archive_stem_key(url_or_name: &str) -> (String, usize) {
    let fname = url_or_name
        .rsplit('/')
        .next()
        .unwrap_or(url_or_name)
        .to_ascii_lowercase();
    // Compression preference rank: smaller rank = preferred compressed archive over raw `.tar`.
    let (stem, rank) = if let Some(s) = fname.strip_suffix(".tar.gz") {
        (s.to_string(), 0)
    } else if let Some(s) = fname.strip_suffix(".tgz") {
        (s.to_string(), 0)
    } else if let Some(s) = fname.strip_suffix(".tar.bz2") {
        (s.to_string(), 1)
    } else if let Some(s) = fname.strip_suffix(".tbz2") {
        (s.to_string(), 1)
    } else if let Some(s) = fname.strip_suffix(".tar.xz") {
        (s.to_string(), 2)
    } else if let Some(s) = fname.strip_suffix(".txz") {
        (s.to_string(), 2)
    } else if let Some(s) = fname.strip_suffix(".zip") {
        (s.to_string(), 3)
    } else if let Some(s) = fname.strip_suffix(".tar") {
        (s.to_string(), 4)
    } else {
        (fname, 5)
    };
    (stem, rank)
}

/// Extract archive URLs (`.tar`, `.tar.gz`, `.tgz`, `.tar.bz2`, `.zip`, etc.) from an HTML or
/// plain-text directory listing at `dir_url`.
pub fn parse_archive_urls_from_listing(listing_html: &str, dir_url: &str) -> Vec<String> {
    let base = if dir_url.ends_with('/') {
        dir_url.to_string()
    } else {
        format!("{dir_url}/")
    };
    let origin = extract_url_origin(&base);

    let mut raw_hrefs = extract_html_hrefs(listing_html);
    // Support plain newline-separated directory listings (e.g. `file://` or FTP NLST) as well.
    for line in listing_html.lines() {
        let trimmed = line.trim();
        if !trimmed.contains('<') && !trimmed.contains(' ') && is_supported_archive_name(trimmed) {
            raw_hrefs.push(trimmed.to_string());
        }
    }

    let mut resolved_set = BTreeSet::new();
    for href in raw_hrefs {
        let clean = href.split('#').next().unwrap_or("").trim();
        let path_part = clean.split('?').next().unwrap_or("").trim();
        if path_part.is_empty()
            || path_part == "."
            || path_part == ".."
            || path_part.starts_with("../")
            || path_part.ends_with('/')
        {
            continue;
        }
        let leaf = path_part.rsplit('/').next().unwrap_or(path_part);
        if !is_supported_archive_name(leaf) {
            continue;
        }
        let full_url = if clean.starts_with("http://")
            || clean.starts_with("https://")
            || clean.starts_with("file://")
        {
            clean.to_string()
        } else if clean.starts_with('/') {
            match &origin {
                Some(orig) => format!("{orig}{clean}"),
                None => continue,
            }
        } else {
            let rel = clean.trim_start_matches("./");
            if rel.split('/').any(|seg| seg == "..") {
                continue;
            }
            format!("{base}{rel}")
        };
        resolved_set.insert(full_url);
    }

    // Deduplicate exact stem duplicates where both `<stem>.tar.gz` and uncompressed `<stem>.tar` exist.
    let all: Vec<String> = resolved_set.into_iter().collect();
    let mut best_by_stem: std::collections::BTreeMap<String, (usize, String)> =
        std::collections::BTreeMap::new();
    for url in all {
        let (stem, rank) = archive_stem_key(&url);
        match best_by_stem.get(&stem) {
            Some((prev_rank, _)) if *prev_rank <= rank => {}
            _ => {
                best_by_stem.insert(stem, (rank, url));
            }
        }
    }
    let mut out: Vec<String> = best_by_stem.into_values().map(|(_, u)| u).collect();
    out.sort();
    out
}

/// Extract relative subdirectory names (`foo/` without `..` or query parameters) from an HTML listing.
fn parse_subdir_urls_from_listing(listing_html: &str, dir_url: &str) -> Vec<(String, String)> {
    let base = if dir_url.ends_with('/') {
        dir_url.to_string()
    } else {
        format!("{dir_url}/")
    };
    let mut out = BTreeSet::new();
    for href in extract_html_hrefs(listing_html) {
        let clean = href.split(['?', '#']).next().unwrap_or("").trim();
        if clean.is_empty()
            || clean == "/"
            || clean == "./"
            || clean == "../"
            || clean.starts_with('/')
            || clean.contains("://")
            || !clean.ends_with('/')
        {
            continue;
        }
        let rel = clean.trim_start_matches("./").trim_end_matches('/');
        if rel.is_empty() || rel.contains('/') || rel == "." || rel == ".." {
            continue;
        }
        out.insert((rel.to_string(), format!("{base}{rel}/")));
    }
    out.into_iter().collect()
}

/// Extract safe direct file links (non-HTML, non-directory) from an unpacked experiment subdirectory listing.
fn parse_direct_file_urls_from_listing(listing_html: &str, dir_url: &str) -> Vec<(String, String)> {
    let base = if dir_url.ends_with('/') {
        dir_url.to_string()
    } else {
        format!("{dir_url}/")
    };
    let mut out = BTreeSet::new();
    for href in extract_html_hrefs(listing_html) {
        let clean = href.split(['?', '#']).next().unwrap_or("").trim();
        if clean.is_empty()
            || clean.starts_with('/')
            || clean.contains("://")
            || clean.ends_with('/')
        {
            continue;
        }
        let rel = clean.trim_start_matches("./");
        if rel.is_empty() || rel.contains('/') || rel == "." || rel == ".." {
            continue;
        }
        out.insert((rel.to_string(), format!("{base}{rel}")));
    }
    out.into_iter().collect()
}

fn extract_html_hrefs(html: &str) -> Vec<String> {
    let mut out = Vec::new();
    let lower = html.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let mut pos = 0usize;
    while let Some(rel_idx) = lower[pos..].find("href") {
        let mut i = pos + rel_idx + 4;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b'=' {
            pos = i;
            continue;
        }
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let quote = bytes[i];
        if quote == b'"' || quote == b'\'' {
            let start = i + 1;
            if let Some(end_rel) = html[start..].find(quote as char) {
                out.push(html[start..start + end_rel].to_string());
                pos = start + end_rel + 1;
                continue;
            }
            break;
        } else {
            let start = i;
            let mut end = start;
            while end < bytes.len() && !bytes[end].is_ascii_whitespace() && bytes[end] != b'>' {
                end += 1;
            }
            out.push(html[start..end].to_string());
            pos = end;
        }
    }
    out
}

fn extract_url_origin(url: &str) -> Option<String> {
    let scheme_end = url.find("://")?;
    let after_scheme = &url[scheme_end + 3..];
    if url.starts_with("file://") {
        return Some("file://".to_string());
    }
    let slash = after_scheme.find('/').unwrap_or(after_scheme.len());
    Some(url[..scheme_end + 3 + slash].to_string())
}

// ============================================================================
// Strict Path-Traversal & Symlink Escape Validation Before Archive Extraction
// ============================================================================

/// Validate a single archive member path (`tar -tf` or `unzip -Z1`), rejecting:
/// - Leading `/` or `\`
/// - Windows drive prefixes (`C:...`)
/// - NUL bytes
/// - Any `..` path segment (`..`, `../x`, `a/../b`, `a\..\b`)
pub fn validate_archive_member_path(member: &str) -> Result<(), String> {
    if member.is_empty() {
        return Err("archive contains an empty member path".to_string());
    }
    if member.contains('\0') {
        return Err("archive member path contains NUL byte".to_string());
    }
    if member.starts_with('/') || member.starts_with('\\') {
        return Err(format!(
            "unsafe archive member path {:?}: leading '/' or '\\' rejected",
            member
        ));
    }
    let b = member.as_bytes();
    if b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':' {
        return Err(format!(
            "unsafe archive member path {:?}: drive-letter prefix rejected",
            member
        ));
    }
    for seg in member.split(['/', '\\']) {
        if seg == ".." {
            return Err(format!(
                "unsafe archive member path {:?}: '..' path-traversal component rejected",
                member
            ));
        }
    }
    Ok(())
}

/// Validate a symbolic or hard link inside a `tar` archive (`link_path -> target`), ensuring
/// the target is neither absolute nor escapes the archive extraction root.
pub fn validate_archive_link_target(link_path: &str, target: &str) -> Result<(), String> {
    validate_archive_member_path(link_path)?;
    if target.is_empty() || target.contains('\0') {
        return Err(format!(
            "unsafe archive link {:?} -> {:?}: empty or NUL target",
            link_path, target
        ));
    }
    if target.starts_with('/') || target.starts_with('\\') {
        return Err(format!(
            "unsafe archive link {:?} -> {:?}: absolute link target rejected",
            link_path, target
        ));
    }
    let tb = target.as_bytes();
    if tb.len() >= 2 && tb[0].is_ascii_alphabetic() && tb[1] == b':' {
        return Err(format!(
            "unsafe archive link {:?} -> {:?}: drive-letter link target rejected",
            link_path, target
        ));
    }

    // Compute directory depth of `link_path`'s parent inside `<dest>`, then walk `target` segments.
    let mut depth: isize = 0;
    let link_segs: Vec<&str> = link_path
        .split(['/', '\\'])
        .filter(|s| !s.is_empty() && *s != ".")
        .collect();
    if link_segs.len() > 1 {
        depth = (link_segs.len() - 1) as isize;
    }
    for seg in target.split(['/', '\\']) {
        if seg.is_empty() || seg == "." {
            continue;
        }
        if seg == ".." {
            depth -= 1;
            if depth < 0 {
                return Err(format!(
                    "unsafe archive link {:?} -> {:?}: link target escapes extraction root",
                    link_path, target
                ));
            }
        } else {
            depth += 1;
        }
    }
    Ok(())
}

/// Inspect all entries of a `.tar*` or `.zip` archive on disk and reject any unsafe path or link
/// BEFORE extracting a single byte.
pub fn inspect_and_validate_archive(archive_path: &Path) -> Result<usize, String> {
    let fname = archive_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    if fname.ends_with(".zip") {
        let out = Command::new("unzip")
            .arg("-Z1")
            .arg(archive_path)
            .output()
            .or_else(|_| Command::new("tar").arg("-tf").arg(archive_path).output())
            .map_err(|e| format!("failed to list zip archive {}: {e}", archive_path.display()))?;
        if !out.status.success() {
            return Err(format!(
                "failed inspecting zip archive {}: {}",
                archive_path.display(),
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        let listing = String::from_utf8_lossy(&out.stdout);
        let mut count = 0usize;
        for line in listing.lines() {
            if line.is_empty() {
                continue;
            }
            validate_archive_member_path(line)?;
            count += 1;
        }
        return Ok(count);
    }

    // 1. Plain member path check via `tar -tf`
    let tf_out = Command::new("tar")
        .arg("-tf")
        .arg(archive_path)
        .output()
        .map_err(|e| format!("failed to run tar -tf on {}: {e}", archive_path.display()))?;
    if !tf_out.status.success() {
        return Err(format!(
            "tar -tf failed on {}: {}",
            archive_path.display(),
            String::from_utf8_lossy(&tf_out.stderr).trim()
        ));
    }
    let tf_text = String::from_utf8_lossy(&tf_out.stdout);
    let mut member_paths: Vec<&str> = Vec::new();
    for line in tf_text.lines() {
        if line.is_empty() {
            continue;
        }
        validate_archive_member_path(line)?;
        member_paths.push(line);
    }

    // 2. Verbose mode check via `tar -tvf` to inspect any symbolic (`l`) or hard (`h`) links
    let tvf_out = Command::new("tar")
        .arg("-tvf")
        .arg(archive_path)
        .output()
        .map_err(|e| format!("failed to run tar -tvf on {}: {e}", archive_path.display()))?;
    if !tvf_out.status.success() {
        return Err(format!(
            "tar -tvf failed on {}: {}",
            archive_path.display(),
            String::from_utf8_lossy(&tvf_out.stderr).trim()
        ));
    }
    let tvf_text = String::from_utf8_lossy(&tvf_out.stdout);
    for (idx, line) in tvf_text.lines().enumerate() {
        let first = line.chars().next().unwrap_or('-');
        if first == 'l' || first == 'h' {
            let link_member = member_paths.get(idx).copied().unwrap_or("link");
            if let Some((_, target)) = line.rsplit_once(" -> ") {
                validate_archive_link_target(link_member, target.trim())?;
            } else if let Some((_, target)) = line.rsplit_once(" link to ") {
                validate_archive_link_target(link_member, target.trim())?;
            }
        }
    }

    Ok(member_paths.len())
}

pub fn extract_validated_archive(archive_path: &Path, dest_dir: &Path) -> Result<usize, String> {
    let n_members = inspect_and_validate_archive(archive_path)?;
    fs::create_dir_all(dest_dir)
        .map_err(|e| format!("failed creating {}: {e}", dest_dir.display()))?;

    let fname = archive_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    if fname.ends_with(".zip") {
        let out = Command::new("unzip")
            .args(["-q", "-o"])
            .arg(archive_path)
            .arg("-d")
            .arg(dest_dir)
            .output()
            .or_else(|_| {
                Command::new("tar")
                    .arg("-xf")
                    .arg(archive_path)
                    .arg("-C")
                    .arg(dest_dir)
                    .output()
            })
            .map_err(|e| format!("failed extracting zip {}: {e}", archive_path.display()))?;
        if !out.status.success() {
            return Err(format!(
                "zip extraction failed for {}: {}",
                archive_path.display(),
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
    } else {
        let out = Command::new("tar")
            .arg("-xf")
            .arg(archive_path)
            .arg("-C")
            .arg(dest_dir)
            .output()
            .map_err(|e| {
                format!(
                    "failed spawning tar -xf for {}: {e}",
                    archive_path.display()
                )
            })?;
        if !out.status.success() {
            return Err(format!(
                "tar extraction failed for {}: {}",
                archive_path.display(),
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
    }

    Ok(n_members)
}

// ============================================================================
// BMRB Entry Resolution, Fetching & PROVENANCE.md Generation
// ============================================================================

#[derive(Clone, Debug)]
pub struct DownloadedArtifact {
    pub relative_path: String,
    pub source_url: String,
    pub size_bytes: u64,
    pub sha256_hex: String,
}

#[derive(Clone, Debug)]
pub struct FetchBmrbConfig {
    pub bmrb_id: String,
    pub dest_dir: PathBuf,
    pub dry_run: bool,
    pub include_all_archives: bool,
    pub max_bytes: u64,
    pub ftp_base_url: String,
    pub api_base_url: String,
    pub min_interval: Duration,
}

pub fn normalize_bmrb_id(raw: &str) -> Result<String, String> {
    let trimmed = raw.trim();
    let stripped = if trimmed.len() > 3 && trimmed[..3].eq_ignore_ascii_case("bmr") {
        &trimmed[3..]
    } else if trimmed.len() > 4 && trimmed[..4].eq_ignore_ascii_case("bmrb") {
        &trimmed[4..]
    } else {
        trimmed
    };
    if stripped.is_empty() || !stripped.chars().all(|c| c.is_ascii_digit()) {
        return Err(format!(
            "invalid --id {raw:?}; expected numeric BMRB accession (e.g. 15477 or bmr15477)"
        ));
    }
    Ok(stripped.to_string())
}

fn candidate_star_urls(id: &str, ftp_base: &str, api_base: &str) -> Vec<String> {
    let ftp = ftp_base.trim_end_matches('/');
    let api = api_base.trim_end_matches('/');
    let mut urls = vec![
        format!("{ftp}/entry_directories/bmr{id}/bmr{id}_3.str"),
        format!("{ftp}/biomolecules/bmr{id}/bmr{id}_3.str"),
    ];
    if !api.is_empty() {
        urls.push(format!("{api}/entry/{id}?format=nmrstar"));
        urls.push(format!("{api}/entry/{id}"));
    }
    urls
}

fn candidate_timedomain_dir_urls(id: &str, ftp_base: &str) -> Vec<String> {
    let ftp = ftp_base.trim_end_matches('/');
    vec![
        format!("{ftp}/entry_directories/bmr{id}/timedomain_data/"),
        format!("{ftp}/timedomain/bmr{id}/"),
    ]
}

#[derive(Clone, Debug)]
enum RemoteTimeDomainItem {
    Archive {
        url: String,
        filename: String,
        content_length: Option<u64>,
    },
    DirectFile {
        url: String,
        rel_path: String,
        content_length: Option<u64>,
    },
}

impl RemoteTimeDomainItem {
    fn url(&self) -> &str {
        match self {
            Self::Archive { url, .. } | Self::DirectFile { url, .. } => url,
        }
    }

    fn display_name(&self) -> &str {
        match self {
            Self::Archive { filename, .. } => filename,
            Self::DirectFile { rel_path, .. } => rel_path,
        }
    }

    fn content_length(&self) -> Option<u64> {
        match self {
            Self::Archive { content_length, .. } | Self::DirectFile { content_length, .. } => {
                *content_length
            }
        }
    }
}

fn resolve_star_url_and_size(
    client: &mut RateLimitedCurl,
    id: &str,
    ftp_base: &str,
    api_base: &str,
) -> Result<(String, Option<u64>), String> {
    let candidates = candidate_star_urls(id, ftp_base, api_base);
    let mut errors = Vec::new();
    for url in candidates {
        match client.head_content_length(&url) {
            Ok(sz) => return Ok((url, sz)),
            Err(e) => errors.push(e),
        }
    }
    Err(format!(
        "unable to locate NMR-STAR entry for BMRB {id}: {}",
        errors.join("; ")
    ))
}

fn resolve_timedomain_items(
    client: &mut RateLimitedCurl,
    id: &str,
    ftp_base: &str,
    include_all_archives: bool,
) -> Result<(String, Vec<RemoteTimeDomainItem>), String> {
    let dir_candidates = candidate_timedomain_dir_urls(id, ftp_base);
    let mut listing_html = None;
    let mut chosen_dir_url = String::new();
    let mut errs = Vec::new();

    for durl in dir_candidates {
        if let Some(local_dir) = durl.strip_prefix("file://") {
            let p = Path::new(local_dir);
            if p.is_dir() {
                let mut synth = String::new();
                let mut entries: Vec<_> = fs::read_dir(p)
                    .map_err(|e| format!("failed reading {}: {e}", p.display()))?
                    .filter_map(|r| r.ok())
                    .collect();
                entries.sort_by_key(|e| e.file_name());
                for ent in entries {
                    let name = ent.file_name().to_string_lossy().into_owned();
                    let suffix = if ent.path().is_dir() { "/" } else { "" };
                    synth.push_str(&format!("<a href=\"{name}{suffix}\">{name}{suffix}</a>\n"));
                }
                listing_html = Some(synth);
                chosen_dir_url = durl;
                break;
            } else {
                errs.push(format!("local directory {} not found", p.display()));
                continue;
            }
        }
        match client.get_bytes(&durl) {
            Ok(bytes) => {
                listing_html = Some(String::from_utf8_lossy(&bytes).into_owned());
                chosen_dir_url = durl;
                break;
            }
            Err(e) => errs.push(e),
        }
    }

    let html = listing_html.ok_or_else(|| {
        format!(
            "unable to fetch time-domain directory listing for BMRB {id}: {}",
            errs.join("; ")
        )
    })?;

    let mut archive_urls = parse_archive_urls_from_listing(&html, &chosen_dir_url);
    let mut items = Vec::new();

    if archive_urls.is_empty() {
        // Check 1-level subdirectories for archives or unpacked `.fid` / Bruker experiment files.
        let subdirs = parse_subdir_urls_from_listing(&html, &chosen_dir_url);
        for (sub_name, sub_url) in subdirs {
            let sub_html_opt = if let Some(local_sub) = sub_url.strip_prefix("file://") {
                let p = Path::new(local_sub);
                if p.is_dir() {
                    let mut s = String::new();
                    if let Ok(rd) = fs::read_dir(p) {
                        let mut ents: Vec<_> = rd.filter_map(|r| r.ok()).collect();
                        ents.sort_by_key(|e| e.file_name());
                        for ent in ents {
                            let nm = ent.file_name().to_string_lossy().into_owned();
                            let suf = if ent.path().is_dir() { "/" } else { "" };
                            s.push_str(&format!("<a href=\"{nm}{suf}\">{nm}{suf}</a>\n"));
                        }
                    }
                    Some(s)
                } else {
                    None
                }
            } else {
                client
                    .get_bytes(&sub_url)
                    .ok()
                    .map(|b| String::from_utf8_lossy(&b).into_owned())
            };

            if let Some(sub_html) = sub_html_opt {
                let sub_archives = parse_archive_urls_from_listing(&sub_html, &sub_url);
                if !sub_archives.is_empty() {
                    archive_urls.extend(sub_archives);
                } else {
                    for (fname, furl) in parse_direct_file_urls_from_listing(&sub_html, &sub_url) {
                        let rel = format!("{sub_name}/{fname}");
                        validate_archive_member_path(&rel)?;
                        let cl = client.head_content_length(&furl).ok().flatten();
                        items.push(RemoteTimeDomainItem::DirectFile {
                            url: furl,
                            rel_path: rel,
                            content_length: cl,
                        });
                    }
                }
            }
        }
    }

    archive_urls.sort();
    archive_urls.dedup();
    if archive_urls.len() > 1 && !include_all_archives {
        let mut kept = Vec::with_capacity(archive_urls.len());
        let mut skipped_names = Vec::new();
        for aurl in &archive_urls {
            let leaf = aurl
                .split(['?', '#'])
                .next()
                .unwrap_or(aurl)
                .rsplit('/')
                .next()
                .unwrap_or(aurl);
            if is_non_backbone_archive_filename(leaf) {
                skipped_names.push(leaf.to_string());
            } else {
                kept.push(aurl.clone());
            }
        }
        if !kept.is_empty() {
            if !skipped_names.is_empty() {
                println!(
                    "  Skipping {} non-backbone archive(s) (pass --all-archives to include): {}",
                    skipped_names.len(),
                    skipped_names.join(", ")
                );
            }
            archive_urls = kept;
        }
    }
    for aurl in archive_urls {
        let fname = aurl
            .split(['?', '#'])
            .next()
            .unwrap_or(&aurl)
            .rsplit('/')
            .next()
            .unwrap_or("archive.tar")
            .to_string();
        validate_archive_member_path(&fname)?;
        let cl = client.head_content_length(&aurl).ok().flatten();
        items.push(RemoteTimeDomainItem::Archive {
            url: aurl,
            filename: fname,
            content_length: cl,
        });
    }

    if items.is_empty() {
        return Err(format!(
            "no time-domain archives (.tar, .tar.gz, .tgz, .tar.bz2, .zip) or FID subdirectories found at {chosen_dir_url}"
        ));
    }

    Ok((chosen_dir_url, items))
}

pub fn render_provenance_markdown(
    bmrb_id: &str,
    utc_timestamp: &str,
    td_dir_url: &str,
    artifacts: &[DownloadedArtifact],
) -> String {
    let total_bytes: u64 = artifacts.iter().map(|a| a.size_bytes).sum();
    let mut md = String::new();
    md.push_str(&format!(
        "# Provenance: BMRB Entry {bmrb_id}\n\n\
         - **BMRB Accession**: `{bmrb_id}` (`bmr{bmrb_id}`)\n\
         - **Time-Domain Directory**: `{td_dir_url}`\n\
         - **Downloaded At (UTC)**: `{utc_timestamp}`\n\
         - **Downloader**: `nmrassign fetch-bmrb` (`User-Agent: {BMRB_USER_AGENT}`, `<= 1 req/s`)\n\
         - **Total Downloaded Size**: `{total_bytes}` bytes ({:.2} MiB)\n\
         - **License / Data Usage Terms**: Biological Magnetic Resonance Data Bank (BMRB) entries \
           and time-domain archives are distributed in the **public domain / CC0 1.0 Universal** \
           (`https://bmrb.io/about/usage/`).\n\n\
         ## Downloaded Files & SHA-256 Digests\n\n\
         | File | Bytes | SHA-256 | Source URL |\n\
         |---|---:|---|---|\n",
        (total_bytes as f64) / (1024.0 * 1024.0),
    ));
    for art in artifacts {
        md.push_str(&format!(
            "| `{}` | {} | `{}` | `{}` |\n",
            art.relative_path, art.size_bytes, art.sha256_hex, art.source_url
        ));
    }
    md
}

pub fn execute_fetch_bmrb(cfg: &FetchBmrbConfig) -> Result<Vec<DownloadedArtifact>, String> {
    let id = normalize_bmrb_id(&cfg.bmrb_id)?;
    let mut client = RateLimitedCurl::new(cfg.min_interval);

    let (star_url, star_len_opt) =
        resolve_star_url_and_size(&mut client, &id, &cfg.ftp_base_url, &cfg.api_base_url)?;
    let (td_dir_url, td_items) = resolve_timedomain_items(
        &mut client,
        &id,
        &cfg.ftp_base_url,
        cfg.include_all_archives,
    )?;

    let mut known_total = star_len_opt.unwrap_or(0);
    let mut unknown_count = usize::from(star_len_opt.is_none());
    for item in &td_items {
        match item.content_length() {
            Some(sz) => known_total = known_total.saturating_add(sz),
            None => unknown_count += 1,
        }
    }

    println!("=== BMRB {id} Time-Domain & NMR-STAR Manifest ===");
    println!(
        "  NMR-STAR : {star_url} ({})",
        format_opt_bytes(star_len_opt)
    );
    for item in &td_items {
        println!(
            "  TD item  : {} ({})",
            item.url(),
            format_opt_bytes(item.content_length())
        );
    }
    println!(
        "  Known total size: {} bytes ({:.2} MiB){}",
        known_total,
        (known_total as f64) / (1024.0 * 1024.0),
        if unknown_count > 0 {
            format!(" [{unknown_count} item(s) without Content-Length]")
        } else {
            String::new()
        }
    );

    if known_total > cfg.max_bytes {
        return Err(format!(
            "BMRB {id} total Content-Length ({known_total} bytes, {:.2} MiB) exceeds --max-bytes limit ({} bytes, {:.2} MiB)",
            (known_total as f64) / (1024.0 * 1024.0),
            cfg.max_bytes,
            (cfg.max_bytes as f64) / (1024.0 * 1024.0),
        ));
    }

    if cfg.dry_run {
        println!(
            "[dry-run] Verified URLs and Content-Length budgets for BMRB {id}; zero files written."
        );
        return Ok(Vec::new());
    }

    fs::create_dir_all(&cfg.dest_dir)
        .map_err(|e| format!("failed creating {}: {e}", cfg.dest_dir.display()))?;

    let mut downloaded: Vec<DownloadedArtifact> = Vec::new();
    let mut cumulative_bytes = 0u64;

    // 1. Download NMR-STAR file
    let star_rel = format!("bmr{id}_3.str");
    let star_dest = cfg.dest_dir.join(&star_rel);
    let rem = cfg.max_bytes.saturating_sub(cumulative_bytes);
    client.download_to_file(&star_url, &star_dest, Some(rem))?;
    let (star_sha, star_sz) = sha256_file_and_size(&star_dest)?;
    cumulative_bytes = cumulative_bytes.saturating_add(star_sz);
    if cumulative_bytes > cfg.max_bytes {
        let _ = fs::remove_file(&star_dest);
        return Err(format!(
            "cumulative download ({cumulative_bytes} bytes) exceeded --max-bytes ({})",
            cfg.max_bytes
        ));
    }
    downloaded.push(DownloadedArtifact {
        relative_path: star_rel,
        source_url: star_url,
        size_bytes: star_sz,
        sha256_hex: star_sha,
    });

    // 2. Download & safely extract time-domain archives (or direct unpacked files)
    let n_archives = td_items
        .iter()
        .filter(|it| matches!(it, RemoteTimeDomainItem::Archive { .. }))
        .count();
    for item in &td_items {
        let rem = cfg.max_bytes.checked_sub(cumulative_bytes).ok_or_else(|| {
            format!(
                "cumulative download ({cumulative_bytes} bytes) reached --max-bytes ({})",
                cfg.max_bytes
            )
        })?;
        match item {
            RemoteTimeDomainItem::Archive { url, filename, .. } => {
                validate_archive_member_path(filename)?;
                let archive_dest = cfg.dest_dir.join(filename);
                client.download_to_file(url, &archive_dest, Some(rem))?;
                let (sha, sz) = sha256_file_and_size(&archive_dest)?;
                cumulative_bytes = cumulative_bytes.saturating_add(sz);
                if cumulative_bytes > cfg.max_bytes {
                    let _ = fs::remove_file(&archive_dest);
                    return Err(format!(
                        "cumulative download ({cumulative_bytes} bytes) exceeded --max-bytes ({})",
                        cfg.max_bytes
                    ));
                }
                // Isolate each archive under `<dest>/<stem>/` when multiple per-experiment archives
                // exist (e.g. `53371` multi-`.zip` or `51351` multi-`.tgz`) so root-level `1/acqus`
                // or `acqus`+`ser` files across archives can never overwrite one another.
                let extract_root = if n_archives > 1 {
                    let (stem, _) = archive_stem_key(filename);
                    let clean_stem = if stem.is_empty() {
                        "archive"
                    } else {
                        stem.as_str()
                    };
                    validate_archive_member_path(clean_stem)?;
                    cfg.dest_dir.join(clean_stem)
                } else {
                    cfg.dest_dir.clone()
                };
                let n_members = extract_validated_archive(&archive_dest, &extract_root)?;
                println!(
                    "  Extracted {} ({} bytes, {} validated entries, sha256={})",
                    item.display_name(),
                    sz,
                    n_members,
                    &sha[..16]
                );
                downloaded.push(DownloadedArtifact {
                    relative_path: filename.clone(),
                    source_url: url.clone(),
                    size_bytes: sz,
                    sha256_hex: sha,
                });
            }
            RemoteTimeDomainItem::DirectFile { url, rel_path, .. } => {
                validate_archive_member_path(rel_path)?;
                let fdest = cfg.dest_dir.join(rel_path);
                client.download_to_file(url, &fdest, Some(rem))?;
                let (sha, sz) = sha256_file_and_size(&fdest)?;
                cumulative_bytes = cumulative_bytes.saturating_add(sz);
                if cumulative_bytes > cfg.max_bytes {
                    let _ = fs::remove_file(&fdest);
                    return Err(format!(
                        "cumulative download ({cumulative_bytes} bytes) exceeded --max-bytes ({})",
                        cfg.max_bytes
                    ));
                }
                downloaded.push(DownloadedArtifact {
                    relative_path: rel_path.clone(),
                    source_url: url.clone(),
                    size_bytes: sz,
                    sha256_hex: sha,
                });
            }
        }
    }

    // 3. Write `<dest>/PROVENANCE.md`
    let ts = now_utc_iso8601();
    let prov_md = render_provenance_markdown(&id, &ts, &td_dir_url, &downloaded);
    let prov_path = cfg.dest_dir.join("PROVENANCE.md");
    fs::write(&prov_path, prov_md)
        .map_err(|e| format!("failed writing {}: {e}", prov_path.display()))?;
    println!(
        "Wrote {} ({} downloaded files, {:.2} MiB total)",
        prov_path.display(),
        downloaded.len(),
        (cumulative_bytes as f64) / (1024.0 * 1024.0)
    );

    Ok(downloaded)
}

fn format_opt_bytes(opt: Option<u64>) -> String {
    match opt {
        Some(b) => format!("{b} bytes ({:.2} MiB)", (b as f64) / (1024.0 * 1024.0)),
        None => "Content-Length unknown".to_string(),
    }
}

pub fn parse_byte_limit(raw: &str) -> Result<u64, String> {
    let trimmed = raw.trim();
    let upper = trimmed.to_ascii_uppercase();
    let (num_str, mult): (&str, u64) = if let Some(s) = upper.strip_suffix("GIB") {
        (s, 1024 * 1024 * 1024)
    } else if let Some(s) = upper.strip_suffix("GB") {
        (s, 1_000_000_000)
    } else if let Some(s) = upper.strip_suffix('G') {
        (s, 1024 * 1024 * 1024)
    } else if let Some(s) = upper.strip_suffix("MIB") {
        (s, 1024 * 1024)
    } else if let Some(s) = upper.strip_suffix("MB") {
        (s, 1_000_000)
    } else if let Some(s) = upper.strip_suffix('M') {
        (s, 1024 * 1024)
    } else if let Some(s) = upper.strip_suffix("KIB") {
        (s, 1024)
    } else if let Some(s) = upper.strip_suffix("KB") {
        (s, 1_000)
    } else if let Some(s) = upper.strip_suffix('K') {
        (s, 1024)
    } else {
        (trimmed, 1)
    };
    let val: u64 = num_str
        .trim()
        .parse()
        .map_err(|e| format!("invalid --max-bytes value {raw:?}: {e}"))?;
    val.checked_mul(mult)
        .ok_or_else(|| format!("--max-bytes value {raw:?} overflowed u64"))
}

pub fn cmd_fetch_bmrb(args: &[String]) -> Result<(), String> {
    let mut bmrb_id: Option<String> = None;
    let mut dest_dir: Option<PathBuf> = None;
    let mut dry_run = false;
    let mut include_all_archives = false;
    let mut max_bytes = DEFAULT_MAX_DOWNLOAD_BYTES;
    let mut ftp_base_url = DEFAULT_BMRB_FTP_BASE.to_string();
    let mut api_base_url = DEFAULT_BMRB_API_BASE.to_string();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--id" => {
                i += 1;
                bmrb_id = args.get(i).cloned();
            }
            "--dest" => {
                i += 1;
                dest_dir = args.get(i).map(PathBuf::from);
            }
            "--dry-run" => {
                dry_run = true;
            }
            "--all-archives" => {
                include_all_archives = true;
            }
            "--max-bytes" => {
                i += 1;
                let raw = args.get(i).ok_or("missing value for --max-bytes")?;
                max_bytes = parse_byte_limit(raw)?;
            }
            "--ftp-base-url" => {
                i += 1;
                ftp_base_url = args
                    .get(i)
                    .ok_or("missing value for --ftp-base-url")?
                    .clone();
            }
            "--api-base-url" => {
                i += 1;
                api_base_url = args
                    .get(i)
                    .ok_or("missing value for --api-base-url")?
                    .clone();
            }
            other => return Err(format!("unknown fetch-bmrb flag {other:?}")),
        }
        i += 1;
    }

    let id = normalize_bmrb_id(&bmrb_id.ok_or("fetch-bmrb requires --id <bmrb_id>")?)?;
    let dest = dest_dir.unwrap_or_else(|| PathBuf::from(format!("data/private/bmrb/{id}")));

    let cfg = FetchBmrbConfig {
        bmrb_id: id,
        dest_dir: dest,
        dry_run,
        include_all_archives,
        max_bytes,
        ftp_base_url,
        api_base_url,
        min_interval: Duration::from_millis(MIN_HTTP_INTERVAL_MS),
    };
    execute_fetch_bmrb(&cfg).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::process;

    #[test]
    fn test_sha256_nist_fips_180_4_vectors() {
        // NIST FIPS 180-4 / CAVP test vector 1: empty string ""
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        // NIST FIPS 180-4 test vector 2: "abc"
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // NIST FIPS 180-4 multi-block test vector 3: 448-bit message
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        // Streaming chunked update parity across 64-byte block boundaries
        let mut streaming = Sha256::new();
        for ch in b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq".chunks(7) {
            streaming.update(ch);
        }
        assert_eq!(
            bytes_to_hex(&streaming.finalize()),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn test_archive_path_traversal_and_symlink_validation() {
        assert!(validate_archive_member_path("bmr15477/hsqc.fid/procpar").is_ok());
        assert!(validate_archive_member_path("./Ubiquitin_090309/2/ser").is_ok());
        assert!(validate_archive_member_path("file_with..dots_in_name.tar.gz").is_ok());

        assert!(validate_archive_member_path("/etc/passwd").is_err());
        assert!(validate_archive_member_path("\\Windows\\System32").is_err());
        assert!(validate_archive_member_path("C:\\evil.txt").is_err());
        assert!(validate_archive_member_path("../outside.txt").is_err());
        assert!(validate_archive_member_path("a/b/../../../outside.txt").is_err());
        assert!(validate_archive_member_path("a\\..\\outside.txt").is_err());
        assert!(validate_archive_member_path("bad\0name").is_err());

        // Safe internal relative symlink vs escaping relative/absolute symlinks
        assert!(validate_archive_link_target("exp/pdata/latest", "1").is_ok());
        assert!(validate_archive_link_target("exp/pdata/1/ref", "../../acqus").is_ok());
        assert!(
            validate_archive_link_target("exp/pdata/1/escape", "../../../../etc/passwd").is_err()
        );
        assert!(validate_archive_link_target("exp/abs", "/etc/passwd").is_err());
    }

    fn build_ustar_archive(entries: &[(&str, u8, &str, &[u8])]) -> Vec<u8> {
        let mut out = Vec::new();
        for &(name, typeflag, linkname, content) in entries {
            let mut hdr = [0u8; 512];
            let nb = name.as_bytes();
            hdr[..nb.len().min(100)].copy_from_slice(&nb[..nb.len().min(100)]);
            hdr[100..108].copy_from_slice(b"0000644\0");
            hdr[108..116].copy_from_slice(b"0001000\0");
            hdr[116..124].copy_from_slice(b"0001000\0");
            let sz_oct = format!(
                "{:011o}\0",
                if typeflag == b'0' { content.len() } else { 0 }
            );
            hdr[124..136].copy_from_slice(sz_oct.as_bytes());
            hdr[136..148].copy_from_slice(b"14712345670\0");
            hdr[148..156].fill(b' ');
            hdr[156] = typeflag;
            let lb = linkname.as_bytes();
            hdr[157..157 + lb.len().min(100)].copy_from_slice(&lb[..lb.len().min(100)]);
            hdr[257..263].copy_from_slice(b"ustar\0");
            hdr[263..265].copy_from_slice(b"00");
            let csum: u32 = hdr.iter().map(|&b| b as u32).sum();
            let csum_str = format!("{:06o}\0 ", csum);
            hdr[148..156].copy_from_slice(csum_str.as_bytes());
            out.extend_from_slice(&hdr);
            if typeflag == b'0' && !content.is_empty() {
                out.extend_from_slice(content);
                let rem = content.len() % 512;
                if rem != 0 {
                    out.extend(std::iter::repeat_n(0u8, 512 - rem));
                }
            }
        }
        out.extend(std::iter::repeat_n(0u8, 1024));
        out
    }

    #[test]
    fn test_malicious_tar_rejected_before_extraction() {
        let dir = env::temp_dir().join(format!("nmr_fetch_tartest_{}", process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let traversal_tar = dir.join("evil_traversal.tar");
        let bytes = build_ustar_archive(&[("../pwned.txt", b'0', "", b"bad")]);
        fs::write(&traversal_tar, bytes).unwrap();

        let extract_dir = dir.join("out");
        let err = extract_validated_archive(&traversal_tar, &extract_dir).unwrap_err();
        assert!(err.contains(".."), "expected '..' rejection, got: {err}");
        assert!(!dir.join("pwned.txt").exists());

        let symlink_tar = dir.join("evil_symlink.tar");
        let sym_bytes = build_ustar_archive(&[("sub/escape", b'2', "../../outside", b"")]);
        fs::write(&symlink_tar, sym_bytes).unwrap();
        let sym_err = extract_validated_archive(&symlink_tar, &extract_dir).unwrap_err();
        assert!(
            sym_err.contains("escapes extraction root") || sym_err.contains("unsafe"),
            "unexpected symlink err: {sym_err}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_parse_archive_urls_and_rate_limit_and_offline_fetch() {
        let html = r#"
            <html><body>
            <a href="?C=N;O=D">Name</a>
            <a href="/ftp/pub/bmrb/entry_directories/bmr15088/">Parent Directory</a>
            <a href="nesg_bmrb15088.tar">nesg_bmrb15088.tar</a>
            <a href="nesg_bmrb15088.tar.gz">nesg_bmrb15088.tar.gz</a>
            <a href="extra_peaks.zip">extra_peaks.zip</a>
            <a href="unpacked.fid/">unpacked.fid/</a>
            </body></html>
        "#;
        let urls = parse_archive_urls_from_listing(
            html,
            "https://bmrb.io/ftp/pub/bmrb/entry_directories/bmr15088/timedomain_data/",
        );
        // Compressed `.tar.gz` deduplicates redundant `.tar` of identical stem, while keeping `.zip`.
        assert_eq!(
            urls,
            vec![
                "https://bmrb.io/ftp/pub/bmrb/entry_directories/bmr15088/timedomain_data/extra_peaks.zip",
                "https://bmrb.io/ftp/pub/bmrb/entry_directories/bmr15088/timedomain_data/nesg_bmrb15088.tar.gz",
            ]
        );

        // Offline `file://` end-to-end test: rate limiting (`>= 1000 ms`), `--dry-run`, `--max-bytes`,
        // archive extraction, and `PROVENANCE.md` SHA-256 verification.
        let root = env::temp_dir().join(format!("nmr_fetch_e2e_{}", process::id()));
        let _ = fs::remove_dir_all(&root);
        let mock_entry = root.join("ftp/entry_directories/bmr99999");
        let mock_td = mock_entry.join("timedomain_data");
        fs::create_dir_all(&mock_td).unwrap();

        let star_bytes = b"data_99999\nsave_assigned_chemical_shifts\nloop_\n_Atom_chem_shift.ID\nstop_\nsave_\n";
        fs::write(mock_entry.join("bmr99999_3.str"), star_bytes).unwrap();

        let td_tar_bytes = build_ustar_archive(&[(
            "bmr99999_td/1/acqus",
            b'0',
            "",
            b"##TITLE= test\n##$TD= 1024\n",
        )]);
        fs::write(mock_td.join("bmr99999_td.tar"), &td_tar_bytes).unwrap();

        // Check `RateLimitedCurl` sleeps >= 1000 ms across back-to-back calls
        let mut rl = RateLimitedCurl::new(Duration::from_millis(MIN_HTTP_INTERVAL_MS));
        let star_file_url = format!("file://{}", mock_entry.join("bmr99999_3.str").display());
        let t0 = Instant::now();
        let b1 = rl.get_bytes(&star_file_url).unwrap();
        let b2 = rl.get_bytes(&star_file_url).unwrap();
        let dt = t0.elapsed();
        assert_eq!(b1, star_bytes);
        assert_eq!(b2, star_bytes);
        assert!(
            dt >= Duration::from_millis(1000),
            "expected >= 1000ms between HTTP requests, got {dt:?}"
        );

        let ftp_base = format!("file://{}", root.join("ftp").display());
        let dest = root.join("dest");

        // 1. `--max-bytes` safety rejection
        let small_cfg = FetchBmrbConfig {
            bmrb_id: "99999".to_string(),
            dest_dir: dest.clone(),
            dry_run: true,
            include_all_archives: false,
            max_bytes: 32,
            ftp_base_url: ftp_base.clone(),
            api_base_url: String::new(),
            min_interval: Duration::from_millis(1),
        };
        assert!(execute_fetch_bmrb(&small_cfg)
            .unwrap_err()
            .contains("--max-bytes"));
        assert!(!dest.exists());

        // 2. `--dry-run` writes nothing
        let dry_cfg = FetchBmrbConfig {
            max_bytes: 1_000_000,
            ..small_cfg.clone()
        };
        let dry_arts = execute_fetch_bmrb(&dry_cfg).unwrap();
        assert!(dry_arts.is_empty());
        assert!(!dest.exists());

        // 3. Full download + extraction + PROVENANCE.md verification
        let full_cfg = FetchBmrbConfig {
            dry_run: false,
            ..dry_cfg
        };
        let arts = execute_fetch_bmrb(&full_cfg).unwrap();
        assert_eq!(arts.len(), 2);
        assert!(dest.join("bmr99999_3.str").is_file());
        assert!(dest.join("bmr99999_td/1/acqus").is_file());
        let prov = fs::read_to_string(dest.join("PROVENANCE.md")).unwrap();
        assert!(prov.contains("CC0 1.0 Universal"));
        assert!(prov.contains(&sha256_hex(star_bytes)));
        assert!(prov.contains(&sha256_hex(&td_tar_bytes)));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_multi_zip_53371_multi_tgz_51351_and_archive_only_6318() {
        // 1. BMRB 51351-style multi-`.tgz` filtering: keeps the 4 backbone `.tgz` archives and
        // skips sidechain/NOESY/TOCSY/`13C`-HSQC `.tgz` archives unless `--all-archives` is set.
        assert!(!is_non_backbone_archive_filename(
            "tsr3nc_nhsqc_800_174.tgz"
        ));
        assert!(!is_non_backbone_archive_filename(
            "tsr3nc_hnco_nus_600_128.tgz"
        ));
        assert!(!is_non_backbone_archive_filename(
            "tsr3nc_hncacb_600_54_55.tgz"
        ));
        assert!(!is_non_backbone_archive_filename(
            "tsr3nc_cbcaconh_600_129_131.tgz"
        ));
        assert!(!is_non_backbone_archive_filename("fHSQC.zip"));
        assert!(!is_non_backbone_archive_filename("HNcaCO.zip"));
        assert!(!is_non_backbone_archive_filename("C_cs_data.zip"));

        assert!(is_non_backbone_archive_filename("tsr3nc_noesy_800.tgz"));
        assert!(is_non_backbone_archive_filename(
            "tsr3nc_hcch_tocsy_600.tgz"
        ));
        assert!(is_non_backbone_archive_filename("tsr3nc_hccconh_600.tgz"));
        assert!(is_non_backbone_archive_filename("tsr3nc_ccconh_600.tgz"));
        assert!(is_non_backbone_archive_filename("tsr3nc_chsqc_600.tgz"));

        // 2. BMRB 6318-style listing: when `At3g51030_timedomain.tar.gz` is present alongside a
        // sibling unpacked `analysis/` directory, ONLY the archive is selected!
        let root = env::temp_dir().join(format!("nmr_fetch_multi_{}", process::id()));
        let _ = fs::remove_dir_all(&root);

        let e6318 = root.join("ftp/entry_directories/bmr6318");
        let td6318 = e6318.join("timedomain_data");
        fs::create_dir_all(td6318.join("analysis")).unwrap();
        fs::write(e6318.join("bmr6318_3.str"), b"data_6318\n").unwrap();
        fs::write(td6318.join("analysis/forbidden.txt"), b"do_not_fetch").unwrap();
        let tar6318 = build_ustar_archive(&[("hsqc/acqus", b'0', "", b"##$PULPROG= <hsqc>\n")]);
        fs::write(td6318.join("At3g51030_timedomain.tar"), &tar6318).unwrap();

        let ftp_base = format!("file://{}", root.join("ftp").display());
        let dest6318 = root.join("out_6318");
        let arts6318 = execute_fetch_bmrb(&FetchBmrbConfig {
            bmrb_id: "6318".to_string(),
            dest_dir: dest6318.clone(),
            dry_run: false,
            include_all_archives: false,
            max_bytes: 10_000_000,
            ftp_base_url: ftp_base.clone(),
            api_base_url: String::new(),
            min_interval: Duration::from_millis(1),
        })
        .unwrap();
        assert_eq!(arts6318.len(), 2); // bmr6318_3.str + At3g51030_timedomain.tar ONLY
        assert!(!dest6318.join("analysis/forbidden.txt").exists());
        assert!(dest6318.join("hsqc/acqus").is_file());

        // 3. BMRB 53371 / 51351 multi-archive isolation + sidechain filtering:
        // Two backbone archives BOTH having root `1/acqus` extract into `<dest>/<stem>/1/acqus`
        // without clobbering each other, while `noesy.tar` is skipped!
        let e53371 = root.join("ftp/entry_directories/bmr53371");
        let td53371 = e53371.join("timedomain_data");
        fs::create_dir_all(&td53371).unwrap();
        fs::write(e53371.join("bmr53371_3.str"), b"data_53371\n").unwrap();
        fs::write(
            td53371.join("fHSQC.tar"),
            build_ustar_archive(&[("1/acqus", b'0', "", b"##$PULPROG= <fhsqc>\n")]),
        )
        .unwrap();
        fs::write(
            td53371.join("HNCO.tar"),
            build_ustar_archive(&[("1/acqus", b'0', "", b"##$PULPROG= <hncogp3d>\n")]),
        )
        .unwrap();
        fs::write(
            td53371.join("noesy_sidechain.tar"),
            build_ustar_archive(&[("1/acqus", b'0', "", b"##$PULPROG= <noesy>\n")]),
        )
        .unwrap();

        let dest53371 = root.join("out_53371");
        let arts53371 = execute_fetch_bmrb(&FetchBmrbConfig {
            bmrb_id: "53371".to_string(),
            dest_dir: dest53371.clone(),
            dry_run: false,
            include_all_archives: false,
            max_bytes: 10_000_000,
            ftp_base_url: ftp_base,
            api_base_url: String::new(),
            min_interval: Duration::from_millis(1),
        })
        .unwrap();
        assert_eq!(arts53371.len(), 3); // STAR + fHSQC.tar + HNCO.tar (noesy_sidechain.tar skipped!)
        assert_eq!(
            fs::read_to_string(dest53371.join("fhsqc/1/acqus")).unwrap(),
            "##$PULPROG= <fhsqc>\n"
        );
        assert_eq!(
            fs::read_to_string(dest53371.join("hnco/1/acqus")).unwrap(),
            "##$PULPROG= <hncogp3d>\n"
        );
        assert!(!dest53371.join("noesy_sidechain.tar").exists());

        let _ = fs::remove_dir_all(&root);
    }
}
