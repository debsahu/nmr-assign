//! Embeds the short git commit SHA (`NMRASSIGN_GIT_SHA`) into `nmrassign` at compile time
//! without external build dependencies.

use std::path::Path;
use std::process::Command;

fn main() {
    for rel in [
        "../../.git/HEAD",
        "../../.git/refs/heads",
        "../../.git/packed-refs",
    ] {
        if Path::new(rel).exists() {
            println!("cargo:rerun-if-changed={rel}");
        }
    }

    let git_sha = Command::new("git")
        .args(["rev-parse", "--short=7", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=NMRASSIGN_GIT_SHA={git_sha}");
}
