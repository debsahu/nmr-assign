//! Self-hosted HTTP job API, background solver worker, and single-page browser dashboard.
//!
//! Provenance: Clean-room original implementation of `docs/specs/08-service.md` built
//! exclusively on the Rust standard library (`std::net::TcpListener`, `std::thread`,
//! `std::sync`) and `serde` / `serde_json` with zero external HTTP or runtime crates.

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::evaluate::to_reference;
use crate::nef::{
    read_nef_str, write_nef_file, NefDihedralRestraint, NefDihedralRestraintList, NefDocument,
};
use crate::nmrstar::{read_star_str, Reference};
use crate::peaks::{read_peaks_str, Peak};
use crate::priors::{one_to_three, Priors, ATOMS};
use crate::report::{
    generate_quality_report_html, predict_csi_dihedrals, ReportInput, SecondaryStructureClass,
};
use crate::simulate::{simulate, SimConfig};
use crate::solver::{assign, AnnealConfig, ScoreConfig};
use crate::spinsystems::{build_spin_systems, Tolerances};

const EMBEDDED_UBIQUITIN_STAR: &str =
    include_str!("../../../data/fixtures/synthetic_ubiquitin.str");

const MAX_HTTP_BODY_BYTES: usize = 64 * 1024 * 1024; // 64 MiB

/// Payload accepted by `POST /api/v1/jobs`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct JobSubmitRequest {
    #[serde(default)]
    pub project_name: Option<String>,
    /// Raw 1-letter sequence or FASTA text (`>header\nMQIFVK...`).
    #[serde(default)]
    pub sequence: Option<String>,
    /// Optional NMR-STAR text supplying the polymer sequence (and optional shifts).
    #[serde(default)]
    pub star_text: Option<String>,
    /// Optional NEF v1.1 text supplying `nef_molecular_system` and/or `nef_nmr_spectrum` peaks.
    #[serde(default)]
    pub nef_text: Option<String>,
    /// Optional TSV peak list (`experiment\tH\tN\tC\tsign...`).
    #[serde(default)]
    pub peaks_tsv: Option<String>,
    /// Number of simulated annealing replicas (defaults to 16).
    #[serde(default)]
    pub n_runs: Option<usize>,
    /// Annealing steps per spin system (defaults to 1500).
    #[serde(default)]
    pub steps_per_system: Option<usize>,
    /// Deterministic PRNG seed (defaults to 0).
    #[serde(default)]
    pub seed: Option<u64>,
    /// Starting `Seq_ID` when `sequence` is provided as raw/FASTA text (defaults to 1).
    #[serde(default)]
    pub first_seq_id: Option<i32>,
}

/// Summary metrics computed for a completed assignment job.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JobMetrics {
    pub n_residues: usize,
    pub n_assignable: usize,
    pub n_peaks: usize,
    pub n_spin_systems: usize,
    pub n_assigned: usize,
    pub n_high_confidence: usize,
    pub mean_confidence: f64,
    pub solver_score: f64,
    pub elapsed_secs: f64,
}

/// Downloadable artifact URLs for a completed job.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobArtifacts {
    pub shifts_star: String,
    pub result_nef: String,
    pub report_html: String,
}

/// Status record persisted to `<storage_dir>/<job_id>/status.json` and returned by `GET /api/v1/jobs/<id>`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JobRecord {
    pub job_id: String,
    pub project_name: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub created_at_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<JobMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifacts: Option<JobArtifacts>,
}

struct SharedState {
    storage_dir: PathBuf,
    jobs: Mutex<BTreeMap<String, JobRecord>>,
    job_order: Mutex<Vec<String>>,
    job_tx: Mutex<Option<mpsc::Sender<String>>>,
    next_id: AtomicU64,
    shutdown: AtomicBool,
}

/// Handle to a background `ServiceServer` instance bound on a local `SocketAddr`.
pub struct ServiceHandle {
    pub addr: SocketAddr,
    state: Arc<SharedState>,
    accept_thread: Option<JoinHandle<()>>,
    worker_thread: Option<JoinHandle<()>>,
}

impl ServiceHandle {
    /// Bound socket address (useful when binding `"127.0.0.1:0"` in tests).
    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    /// Base `http://<ip>:<port>` URL string.
    pub fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// Signal clean shutdown, unblock the TCP accept loop, and join background threads.
    pub fn shutdown(&mut self) {
        if !self.state.shutdown.swap(true, Ordering::SeqCst) {
            // Close the worker job queue channel so the worker exits cleanly
            if let Ok(mut tx_guard) = self.state.job_tx.lock() {
                *tx_guard = None;
            }
            // Wake the blocking `TcpListener::accept()` call immediately
            let _ = TcpStream::connect_timeout(&self.addr, Duration::from_millis(200));
        }
        if let Some(h) = self.accept_thread.take() {
            let _ = h.join();
        }
        if let Some(h) = self.worker_thread.take() {
            let _ = h.join();
        }
    }
}

impl Drop for ServiceHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Self-hosted NMR assignment pipeline HTTP server.
pub struct ServiceServer;

impl ServiceServer {
    /// Start the HTTP server in the foreground (blocking until interrupted).
    pub fn start_server(bind_addr: &str, storage_dir: PathBuf) -> Result<(), String> {
        let mut handle = Self::spawn_background(bind_addr, storage_dir)?;
        if let Some(accept_join) = handle.accept_thread.take() {
            accept_join
                .join()
                .map_err(|_| "accept loop thread panicked".to_string())?;
        }
        Ok(())
    }

    /// Spawn the HTTP server + background solver worker on dedicated threads and return a `ServiceHandle`.
    pub fn spawn_background(
        bind_addr: &str,
        storage_dir: PathBuf,
    ) -> Result<ServiceHandle, String> {
        fs::create_dir_all(&storage_dir).map_err(|e| {
            format!(
                "failed to create service storage directory {}: {e}",
                storage_dir.display()
            )
        })?;

        let listener = TcpListener::bind(bind_addr)
            .map_err(|e| format!("failed to bind HTTP listener on {bind_addr}: {e}"))?;
        let bound_addr = listener
            .local_addr()
            .map_err(|e| format!("failed to query local socket addr: {e}"))?;

        let (tx, rx) = mpsc::channel::<String>();

        let (existing_jobs, existing_order, max_num) = reload_existing_jobs(&storage_dir);
        let state = Arc::new(SharedState {
            storage_dir,
            jobs: Mutex::new(existing_jobs),
            job_order: Mutex::new(existing_order),
            job_tx: Mutex::new(Some(tx)),
            next_id: AtomicU64::new(max_num + 1),
            shutdown: AtomicBool::new(false),
        });

        let worker_state = Arc::clone(&state);
        let worker_thread = thread::Builder::new()
            .name("nmr-service-worker".to_string())
            .spawn(move || worker_loop(worker_state, rx))
            .map_err(|e| format!("failed to spawn worker thread: {e}"))?;

        let accept_state = Arc::clone(&state);
        let accept_thread = thread::Builder::new()
            .name("nmr-service-http".to_string())
            .spawn(move || accept_loop(listener, accept_state))
            .map_err(|e| format!("failed to spawn accept thread: {e}"))?;

        Ok(ServiceHandle {
            addr: bound_addr,
            state,
            accept_thread: Some(accept_thread),
            worker_thread: Some(worker_thread),
        })
    }
}

/// Start the hosted NMR pipeline HTTP server in the foreground.
pub fn start_server(bind_addr: &str, storage_dir: PathBuf) -> Result<(), String> {
    ServiceServer::start_server(bind_addr, storage_dir)
}

/// Spawn the hosted NMR pipeline HTTP server in the background and return its bound address + shutdown handle.
pub fn spawn_background_server(
    bind_addr: &str,
    storage_dir: PathBuf,
) -> Result<ServiceHandle, String> {
    ServiceServer::spawn_background(bind_addr, storage_dir)
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn reload_existing_jobs(storage_dir: &Path) -> (BTreeMap<String, JobRecord>, Vec<String>, u64) {
    let mut map = BTreeMap::new();
    let mut order = Vec::new();
    let mut max_num = 0u64;

    if let Ok(entries) = fs::read_dir(storage_dir) {
        let mut recs: Vec<JobRecord> = Vec::new();
        for entry in entries.flatten() {
            let status_path = entry.path().join("status.json");
            if let Ok(txt) = fs::read_to_string(&status_path) {
                if let Ok(rec) = serde_json::from_str::<JobRecord>(&txt) {
                    if let Some(num_part) = rec
                        .job_id
                        .strip_prefix("job-")
                        .and_then(|s| s.split('-').next())
                        .and_then(|s| s.parse::<u64>().ok())
                    {
                        max_num = max_num.max(num_part);
                    }
                    recs.push(rec);
                }
            }
        }
        recs.sort_by_key(|r| r.created_at_ms);
        for r in recs {
            order.push(r.job_id.clone());
            map.insert(r.job_id.clone(), r);
        }
    }
    (map, order, max_num)
}

// ============================================================================
// HTTP/1.1 Accept Loop & Request Router
// ============================================================================

fn accept_loop(listener: TcpListener, state: Arc<SharedState>) {
    for incoming in listener.incoming() {
        if state.shutdown.load(Ordering::SeqCst) {
            break;
        }
        let Ok(stream) = incoming else {
            continue;
        };
        if state.shutdown.load(Ordering::SeqCst) {
            break;
        }
        let req_state = Arc::clone(&state);
        // Handle each HTTP request on a short-lived thread so slow uploads never block polling
        thread::spawn(move || {
            let _ = handle_connection(stream, &req_state);
        });
    }
}

struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn read_http_request(stream: &mut TcpStream) -> Result<Option<HttpRequest>, String> {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));

    let mut reader = BufReader::new(stream);
    let mut req_line = String::new();
    let n = reader
        .read_line(&mut req_line)
        .map_err(|e| format!("read request line failed: {e}"))?;
    if n == 0 || req_line.trim().is_empty() {
        return Ok(None);
    }

    let mut parts = req_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_ascii_uppercase();
    let raw_target = parts.next().unwrap_or("/");
    let path = raw_target.split('?').next().unwrap_or("/").to_string();

    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        let bytes = reader
            .read_line(&mut line)
            .map_err(|e| format!("read header failed: {e}"))?;
        if bytes == 0 || line == "\r\n" || line == "\n" {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                content_length = v
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| "invalid Content-Length header".to_string())?;
                if content_length > MAX_HTTP_BODY_BYTES {
                    return Err("payload exceeds maximum size limit (64 MiB)".to_string());
                }
            }
        }
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader
            .read_exact(&mut body)
            .map_err(|e| format!("incomplete request body: {e}"))?;
    }

    Ok(Some(HttpRequest { method, path, body }))
}

fn write_http_response(
    stream: &mut TcpStream,
    status_code: u16,
    status_text: &str,
    content_type: &str,
    extra_headers: &[(&str, &str)],
    body: &[u8],
) -> Result<(), String> {
    let mut hdr = format!(
        "HTTP/1.1 {status_code} {status_text}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
         Access-Control-Allow-Headers: Content-Type\r\n\
         Connection: close\r\n",
        body.len()
    );
    for &(k, v) in extra_headers {
        hdr.push_str(&format!("{k}: {v}\r\n"));
    }
    hdr.push_str("\r\n");

    stream
        .write_all(hdr.as_bytes())
        .and_then(|_| stream.write_all(body))
        .and_then(|_| stream.flush())
        .map_err(|e| format!("failed to write HTTP response: {e}"))
}

fn write_json<T: Serialize>(
    stream: &mut TcpStream,
    status_code: u16,
    status_text: &str,
    payload: &T,
) -> Result<(), String> {
    let bytes = serde_json::to_vec(payload).unwrap_or_else(|_| b"{\"error\":\"json\"}".to_vec());
    write_http_response(
        stream,
        status_code,
        status_text,
        "application/json; charset=utf-8",
        &[],
        &bytes,
    )
}

fn write_json_error(
    stream: &mut TcpStream,
    status_code: u16,
    status_text: &str,
    msg: &str,
) -> Result<(), String> {
    #[derive(Serialize)]
    struct ErrBody<'a> {
        error: &'a str,
    }
    write_json(stream, status_code, status_text, &ErrBody { error: msg })
}

fn is_safe_job_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 80
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn handle_connection(mut stream: TcpStream, state: &Arc<SharedState>) -> Result<(), String> {
    let req = match read_http_request(&mut stream) {
        Ok(Some(r)) => r,
        Ok(None) => return Ok(()),
        Err(e) => {
            return write_json_error(&mut stream, 400, "Bad Request", &e);
        }
    };

    if req.method == "OPTIONS" {
        return write_http_response(&mut stream, 204, "No Content", "text/plain", &[], b"");
    }

    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/") | ("GET", "/index.html") => {
            let html = DASHBOARD_HTML.replace("__NMR_VERSION__", crate::VERSION);
            write_http_response(
                &mut stream,
                200,
                "OK",
                "text/html; charset=utf-8",
                &[],
                html.as_bytes(),
            )
        }
        ("GET", "/api/v1/health") => {
            #[derive(Serialize)]
            struct HealthResp<'a> {
                status: &'a str,
                version: &'a str,
            }
            write_json(
                &mut stream,
                200,
                "OK",
                &HealthResp {
                    status: "ok",
                    version: crate::VERSION,
                },
            )
        }
        ("GET", "/api/v1/demo/ubiquitin") => handle_demo_ubiquitin(&mut stream),
        ("POST", "/api/v1/jobs") => handle_submit_job(&mut stream, &req.body, state),
        ("GET", "/api/v1/jobs") => handle_list_jobs(&mut stream, state),
        ("GET", path) if path.starts_with("/api/v1/jobs/") => {
            let tail = &path["/api/v1/jobs/".len()..];
            if let Some((job_id, artifact)) = tail.split_once('/') {
                handle_get_artifact(&mut stream, job_id, artifact, state)
            } else {
                handle_get_job(&mut stream, tail, state)
            }
        }
        _ => write_json_error(&mut stream, 404, "Not Found", "endpoint not found"),
    }
}

fn handle_demo_ubiquitin(stream: &mut TcpStream) -> Result<(), String> {
    #[derive(Serialize)]
    struct DemoPayload {
        project_name: String,
        sequence: String,
        star_text: String,
        peaks_tsv: String,
    }
    let ref_data = read_star_str(EMBEDDED_UBIQUITIN_STAR, None)?;
    let (peaks, _) = simulate(
        &ref_data,
        &SimConfig {
            seed: 0,
            ..SimConfig::default()
        },
    );
    let peaks_tsv = format_peaks_tsv(&peaks);
    let payload = DemoPayload {
        project_name: "ubiquitin_demo".to_string(),
        sequence: ref_data.one_letter(),
        star_text: EMBEDDED_UBIQUITIN_STAR.to_string(),
        peaks_tsv,
    };
    write_json(stream, 200, "OK", &payload)
}

fn handle_submit_job(
    stream: &mut TcpStream,
    body: &[u8],
    state: &Arc<SharedState>,
) -> Result<(), String> {
    let submit_req: JobSubmitRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            let err_msg = format!("invalid JSON job request: {e}");
            let _ = write_json_error(stream, 400, "Bad Request", &err_msg);
            return Err(err_msg);
        }
    };

    if let Err(validation_err) = validate_job_request(&submit_req) {
        return write_json_error(stream, 400, "Bad Request", &validation_err);
    }

    let seq_num = state.next_id.fetch_add(1, Ordering::SeqCst);
    let ts_ms = now_unix_ms();
    let job_id = format!("job-{seq_num:04}-{:06x}", ts_ms & 0xFF_FFFF);
    let project_name = submit_req
        .project_name
        .clone()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| format!("nmr_job_{seq_num:04}"));

    let job_dir = state.storage_dir.join(&job_id);
    if let Err(e) = fs::create_dir_all(&job_dir) {
        return write_json_error(
            stream,
            500,
            "Internal Server Error",
            &format!("failed to create job directory: {e}"),
        );
    }

    let req_json = serde_json::to_vec_pretty(&submit_req).unwrap_or_default();
    if let Err(e) = fs::write(job_dir.join("request.json"), req_json) {
        return write_json_error(
            stream,
            500,
            "Internal Server Error",
            &format!("failed to persist job request: {e}"),
        );
    }

    let record = JobRecord {
        job_id: job_id.clone(),
        project_name,
        status: "queued".to_string(),
        error: None,
        created_at_ms: ts_ms,
        completed_at_ms: None,
        metrics: None,
        artifacts: None,
    };
    persist_job_status(&job_dir, &record);

    {
        if let Ok(mut jobs) = state.jobs.lock() {
            jobs.insert(job_id.clone(), record);
        }
        if let Ok(mut order) = state.job_order.lock() {
            order.push(job_id.clone());
        }
    }

    let sent = state
        .job_tx
        .lock()
        .ok()
        .and_then(|opt| opt.as_ref().map(|tx| tx.send(job_id.clone()).is_ok()))
        .unwrap_or(false);
    if !sent {
        return write_json_error(
            stream,
            503,
            "Service Unavailable",
            "background worker is shutting down",
        );
    }

    #[derive(Serialize)]
    struct AcceptedResp<'a> {
        job_id: &'a str,
        status: &'a str,
    }
    write_json(
        stream,
        202,
        "Accepted",
        &AcceptedResp {
            job_id: &job_id,
            status: "queued",
        },
    )
}

fn handle_list_jobs(stream: &mut TcpStream, state: &Arc<SharedState>) -> Result<(), String> {
    #[derive(Serialize)]
    struct JobsListResp {
        jobs: Vec<JobRecord>,
    }
    let mut list = Vec::new();
    if let (Ok(order), Ok(jobs)) = (state.job_order.lock(), state.jobs.lock()) {
        for id in order.iter().rev() {
            if let Some(rec) = jobs.get(id) {
                list.push(rec.clone());
            }
        }
    }
    write_json(stream, 200, "OK", &JobsListResp { jobs: list })
}

fn handle_get_job(
    stream: &mut TcpStream,
    job_id: &str,
    state: &Arc<SharedState>,
) -> Result<(), String> {
    if !is_safe_job_id(job_id) {
        return write_json_error(stream, 400, "Bad Request", "invalid job id");
    }
    let rec_opt = state
        .jobs
        .lock()
        .ok()
        .and_then(|jobs| jobs.get(job_id).cloned());
    match rec_opt {
        Some(rec) => write_json(stream, 200, "OK", &rec),
        None => write_json_error(stream, 404, "Not Found", "job not found"),
    }
}

fn handle_get_artifact(
    stream: &mut TcpStream,
    job_id: &str,
    artifact: &str,
    state: &Arc<SharedState>,
) -> Result<(), String> {
    if !is_safe_job_id(job_id) {
        return write_json_error(stream, 400, "Bad Request", "invalid job id");
    }
    let (filename, content_type, disposition) = match artifact {
        "shifts.str" => (
            "shifts.str",
            "text/plain; charset=utf-8",
            Some(("Content-Disposition", "inline; filename=\"shifts.str\"")),
        ),
        "result.nef" => (
            "result.nef",
            "text/plain; charset=utf-8",
            Some(("Content-Disposition", "inline; filename=\"result.nef\"")),
        ),
        "report.html" => ("report.html", "text/html; charset=utf-8", None),
        _ => return write_json_error(stream, 404, "Not Found", "unknown artifact"),
    };

    let path = state.storage_dir.join(job_id).join(filename);
    let Ok(bytes) = fs::read(&path) else {
        return write_json_error(stream, 404, "Not Found", "artifact not available yet");
    };

    let extra: Vec<(&str, &str)> = disposition.into_iter().collect();
    write_http_response(stream, 200, "OK", content_type, &extra, &bytes)
}

fn validate_job_request(req: &JobSubmitRequest) -> Result<(), String> {
    let has_seq = req.sequence.as_ref().is_some_and(|s| !s.trim().is_empty())
        || req.star_text.as_ref().is_some_and(|s| !s.trim().is_empty())
        || req.nef_text.as_ref().is_some_and(|s| !s.trim().is_empty());
    if !has_seq {
        return Err(
            "job requires `sequence` (FASTA/1-letter), `star_text`, or `nef_text`".to_string(),
        );
    }

    let has_peaks = req.peaks_tsv.as_ref().is_some_and(|s| !s.trim().is_empty())
        || req.nef_text.as_ref().is_some_and(|s| !s.trim().is_empty());
    if !has_peaks {
        return Err("job requires `peaks_tsv` or `nef_text` containing peak lists".to_string());
    }
    Ok(())
}

// ============================================================================
// Background Worker Pipeline
// ============================================================================

fn persist_job_status(job_dir: &Path, rec: &JobRecord) {
    if let Ok(bytes) = serde_json::to_vec_pretty(rec) {
        let _ = fs::write(job_dir.join("status.json"), bytes);
    }
}

fn update_job_state<F: FnOnce(&mut JobRecord)>(state: &Arc<SharedState>, job_id: &str, mutate: F) {
    let mut snapshot = None;
    if let Ok(mut jobs) = state.jobs.lock() {
        if let Some(rec) = jobs.get_mut(job_id) {
            mutate(rec);
            snapshot = Some(rec.clone());
        }
    }
    if let Some(rec) = snapshot {
        persist_job_status(&state.storage_dir.join(job_id), &rec);
    }
}

fn worker_loop(state: Arc<SharedState>, rx: mpsc::Receiver<String>) {
    while let Ok(job_id) = rx.recv() {
        if state.shutdown.load(Ordering::SeqCst) {
            break;
        }
        update_job_state(&state, &job_id, |rec| {
            rec.status = "running".to_string();
        });

        let job_dir = state.storage_dir.join(&job_id);
        match execute_job(&job_id, &job_dir, &state) {
            Ok((metrics, artifacts)) => {
                update_job_state(&state, &job_id, |rec| {
                    rec.status = "completed".to_string();
                    rec.completed_at_ms = Some(now_unix_ms());
                    rec.metrics = Some(metrics);
                    rec.artifacts = Some(artifacts);
                });
            }
            Err(err) => {
                update_job_state(&state, &job_id, |rec| {
                    rec.status = "failed".to_string();
                    rec.completed_at_ms = Some(now_unix_ms());
                    rec.error = Some(err);
                });
            }
        }
    }
}

fn execute_job(
    job_id: &str,
    job_dir: &Path,
    state: &Arc<SharedState>,
) -> Result<(JobMetrics, JobArtifacts), String> {
    let t0 = Instant::now();
    let req_bytes = fs::read(job_dir.join("request.json"))
        .map_err(|e| format!("failed to read job request.json: {e}"))?;
    let req: JobSubmitRequest = serde_json::from_slice(&req_bytes)
        .map_err(|e| format!("failed to parse job request.json: {e}"))?;

    let project_name = state
        .jobs
        .lock()
        .ok()
        .and_then(|j| j.get(job_id).map(|r| r.project_name.clone()))
        .unwrap_or_else(|| job_id.to_string());

    let parsed_nef = match req.nef_text.as_deref() {
        Some(txt) if !txt.trim().is_empty() => Some(read_nef_str(txt)?),
        _ => None,
    };

    // 1. Resolve sequence & Seq_IDs
    let (sequence, seq_ids) =
        if let Some(seq_txt) = req.sequence.as_deref().filter(|s| !s.trim().is_empty()) {
            parse_fasta_or_raw_sequence(seq_txt, req.first_seq_id.unwrap_or(1))?
        } else if let Some(star_txt) = req.star_text.as_deref().filter(|s| !s.trim().is_empty()) {
            let r = read_star_str(star_txt, None)?;
            (r.sequence, r.seq_ids)
        } else if let Some(ref nef_doc) = parsed_nef {
            let r = nef_doc.to_reference()?;
            (r.sequence, r.seq_ids)
        } else {
            return Err("no sequence found in job request".to_string());
        };

    if sequence.is_empty() {
        return Err("parsed sequence has zero residues".to_string());
    }

    // 2. Resolve peak lists
    let peaks: Vec<Peak> =
        if let Some(tsv) = req.peaks_tsv.as_deref().filter(|s| !s.trim().is_empty()) {
            read_peaks_str(tsv)?
        } else if let Some(ref nef_doc) = parsed_nef {
            let p = nef_doc.to_peaks();
            if p.is_empty() {
                return Err("NEF input contains no spectrum peak lists".to_string());
            }
            p
        } else {
            return Err("no peak lists found in job request".to_string());
        };

    // 3. Spin-system assembly & multi-replica simulated annealing
    let systems = build_spin_systems(&peaks, &Tolerances::default(), 3, false)?;
    let priors = Priors::default();
    let anneal_cfg = AnnealConfig {
        runs: req.n_runs.unwrap_or(16).clamp(1, 256),
        steps_per_system: req.steps_per_system.unwrap_or(1500).clamp(100, 20_000),
        seed: req.seed.unwrap_or(0),
        ..AnnealConfig::default()
    };
    let score_cfg = ScoreConfig::default();

    let assign_res = assign(&systems, &sequence, &priors, &score_cfg, &anneal_cfg);
    let (pred_ref, conf) = to_reference(
        &assign_res,
        &systems,
        &sequence,
        &seq_ids,
        0.0,
        Some(&priors),
    );
    let elapsed_secs = t0.elapsed().as_secs_f64();

    // 4. Write shifts.str (including _Entity_poly_seq so sequence bounds are always preserved)
    let star_out = format_star_with_sequence(&pred_ref, &conf, &project_name);
    fs::write(job_dir.join("shifts.str"), star_out)
        .map_err(|e| format!("failed to write shifts.str: {e}"))?;

    // 5. Predict secondary-shift dihedrals & write canonical NEF v1.1 bundle (result.nef)
    let csi_dihedrals = predict_csi_dihedrals(&pred_ref);
    let mut nef_doc =
        NefDocument::from_reference_and_peaks(&project_name, &pred_ref, Some(&conf), &peaks);
    let dih_restraints = build_nef_dihedrals_from_csi(&pred_ref, &csi_dihedrals);
    if !dih_restraints.is_empty() {
        nef_doc
            .dihedral_restraint_lists
            .push(NefDihedralRestraintList {
                name: "default".to_string(),
                potential_type: "square-well-parabolic".to_string(),
                restraints: dih_restraints,
            });
    }
    if let Some(in_nef) = parsed_nef {
        if !in_nef.distance_restraint_lists.is_empty() {
            nef_doc.distance_restraint_lists = in_nef.distance_restraint_lists;
        }
    }
    write_nef_file(job_dir.join("result.nef"), &nef_doc)?;

    // 6. Generate self-contained HTML + SVG quality report (report.html)
    let mut report_in = ReportInput::new(&project_name, pred_ref.clone(), conf.clone());
    report_in.n_peaks = Some(peaks.len());
    report_in.n_spin_systems = Some(systems.len());
    report_in.solver_score = Some(assign_res.score);
    report_in.elapsed_secs = Some(elapsed_secs);
    report_in.engine_label = Some(format!("CPU ({} replicas)", anneal_cfg.runs));
    report_in.systems_and_assignment = Some((systems.clone(), assign_res.clone()));
    report_in.dihedrals = csi_dihedrals;
    let report_html = generate_quality_report_html(&report_in);
    fs::write(job_dir.join("report.html"), report_html)
        .map_err(|e| format!("failed to write report.html: {e}"))?;

    // 7. Compute summary metrics
    let n_residues = sequence.len();
    let n_assignable = sequence
        .iter()
        .enumerate()
        .filter(|&(k, r)| k > 0 && r != "PRO")
        .count();
    let mut n_assigned = 0usize;
    let mut n_hc = 0usize;
    let mut conf_sum = 0.0f64;
    for (s, &pos) in assign_res.position.iter().enumerate() {
        if pos >= 0 {
            n_assigned += 1;
            let c = assign_res.confidence[s];
            conf_sum += c;
            if c >= 0.80 {
                n_hc += 1;
            }
        }
    }
    let mean_confidence = if n_assigned > 0 {
        conf_sum / (n_assigned as f64)
    } else {
        0.0
    };

    let metrics = JobMetrics {
        n_residues,
        n_assignable,
        n_peaks: peaks.len(),
        n_spin_systems: systems.len(),
        n_assigned,
        n_high_confidence: n_hc,
        mean_confidence,
        solver_score: assign_res.score,
        elapsed_secs,
    };
    let artifacts = JobArtifacts {
        shifts_star: format!("/api/v1/jobs/{job_id}/shifts.str"),
        result_nef: format!("/api/v1/jobs/{job_id}/result.nef"),
        report_html: format!("/api/v1/jobs/{job_id}/report.html"),
    };

    Ok((metrics, artifacts))
}

fn parse_fasta_or_raw_sequence(
    text: &str,
    first_seq_id: i32,
) -> Result<(Vec<String>, Vec<i32>), String> {
    let raw: String = text
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            !t.starts_with('>') && !t.starts_with(';') && !t.starts_with('#')
        })
        .flat_map(|l| l.chars())
        .filter(|c| c.is_ascii_alphabetic())
        .collect();

    if raw.is_empty() {
        return Err("sequence string contains no amino-acid letters".to_string());
    }

    let mut sequence = Vec::with_capacity(raw.len());
    for ch in raw.chars() {
        let three = one_to_three(ch);
        if three == "UNK" {
            return Err(format!("unrecognized one-letter amino acid code '{ch}'"));
        }
        sequence.push(three.to_string());
    }
    let seq_ids: Vec<i32> = (first_seq_id..(first_seq_id + sequence.len() as i32)).collect();
    Ok((sequence, seq_ids))
}

fn format_peaks_tsv(peaks: &[Peak]) -> String {
    let mut out = String::from("experiment\tH\tN\tC\tsign\theight\tvolume\tH2\n");
    for p in peaks {
        let c_str = p.c.map(|v| format!("{v:.3}")).unwrap_or_default();
        let h_str = p.height.map(|v| format!("{v:.2}")).unwrap_or_default();
        let v_str = p.volume.map(|v| format!("{v:.2}")).unwrap_or_default();
        let h2_str = p.h2.map(|v| format!("{v:.4}")).unwrap_or_default();
        out.push_str(&format!(
            "{}\t{:.4}\t{:.3}\t{}\t{}\t{}\t{}\t{}\n",
            p.exp, p.h, p.n, c_str, p.sign, h_str, v_str, h2_str
        ));
    }
    out
}

fn format_star_with_sequence(
    ref_data: &Reference,
    confidence: &BTreeMap<usize, f64>,
    project_name: &str,
) -> String {
    let clean_proj: String = project_name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let mut lines = vec![
        format!(
            "data_{}",
            if clean_proj.is_empty() {
                "nmr_assign_output"
            } else {
                &clean_proj
            }
        ),
        String::new(),
        "save_entity_1".to_string(),
        "   _Entity.Sf_category  entity".to_string(),
        "   _Entity.ID           1".to_string(),
        String::new(),
        "   loop_".to_string(),
        "      _Entity_poly_seq.Entity_ID".to_string(),
        "      _Entity_poly_seq.Num".to_string(),
        "      _Entity_poly_seq.Mon_ID".to_string(),
        String::new(),
    ];
    for (idx, res) in ref_data.sequence.iter().enumerate() {
        let sid = ref_data
            .seq_ids
            .get(idx)
            .copied()
            .unwrap_or((idx + 1) as i32);
        lines.push(format!("      1 {:5} {:3}", sid, res));
    }
    lines.extend([
        "   stop_".to_string(),
        "save_".to_string(),
        String::new(),
        "save_assigned_chem_shift_list_1".to_string(),
        "   _Assigned_chem_shift_list.Sf_category  assigned_chemical_shifts".to_string(),
        String::new(),
        "   loop_".to_string(),
    ]);
    for t in [
        "ID",
        "Entity_ID",
        "Seq_ID",
        "Comp_ID",
        "Atom_ID",
        "Atom_type",
        "Val",
        "Ambiguity_code",
        "Details",
    ] {
        lines.push(format!("      _Atom_chem_shift.{t}"));
    }
    lines.push(String::new());

    let mut entries: Vec<(usize, &str, f64)> = ref_data
        .shifts
        .iter()
        .map(|(&(idx, ref atom), &v)| (idx, atom.as_str(), v))
        .collect();
    entries.sort_by(|&(ia, aa, _), &(ib, ab, _)| {
        let oa = ATOMS.iter().position(|&a| a == aa).unwrap_or(99);
        let ob = ATOMS.iter().position(|&a| a == ab).unwrap_or(99);
        (ia, oa, aa).cmp(&(ib, ob, ab))
    });

    for (i, (idx, atom, v)) in entries.into_iter().enumerate() {
        let det = confidence
            .get(&idx)
            .map(|&c| format!("conf={c:.2}"))
            .unwrap_or_else(|| ".".to_string());
        let atom_type = atom
            .chars()
            .find(|c| c.is_ascii_alphabetic())
            .map(|c| c.to_string())
            .unwrap_or_else(|| "X".to_string());
        lines.push(format!(
            "      {:5} 1 {:5} {:3} {:4} {} {:9.3} 1 {}",
            i + 1,
            ref_data.seq_ids[idx],
            ref_data.sequence[idx],
            atom,
            atom_type,
            v,
            det
        ));
    }
    lines.push("   stop_".to_string());
    lines.push("save_".to_string());
    lines.push(String::new());
    lines.join("\n")
}

fn build_nef_dihedrals_from_csi(
    ref_data: &Reference,
    points: &[crate::report::DihedralPoint],
) -> Vec<NefDihedralRestraint> {
    let mut seq_to_idx = BTreeMap::new();
    for (idx, &sid) in ref_data.seq_ids.iter().enumerate() {
        seq_to_idx.insert(sid, idx);
    }
    let mut out = Vec::new();
    let mut rid = 1usize;
    let mut idx_row = 1usize;

    for pt in points {
        if pt.ss_class == SecondaryStructureClass::Coil {
            continue;
        }
        let Some(&k) = seq_to_idx.get(&pt.seq_id) else {
            continue;
        };
        if k == 0 || k + 1 >= ref_data.sequence.len() {
            continue;
        }
        let prev_sid = ref_data.seq_ids[k - 1];
        let prev_res = &ref_data.sequence[k - 1];
        let next_sid = ref_data.seq_ids[k + 1];
        let next_res = &ref_data.sequence[k + 1];
        let tol = match pt.ss_class {
            SecondaryStructureClass::Helix => 20.0,
            SecondaryStructureClass::Strand => 28.0,
            SecondaryStructureClass::Coil => 35.0,
        };

        out.push(NefDihedralRestraint::phi(
            idx_row,
            rid,
            "A",
            prev_sid,
            prev_res,
            pt.seq_id,
            &pt.residue,
            pt.phi_deg,
            tol,
            1.0,
        ));
        idx_row += 1;
        rid += 1;

        out.push(NefDihedralRestraint::psi(
            idx_row,
            rid,
            "A",
            pt.seq_id,
            &pt.residue,
            next_sid,
            next_res,
            pt.psi_deg,
            tol,
            1.0,
        ));
        idx_row += 1;
        rid += 1;
    }
    out
}

// ============================================================================
// Single-Page Browser Dashboard UI (GET /)
// ============================================================================

const DASHBOARD_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>nmr-assign — Automated NMR Resonance Assignment Workbench</title>
<style>
  :root {
    --bg: #0b1120;
    --panel: #111827;
    --panel-alt: #1e293b;
    --border: #334155;
    --text: #f8fafc;
    --muted: #94a3b8;
    --accent: #14b8a6;
    --accent-hover: #0d9488;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
  }
  header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 28px;
    background: rgba(17, 24, 39, 0.92);
    border-bottom: 1px solid var(--border);
  }
  .brand { display: flex; align-items: center; gap: 12px; }
  .brand-badge {
    background: linear-gradient(135deg, #14b8a6, #3b82f6);
    color: #0f172a; font-weight: 800; font-size: 13px;
    padding: 4px 9px; border-radius: 6px;
  }
  header h1 { margin: 0; font-size: 17px; font-weight: 600; }
  .demo-btn {
    background: #1e293b; color: #5eead4; border: 1px solid #14b8a6;
    padding: 8px 14px; border-radius: 8px; font-weight: 600; font-size: 13px;
    cursor: pointer; transition: all 0.15s;
  }
  .demo-btn:hover { background: #14b8a6; color: #0f172a; }
  .layout {
    display: grid;
    grid-template-columns: 390px 1fr;
    gap: 20px;
    padding: 20px 28px;
    max-width: 1600px;
    margin: 0 auto;
  }
  @media (max-width: 1024px) { .layout { grid-template-columns: 1fr; } }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 16px;
  }
  .card h2 { margin: 0 0 12px; font-size: 15px; color: #e2e8f0; }
  label { display: block; font-size: 12px; font-weight: 600; color: var(--muted); margin: 10px 0 4px; }
  input[type="text"], input[type="number"], textarea {
    width: 100%; background: #0f172a; color: var(--text);
    border: 1px solid var(--border); border-radius: 8px;
    padding: 8px 10px; font-size: 12.5px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  textarea { resize: vertical; min-height: 78px; }
  .dropzone {
    border: 1.5px dashed #475569; border-radius: 8px;
    padding: 10px; text-align: center; font-size: 12px; color: var(--muted);
    margin-bottom: 8px; cursor: pointer;
  }
  .dropzone.dragover { border-color: var(--accent); background: rgba(20,184,166,0.08); color: #5eead4; }
  .row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .submit-btn {
    width: 100%; margin-top: 14px; padding: 11px;
    background: var(--accent); color: #042f2e; border: none; border-radius: 8px;
    font-weight: 700; font-size: 14px; cursor: pointer;
  }
  .submit-btn:hover { background: var(--accent-hover); color: #ffffff; }
  .job-item {
    display: flex; justify-content: space-between; align-items: center;
    padding: 9px 11px; border-radius: 8px; background: #0f172a;
    border: 1px solid #1e293b; margin-bottom: 6px; cursor: pointer; font-size: 12.5px;
  }
  .job-item.active { border-color: var(--accent); }
  .status-pill {
    padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; text-transform: uppercase;
  }
  .st-queued, .st-running { background: #fef3c7; color: #92400e; }
  .st-completed { background: #d1fae5; color: #065f46; }
  .st-failed { background: #fee2e2; color: #991b1b; }
  .toolbar {
    display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
    gap: 10px; margin-bottom: 12px;
  }
  .dl-group { display: flex; gap: 8px; flex-wrap: wrap; }
  .dl-link {
    display: inline-block; padding: 7px 12px; border-radius: 7px;
    background: var(--panel-alt); color: #e2e8f0; text-decoration: none;
    border: 1px solid var(--border); font-size: 12.5px; font-weight: 600;
  }
  .dl-link:hover { border-color: var(--accent); color: #5eead4; }
  iframe#report-frame {
    width: 100%; height: 780px; border: 1px solid var(--border);
    border-radius: 10px; background: #f1f5f9;
  }
  .empty-state {
    padding: 80px 24px; text-align: center; color: var(--muted);
    border: 1px dashed var(--border); border-radius: 10px;
  }
</style>
</head>
<body>
<header>
  <div class="brand">
    <span class="brand-badge">NMR v__NMR_VERSION__</span>
    <h1>Automated Protein NMR Backbone Assignment &amp; NEF Pipeline</h1>
  </div>
  <button class="demo-btn" id="btn-demo" type="button">Load Synthetic Ubiquitin Demo</button>
</header>

<div class="layout">
  <div>
    <div class="card">
      <h2>Submit Assignment Job</h2>
      <div class="dropzone" id="dropzone">
        Drag &amp; drop <code>.fasta</code>, <code>.str</code>, <code>.tsv</code>, or <code>.nef</code> files here (or click to browse)
        <input type="file" id="file-input" multiple style="display:none">
      </div>
      <label for="inp-project">Project Name</label>
      <input type="text" id="inp-project" value="ubiquitin_run" placeholder="project_name">

      <label for="inp-seq">Sequence (FASTA / 1-Letter) or NMR-STAR (.str)</label>
      <textarea id="inp-seq" placeholder=">ubiquitin&#10;MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"></textarea>

      <label for="inp-peaks">Triple-Resonance Peak List (.tsv)</label>
      <textarea id="inp-peaks" placeholder="experiment&#9;H&#9;N&#9;C&#9;sign&#10;HSQC&#9;8.429&#9;121.93&#9;&#9;1"></textarea>

      <label for="inp-nef">Or Unified NEF v1.1 Document (.nef)</label>
      <textarea id="inp-nef" placeholder="data_nef_project ..."></textarea>

      <div class="row2">
        <div>
          <label for="inp-runs">SA Replicas</label>
          <input type="number" id="inp-runs" value="16" min="1" max="128">
        </div>
        <div>
          <label for="inp-seed">Random Seed</label>
          <input type="number" id="inp-seed" value="0" min="0">
        </div>
      </div>

      <button class="submit-btn" id="btn-submit" type="button">Run Resonance Assignment</button>
      <div id="submit-msg" style="margin-top:8px;font-size:12px;color:#5eead4;"></div>
    </div>

    <div class="card">
      <h2>Submitted Jobs</h2>
      <div id="job-list"><div style="font-size:12px;color:#64748b;">No jobs submitted yet.</div></div>
    </div>
  </div>

  <div>
    <div class="card">
      <div class="toolbar">
        <div>
          <strong id="active-title">Quality Report Viewer</strong>
          <span id="active-status" style="margin-left:10px;"></span>
        </div>
        <div class="dl-group" id="dl-links" style="display:none;">
          <a class="dl-link" id="dl-star" href="#" download="shifts.str">Download NMR-STAR (.str)</a>
          <a class="dl-link" id="dl-nef" href="#" download="result.nef">Download NEF v1.1 (.nef)</a>
          <a class="dl-link" id="dl-html" href="#" download="report.html">Download HTML Report</a>
        </div>
      </div>
      <div id="empty-viewer" class="empty-state">
        Click <strong>Load Synthetic Ubiquitin Demo</strong> above, then press <strong>Run Resonance Assignment</strong> to inspect interactive SVG confidence ribbons, CSI secondary shifts, strip ladders, and Ramachandran plots.
      </div>
      <iframe id="report-frame" style="display:none;" title="Assignment Quality Report"></iframe>
    </div>
  </div>
</div>

<script>
let currentJobId = null;
let loadedReportFor = null;

async function loadDemo() {
  const msg = document.getElementById('submit-msg');
  msg.textContent = 'Loading synthetic ubiquitin fixture...';
  const res = await fetch('/api/v1/demo/ubiquitin');
  const data = await res.json();
  document.getElementById('inp-project').value = data.project_name;
  document.getElementById('inp-seq').value = '>ubiquitin_1_76\n' + data.sequence;
  document.getElementById('inp-peaks').value = data.peaks_tsv;
  document.getElementById('inp-nef').value = '';
  msg.textContent = 'Loaded 76-residue ubiquitin sequence + simulated 7-spectrum peak list.';
}

async function submitJob() {
  const msg = document.getElementById('submit-msg');
  const rawSeq = document.getElementById('inp-seq').value.trim();
  const peaksTsv = document.getElementById('inp-peaks').value.trim();
  const nefText = document.getElementById('inp-nef').value.trim();
  const payload = {
    project_name: document.getElementById('inp-project').value.trim() || 'nmr_job',
    n_runs: parseInt(document.getElementById('inp-runs').value || '16', 10),
    seed: parseInt(document.getElementById('inp-seed').value || '0', 10)
  };
  if (rawSeq.includes('save_') || rawSeq.includes('_Atom_chem_shift')) {
    payload.star_text = rawSeq;
  } else if (rawSeq.length > 0) {
    payload.sequence = rawSeq;
  }
  if (peaksTsv.length > 0) payload.peaks_tsv = peaksTsv;
  if (nefText.length > 0) payload.nef_text = nefText;

  msg.textContent = 'Submitting job...';
  const res = await fetch('/api/v1/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  if (!res.ok) {
    msg.textContent = 'Error: ' + (data.error || res.statusText);
    return;
  }
  currentJobId = data.job_id;
  loadedReportFor = null;
  msg.textContent = 'Queued ' + data.job_id;
  pollJobs();
}

async function pollJobs() {
  const res = await fetch('/api/v1/jobs');
  if (!res.ok) return;
  const data = await res.json();
  const listEl = document.getElementById('job-list');
  if (!data.jobs || data.jobs.length === 0) return;

  if (!currentJobId) currentJobId = data.jobs[0].job_id;
  listEl.innerHTML = '';
  for (const j of data.jobs) {
    const div = document.createElement('div');
    div.className = 'job-item' + (j.job_id === currentJobId ? ' active' : '');
    div.innerHTML = `<span><strong>${j.project_name}</strong> <small style="color:#64748b">(${j.job_id})</small></span>
                     <span class="status-pill st-${j.status}">${j.status}</span>`;
    div.onclick = () => { currentJobId = j.job_id; loadedReportFor = null; pollActiveJob(); pollJobs(); };
    listEl.appendChild(div);
  }
  pollActiveJob();
}

async function pollActiveJob() {
  if (!currentJobId) return;
  const res = await fetch('/api/v1/jobs/' + currentJobId);
  if (!res.ok) return;
  const j = await res.json();
  document.getElementById('active-title').textContent = j.project_name + ' (' + j.job_id + ')';
  document.getElementById('active-status').innerHTML = `<span class="status-pill st-${j.status}">${j.status}</span>`;

  if (j.status === 'completed' && j.artifacts) {
    document.getElementById('dl-links').style.display = 'flex';
    document.getElementById('dl-star').href = j.artifacts.shifts_star;
    document.getElementById('dl-nef').href = j.artifacts.result_nef;
    document.getElementById('dl-html').href = j.artifacts.report_html;
    if (loadedReportFor !== j.job_id) {
      loadedReportFor = j.job_id;
      document.getElementById('empty-viewer').style.display = 'none';
      const frame = document.getElementById('report-frame');
      frame.style.display = 'block';
      frame.src = j.artifacts.report_html;
    }
  } else if (j.status === 'failed') {
    document.getElementById('dl-links').style.display = 'none';
    document.getElementById('report-frame').style.display = 'none';
    const ev = document.getElementById('empty-viewer');
    ev.style.display = 'block';
    ev.textContent = 'Job failed: ' + (j.error || 'unknown error');
  }
}

function handleFiles(files) {
  for (const f of files) {
    const reader = new FileReader();
    reader.onload = (ev) => {
      const txt = ev.target.result;
      const low = f.name.toLowerCase();
      if (low.endsWith('.nef') || txt.includes('data_nef_')) {
        document.getElementById('inp-nef').value = txt;
      } else if (low.endsWith('.tsv') || txt.startsWith('experiment\t')) {
        document.getElementById('inp-peaks').value = txt;
      } else {
        document.getElementById('inp-seq').value = txt;
      }
    };
    reader.readAsText(f);
  }
}

const dz = document.getElementById('dropzone');
const fi = document.getElementById('file-input');
dz.onclick = () => fi.click();
fi.onchange = (e) => handleFiles(e.target.files);
dz.ondragover = (e) => { e.preventDefault(); dz.classList.add('dragover'); };
dz.ondragleave = () => dz.classList.remove('dragover');
dz.ondrop = (e) => { e.preventDefault(); dz.classList.remove('dragover'); handleFiles(e.dataTransfer.files); };

document.getElementById('btn-demo').onclick = loadDemo;
document.getElementById('btn-submit').onclick = submitJob;
setInterval(pollJobs, 1000);
pollJobs();
</script>
</body>
</html>
"##;
