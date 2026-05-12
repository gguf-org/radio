use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager};

const SERVER_PORT: u16 = 8081;
const SERVER_HOST: &str = "127.0.0.1";
const HEALTH_POLL_MS: u64 = 500;
const HEALTH_TIMEOUT_S: u64 = 120;

struct AppState {
    llama_child: Mutex<Option<Child>>,
}

// ── Platform helpers ────────────────────────────────────────────────────────

#[cfg(windows)]
fn suppress_console(cmd: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x08000000;
    cmd.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn suppress_console(_: &mut Command) {}

fn binary_name(stem: &str) -> String {
    if cfg!(windows) {
        format!("{}.exe", stem)
    } else {
        stem.to_string()
    }
}

// ── Path resolution ─────────────────────────────────────────────────────────

/// Find the llama-server binary bundled with this app.
/// Tauri's externalBin places the binary (without target triple) next to the exe
/// in production. During `tauri dev`, it resolves from the src-tauri/binaries/ dir.
fn resolve_llama_server(app: &AppHandle) -> Option<PathBuf> {
    let name = binary_name("llama-server");

    // 1. resource_dir — where Tauri places externalBin in production bundles
    if let Ok(dir) = app.path().resource_dir() {
        let candidate = dir.join(&name);
        if candidate.exists() {
            return Some(candidate);
        }
    }

    // 2. Same directory as the running exe (production on some platforms)
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join(&name);
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    // 3. src-tauri/binaries/ — works during `tauri dev`
    if let Ok(exe) = std::env::current_exe() {
        // Walk up from the exe to find the workspace root heuristically
        let mut dir = exe.parent().map(Path::to_path_buf);
        for _ in 0..8 {
            if let Some(d) = dir {
                let candidate = d.join("binaries").join(&name);
                if candidate.exists() {
                    return Some(candidate);
                }
                dir = d.parent().map(Path::to_path_buf);
            } else {
                break;
            }
        }
    }

    None
}


// ── Server lifecycle ────────────────────────────────────────────────────────

fn kill_existing(state: &AppState) {
    if let Ok(mut guard) = state.llama_child.lock() {
        if let Some(mut child) = guard.take() {
            child.kill().ok();
            child.wait().ok();
        }
    }
}

fn spawn_llama_server(
    binary: &Path,
    model: &Path,
    mmproj: &Path,
    gpu_layers: u32,
    dylib_dirs: &[PathBuf],
) -> Result<Child, String> {
    let mut cmd = Command::new(binary);
    cmd.args([
        "--model",
        &model.to_string_lossy(),
        "--mmproj",
        &mmproj.to_string_lossy(),
        "--host",
        SERVER_HOST,
        "--port",
        &SERVER_PORT.to_string(),
        "--ctx-size",
        "4096",
        "--n-gpu-layers",
        &gpu_layers.to_string(),
        "--log-disable",
    ])
    .stdout(Stdio::null())
    .stderr(Stdio::null())
    .stdin(Stdio::null());

    // Windows: set working dir so the exe-directory DLL search finds sibling DLLs.
    // macOS: set DYLD_LIBRARY_PATH so the dynamic linker finds bundled .dylib files.
    #[cfg(windows)]
    if let Some(parent) = binary.parent() {
        cmd.current_dir(parent);
    }

    #[cfg(target_os = "macos")]
    if !dylib_dirs.is_empty() {
        let path_val = dylib_dirs
            .iter()
            .map(|p| p.to_string_lossy())
            .collect::<Vec<_>>()
            .join(":");
        cmd.env("DYLD_LIBRARY_PATH", path_val);
    }

    suppress_console(&mut cmd);
    cmd.spawn().map_err(|e| format!("Failed to spawn llama-server: {e}"))
}

/// Poll until the server's TCP port accepts a connection or we time out.
fn wait_for_server(app: &AppHandle, model_id: String) {
    let addr = format!("{}:{}", SERVER_HOST, SERVER_PORT);
    let max_iters = (HEALTH_TIMEOUT_S * 1000) / HEALTH_POLL_MS;
    let app = app.clone();

    std::thread::spawn(move || {
        for _ in 0..max_iters {
            std::thread::sleep(Duration::from_millis(HEALTH_POLL_MS));
            if TcpStream::connect(&addr).is_ok() {
                let _ = app.emit(
                    "llama-server-ready",
                    serde_json::json!({
                        "running": true,
                        "url": format!("http://{}:{}/v1", SERVER_HOST, SERVER_PORT),
                        "modelId": model_id,
                    }),
                );
                return;
            }
        }
        let _ = app.emit(
            "llama-server-ready",
            serde_json::json!({
                "error": format!(
                    "llama-server did not become ready within {} seconds on port {}",
                    HEALTH_TIMEOUT_S, SERVER_PORT
                )
            }),
        );
    });
}

// ── Tauri commands ──────────────────────────────────────────────────────────

#[derive(serde::Serialize)]
struct StartResult {
    starting: bool,
    port: u16,
}

#[tauri::command]
fn start_llama_server(
    app: AppHandle,
    state: tauri::State<AppState>,
    gpu_layers: Option<u32>,
    model_path: String,
    mmproj_path: String,
) -> Result<StartResult, String> {
    kill_existing(&state);

    let binary = resolve_llama_server(&app)
        .ok_or_else(|| {
            "llama-server binary not found. Place llama-server-{TARGET_TRIPLE} in src-tauri/binaries/ \
             (see https://github.com/ggerganov/llama.cpp/releases for pre-built binaries)."
                .to_string()
        })?;

    let model = PathBuf::from(&model_path);
    let mmproj = PathBuf::from(&mmproj_path);

    if !model.exists() {
        return Err(format!("Model file not found: {model_path}"));
    }
    if !mmproj.exists() {
        return Err(format!("mmproj file not found: {mmproj_path}"));
    }

    let layers = gpu_layers.unwrap_or(0);
    let model_id = model
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "medical-model".to_string());

    // Build the list of directories where dylibs/DLLs live.
    // macOS: binary parent (dev) + resource_dir (production bundle).
    // Windows: handled via current_dir inside spawn_llama_server.
    let mut dylib_dirs: Vec<PathBuf> = Vec::new();
    #[cfg(target_os = "macos")]
    {
        if let Some(parent) = binary.parent() {
            dylib_dirs.push(parent.to_path_buf());
        }
        if let Ok(rd) = app.path().resource_dir() {
            dylib_dirs.push(rd);
        }
    }

    let child = spawn_llama_server(&binary, &model, &mmproj, layers, &dylib_dirs)?;
    *state.llama_child.lock().unwrap() = Some(child);

    // Background thread signals the frontend when the server is up
    wait_for_server(&app, model_id);

    Ok(StartResult { starting: true, port: SERVER_PORT })
}

#[tauri::command]
fn stop_llama_server(state: tauri::State<AppState>) -> Result<(), String> {
    kill_existing(&state);
    Ok(())
}

#[derive(serde::Serialize)]
struct ServerStatus {
    running: bool,
    port: u16,
}

#[tauri::command]
fn get_server_status(state: tauri::State<AppState>) -> ServerStatus {
    let running = state
        .llama_child
        .lock()
        .map(|g| g.is_some())
        .unwrap_or(false);
    ServerStatus { running, port: SERVER_PORT }
}

// ── Entry point ─────────────────────────────────────────────────────────────

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState {
            llama_child: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![
            start_llama_server,
            stop_llama_server,
            get_server_status,
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                // Kill the sidecar when the window closes
                if let Some(state) = window.try_state::<AppState>() {
                    kill_existing(&state);
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running Radiology Desktop");
}
