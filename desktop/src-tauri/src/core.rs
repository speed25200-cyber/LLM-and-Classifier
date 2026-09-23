//! Superviseur du coeur Python : lancement, lecture de `PROPHET_READY`, verification de `/api/health`,
//! journal circulaire, evenements vers l'interface, redemarrage (une fois automatiquement), arret.
//!
//! Evenements emis :
//! * `core://status` : `CoreInfo` complet a chaque changement d'etat ;
//! * `core://log` : `{ "line": "..." }` pour chaque ligne de stdout / stderr du coeur.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::process::{Child, ChildStderr, ChildStdout, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

use rand::distr::Alphanumeric;
use rand::Rng;
use serde::Serialize;
use tauri::{AppHandle, Emitter};

use crate::launch;
use crate::process;

const MAX_LOG_LINES: usize = 200;
const READY_PREFIX: &str = "PROPHET_READY ";
/// Delai laisse au coeur pour s'arreter proprement (il arrete aussi les llama-server).
const STOP_GRACE: Duration = Duration::from_secs(8);
/// `PROPHET_READY` est imprime juste avant que uvicorn n'ouvre le port : on attend `/api/health`.
const HEALTH_TIMEOUT: Duration = Duration::from_secs(60);
const POLL: Duration = Duration::from_millis(250);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Starting,
    Ready,
    Error,
}

/// Charge utile de `core_info` et de `core://status`.
#[derive(Clone, Debug, Serialize)]
pub struct CoreInfo {
    pub url: Option<String>,
    pub token: String,
    pub status: Status,
    pub error: Option<String>,
    pub logs: Vec<String>,
}

#[derive(Clone, Serialize)]
struct LogLine<'a> {
    line: &'a str,
}

struct State {
    status: Status,
    url: Option<String>,
    error: Option<String>,
    logs: VecDeque<String>,
    /// Incremente a chaque lancement : les fils d'un ancien processus se reconnaissent et s'effacent.
    generation: u64,
    /// Le redemarrage automatique apres un plantage n'a lieu qu'une fois (remis a zero par un redemarrage manuel).
    auto_restarted: bool,
    shutting_down: bool,
}

struct Running {
    generation: u64,
    child: Child,
    tree: process::Tree,
}

pub struct Core {
    app: AppHandle,
    token: String,
    state: Mutex<State>,
    running: Mutex<Option<Running>>,
    /// Serialise les lancements / arrets (redemarrages concurrents depuis la barre d'etat et l'interface).
    lifecycle: Mutex<()>,
    logfile: Mutex<Option<File>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Jeton de session : 48 caracteres alphanumeriques (sans ambiguite dans une URL), RNG cryptographique.
fn new_token() -> String {
    rand::rng().sample_iter(&Alphanumeric).take(48).map(char::from).collect()
}

impl Core {
    pub fn new(app: AppHandle) -> Arc<Self> {
        let logfile = {
            let dir = launch::data_dir(&app).join("logs");
            std::fs::create_dir_all(&dir).ok().and_then(|_| File::create(dir.join("desktop-core.log")).ok())
        };
        Arc::new(Core {
            app,
            token: new_token(),
            state: Mutex::new(State {
                status: Status::Starting,
                url: None,
                error: None,
                logs: VecDeque::with_capacity(MAX_LOG_LINES),
                generation: 0,
                auto_restarted: false,
                shutting_down: false,
            }),
            running: Mutex::new(None),
            lifecycle: Mutex::new(()),
            logfile: Mutex::new(logfile),
        })
    }

    pub fn info(&self) -> CoreInfo {
        let st = lock(&self.state);
        CoreInfo {
            url: st.url.clone(),
            token: self.token.clone(),
            status: st.status,
            error: st.error.clone(),
            logs: st.logs.iter().cloned().collect(),
        }
    }

    /// Premier lancement (en arriere-plan : la copie du coeur embarque et uv peuvent prendre du temps).
    pub fn start(self: &Arc<Self>) {
        let core = Arc::clone(self);
        thread::spawn(move || {
            let _guard = lock(&core.lifecycle);
            core.launch();
        });
    }

    /// Redemarrage demande par l'utilisateur (interface ou barre d'etat).
    pub fn restart(self: &Arc<Self>) {
        let core = Arc::clone(self);
        thread::spawn(move || {
            let _guard = lock(&core.lifecycle);
            if lock(&core.state).shutting_down {
                return;
            }
            core.log("[bureau] redemarrage du moteur demande");
            core.stop_process();
            lock(&core.state).auto_restarted = false;
            core.launch();
        });
    }

    /// Arret a la sortie de l'application (bloquant, borne par STOP_GRACE). Le drapeau est pose avant de
    /// prendre `lifecycle` : un lancement en attente abandonne, un lancement en cours finit puis est arrete.
    pub fn shutdown(&self) {
        lock(&self.state).shutting_down = true;
        let _guard = lock(&self.lifecycle);
        self.stop_process();
    }

    // ---- lancement --------------------------------------------------------------------------------------------

    fn launch(self: &Arc<Self>) {
        let generation = {
            let mut st = lock(&self.state);
            if st.shutting_down {
                return;
            }
            st.generation += 1;
            st.status = Status::Starting;
            st.url = None;
            st.error = None;
            st.generation
        };
        self.emit_status();

        let plan = match launch::resolve(&self.app, &self.token) {
            Ok(plan) => plan,
            Err(msg) => return self.fail(generation, msg),
        };
        self.log(&format!("[bureau] lancement du moteur : {}", plan.source));

        let mut cmd = Command::new(&plan.program);
        for (key, value) in launch::appimage_env_fixes() {
            match value {
                Some(value) => cmd.env(key, value),
                None => cmd.env_remove(key),
            };
        }
        if plan.isolate_python {
            cmd.env_remove("PYTHONHOME").env_remove("PYTHONPATH");
        }
        cmd.args(&plan.args)
            .envs(plan.env.iter().map(|(k, v)| (k, v)))
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONUTF8", "1")
            .env("PYTHONIOENCODING", "utf-8")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(dir) = &plan.cwd {
            let _ = std::fs::create_dir_all(dir);
            cmd.current_dir(dir);
        }
        process::configure(&mut cmd);

        let mut child = match cmd.spawn() {
            Ok(child) => child,
            Err(e) => return self.fail(generation, format!("Impossible de lancer {} : {e}", plan.program.display())),
        };
        let tree = process::Tree::attach(&child);
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        *lock(&self.running) = Some(Running { generation, child, tree });

        if let Some(out) = stdout {
            let core = Arc::clone(self);
            thread::spawn(move || core.read_stdout(out, generation));
        }
        if let Some(err) = stderr {
            let core = Arc::clone(self);
            thread::spawn(move || core.read_stderr(err));
        }
        let core = Arc::clone(self);
        thread::spawn(move || core.watch(generation));
    }

    fn read_stdout(self: Arc<Self>, out: ChildStdout, generation: u64) {
        for line in lines(out) {
            self.log(&line);
            if let Some(json) = line.strip_prefix(READY_PREFIX) {
                match serde_json::from_str::<serde_json::Value>(json.trim()) {
                    Ok(v) => match v.get("url").and_then(|u| u.as_str()) {
                        Some(url) => {
                            let (core, url) = (Arc::clone(&self), url.to_owned());
                            thread::spawn(move || core.await_health(generation, url));
                        }
                        None => self.log("[bureau] PROPHET_READY sans url"),
                    },
                    Err(e) => self.log(&format!("[bureau] PROPHET_READY illisible : {e}")),
                }
            }
        }
    }

    fn read_stderr(self: Arc<Self>, err: ChildStderr) {
        for line in lines(err) {
            self.log(&line);
        }
    }

    /// Attend que le serveur reponde reellement avant d'annoncer `ready` a l'interface.
    fn await_health(self: Arc<Self>, generation: u64, url: String) {
        let deadline = Instant::now() + HEALTH_TIMEOUT;
        while Instant::now() < deadline {
            {
                // abandon si relance, arret, ou processus deja mort (on_exit a signale l'erreur)
                let st = lock(&self.state);
                if st.generation != generation || st.shutting_down || st.status != Status::Starting {
                    return;
                }
            }
            if health_ok(&url) {
                {
                    let mut st = lock(&self.state);
                    if st.generation != generation || st.shutting_down {
                        return;
                    }
                    st.status = Status::Ready;
                    st.url = Some(url.clone());
                    st.error = None;
                }
                self.log(&format!("[bureau] moteur pret : {url}"));
                self.emit_status();
                return;
            }
            thread::sleep(POLL);
        }
        self.fail(generation, format!("Le moteur a annonce {url} mais ne repond pas a /api/health."));
    }

    /// Surveille la fin du processus (sondage : pas de course avec `stop_process` sur le pid).
    fn watch(self: Arc<Self>, generation: u64) {
        loop {
            thread::sleep(POLL);
            let status = {
                let mut running = lock(&self.running);
                let Some(r) = running.as_mut().filter(|r| r.generation == generation) else {
                    return; // arrete volontairement ou remplace par un autre lancement
                };
                match r.child.try_wait() {
                    Ok(None) => continue,
                    Ok(Some(status)) => {
                        running.take(); // libere l'arbre (Windows : le Job tue les restes)
                        Some(status)
                    }
                    Err(_) => {
                        running.take();
                        None
                    }
                }
            };
            return self.on_exit(generation, status);
        }
    }

    fn on_exit(self: &Arc<Self>, generation: u64, status: Option<ExitStatus>) {
        let code = status.map_or_else(|| "inconnu".to_owned(), describe_exit);
        let restart = {
            let mut st = lock(&self.state);
            if st.generation != generation || st.shutting_down {
                return;
            }
            let restart = st.status == Status::Ready && !st.auto_restarted;
            if restart {
                st.auto_restarted = true;
            }
            restart
        };
        if restart {
            self.log(&format!("[bureau] le moteur s'est arrete ({code}) : redemarrage automatique"));
            let core = Arc::clone(self);
            thread::spawn(move || {
                let _guard = lock(&core.lifecycle);
                if core.is_current(generation) {
                    core.launch();
                }
            });
        } else {
            self.fail(generation, format!("Le moteur s'est arrete ({code}). Consultez le journal ci-dessous."));
        }
    }

    /// Arrete le processus courant : demande polie, puis arret force de tout l'arbre.
    fn stop_process(&self) {
        let Some(mut r) = lock(&self.running).take() else {
            return;
        };
        r.tree.terminate(&mut r.child);
        let deadline = Instant::now() + STOP_GRACE;
        while Instant::now() < deadline {
            if !matches!(r.child.try_wait(), Ok(None)) {
                break;
            }
            thread::sleep(Duration::from_millis(100));
        }
        r.tree.kill(&mut r.child);
        let _ = r.child.wait();
        self.log("[bureau] moteur arrete");
    }

    // ---- etat, journal, evenements -----------------------------------------------------------------------------

    fn is_current(&self, generation: u64) -> bool {
        let st = lock(&self.state);
        st.generation == generation && !st.shutting_down
    }

    fn fail(&self, generation: u64, msg: String) {
        {
            let mut st = lock(&self.state);
            if st.generation != generation || st.shutting_down {
                return;
            }
            st.status = Status::Error;
            st.url = None;
            st.error = Some(msg.clone());
        }
        self.log(&format!("[bureau] erreur : {msg}"));
        self.emit_status();
    }

    pub fn log(&self, line: &str) {
        {
            let mut st = lock(&self.state);
            if st.logs.len() == MAX_LOG_LINES {
                st.logs.pop_front();
            }
            st.logs.push_back(line.to_owned());
        }
        if let Some(f) = lock(&self.logfile).as_mut() {
            let _ = writeln!(f, "{line}");
        }
        if cfg!(debug_assertions) {
            let _ = writeln!(std::io::stderr(), "[core] {line}");
        }
        let _ = self.app.emit("core://log", LogLine { line });
    }

    fn emit_status(&self) {
        let _ = self.app.emit("core://status", self.info());
    }
}

/// Lignes d'un flux, decodees sans echouer (UTF-8 avec remplacement, fin de ligne CRLF toleree).
fn lines(stream: impl Read) -> impl Iterator<Item = String> {
    let mut reader = BufReader::new(stream);
    let mut buf = Vec::new();
    std::iter::from_fn(move || {
        buf.clear();
        match reader.read_until(b'\n', &mut buf) {
            Ok(0) | Err(_) => None,
            Ok(_) => {
                while buf.last().is_some_and(|b| *b == b'\n' || *b == b'\r') {
                    buf.pop();
                }
                Some(String::from_utf8_lossy(&buf).into_owned())
            }
        }
    })
}

/// GET /api/health en HTTP/1.1 minimal (le coeur n'ecoute que sur la boucle locale).
fn health_ok(url: &str) -> bool {
    let Some(authority) = url.strip_prefix("http://").map(|r| r.split('/').next().unwrap_or(r)) else {
        return false;
    };
    let Ok(addr) = authority.parse::<SocketAddr>() else {
        return false;
    };
    let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_secs(1)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(3)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(3)));
    let request = format!("GET /api/health HTTP/1.1\r\nHost: {authority}\r\nConnection: close\r\n\r\n");
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = String::new();
    let _ = stream.take(64 * 1024).read_to_string(&mut response);
    response.starts_with("HTTP/1.1 200") && response.contains("\"prophet-studio\"")
}

fn describe_exit(status: ExitStatus) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = status.signal() {
            return format!("signal {sig}");
        }
    }
    status.code().map_or_else(|| "code inconnu".to_owned(), |c| format!("code {c}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_is_long_and_url_safe() {
        let t = new_token();
        assert_eq!(t.len(), 48);
        assert!(t.chars().all(|c| c.is_ascii_alphanumeric()));
        assert_ne!(t, new_token());
    }

    #[test]
    fn lines_strip_crlf_and_survive_bad_utf8() {
        let data: &[u8] = b"PROPHET_READY {\"url\": \"http://127.0.0.1:1\"}\r\nd\xe9j\xe0\nfin";
        let got: Vec<String> = lines(data).collect();
        assert_eq!(got[0], "PROPHET_READY {\"url\": \"http://127.0.0.1:1\"}");
        assert!(got[1].starts_with('d'));
        assert_eq!(got[2], "fin");
    }

    #[test]
    fn health_rejects_bad_urls() {
        assert!(!health_ok("ftp://127.0.0.1:1"));
        assert!(!health_ok("http://pas-une-adresse"));
    }

    #[test]
    fn health_accepts_prophet_server() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let mut buf = [0u8; 1024];
            let _ = s.read(&mut buf);
            let body = r#"{"ok": true, "app": "prophet-studio"}"#;
            let _ = write!(s, "HTTP/1.1 200 OK\r\ncontent-length: {}\r\n\r\n{body}", body.len());
        });
        assert!(health_ok(&format!("http://127.0.0.1:{port}")));
        server.join().unwrap();
    }
}
