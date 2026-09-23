//! Ou vit le coeur Python et comment le lancer.
//!
//! Ordre de resolution de la commande (la premiere qui s'applique gagne) :
//! 1. `PROPHET_CORE_CMD` : ligne de commande complete (developpement), decoupee sur les espaces ;
//! 2. `uv` embarque (sidecar a cote de l'executable) + coeur embarque (ressources `core/`) ;
//! 3. `uv` du PATH + depot source (on remonte depuis l'executable ou le dossier courant jusqu'au
//!    `pyproject.toml` qui mentionne `prophet_studio`) ; dans une compilation de developpement, cette
//!    etape passe avant la 2 pour que les modifications du code Python s'appliquent sans recompiler ;
//! 4. `prophet-studio` du PATH (ou de l'environnement cree par les scripts d'installation).
//!
//! La coquille ajoute toujours `--no-browser --port 0 --token <jeton> --parent-pid <pid>`.

use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use tauri::{AppHandle, Manager};

/// Version de Python imposee a uv (celle testee par le coeur).
const PYTHON_VERSION: &str = "3.11";
/// Un vrai binaire uv pese des dizaines de Mo ; en dessous, c'est l'espace reserve vide de la CI.
const MIN_SIDECAR_BYTES: u64 = 256 * 1024;
const STAMP_FILE: &str = ".bundle-stamp";

/// Commande prete a lancer.
pub struct LaunchPlan {
    pub program: PathBuf,
    pub args: Vec<OsString>,
    pub env: Vec<(&'static str, OsString)>,
    pub cwd: Option<PathBuf>,
    /// Modes uv : PYTHONHOME / PYTHONPATH herites casseraient le Python gere par uv.
    pub isolate_python: bool,
    /// Resume lisible pour le journal.
    pub source: String,
}

/// Dossier de donnees, identique a `prophet_studio.config.data_dir()`.
pub fn data_dir(app: &AppHandle) -> PathBuf {
    if let Some(home) = std::env::var_os("PROPHET_HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(home);
    }
    let name = if cfg!(target_os = "linux") { "prophet-studio" } else { "ProphetStudio" };
    // %LOCALAPPDATA% (Windows), ~/Library/Application Support (macOS), $XDG_DATA_HOME ou ~/.local/share (Linux)
    let base = app
        .path()
        .local_data_dir()
        .or_else(|_| app.path().home_dir().map(|h| h.join(".local").join("share")))
        .unwrap_or_else(|_| std::env::temp_dir());
    base.join(name)
}

/// `vram_saver` de settings.json (vrai si absent ou illisible : la VRAM va au modele par defaut).
pub fn vram_saver(data_dir: &Path) -> bool {
    fs::read_to_string(data_dir.join("settings.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| v.get("vram_saver").and_then(serde_json::Value::as_bool))
        .unwrap_or(true)
}

/// Determine la commande du coeur. `Err` = message lisible pour l'interface.
pub fn resolve(app: &AppHandle, token: &str) -> Result<LaunchPlan, String> {
    let mut core_args: Vec<OsString> =
        ["--no-browser", "--port", "0", "--token", token, "--parent-pid"].iter().map(OsString::from).collect();
    core_args.push(std::process::id().to_string().into());
    if cfg!(dev) {
        // l'interface est servie par Vite (localhost:5173) : le coeur doit accepter cette origine
        core_args.push("--dev".into());
    }
    if let Ok(extra) = std::env::var("PROPHET_CORE_ARGS") {
        core_args.extend(extra.split_whitespace().map(OsString::from)); // ex. "--demo"
    }
    let data = data_dir(app);

    // 1. commande imposee (developpement), lancee depuis la racine du depot si on la trouve :
    //    `python -m prophet_studio` marche alors sans installation du paquet
    if let Ok(cmd) = std::env::var("PROPHET_CORE_CMD") {
        let mut parts = cmd.split_whitespace();
        if let Some(program) = parts.next() {
            let mut args: Vec<OsString> = parts.map(OsString::from).collect();
            args.extend(core_args);
            return Ok(LaunchPlan {
                program: program.into(),
                args,
                env: Vec::new(),
                cwd: find_source_root(),
                isolate_python: false,
                source: format!("PROPHET_CORE_CMD ({cmd})"),
            });
        }
    }

    // En developpement, le depot source prime sur le coeur embarque (copie figee a la compilation) :
    // une modification du code Python s'applique au prochain lancement.
    let source_root = find_source_root();
    let prefer_source = cfg!(dev) && source_root.is_some();

    // 2. uv embarque + coeur embarque, recopie dans le dossier de donnees
    if let (false, Some(uv), Ok(resources)) = (prefer_source, bundled_uv(), app.path().resource_dir()) {
        let bundled = resources.join("core");
        if bundled.join("pyproject.toml").is_file() {
            let project = data.join("core");
            sync_bundled_core(&bundled, &project, &app.package_info().version.to_string())
                .map_err(|e| format!("Copie du coeur embarque vers {} impossible : {e}", project.display()))?;
            let env = vec![
                ("UV_PROJECT_ENVIRONMENT", data.join("venv").into_os_string()),
                ("UV_PYTHON_INSTALL_DIR", data.join("python").into_os_string()),
                ("UV_CACHE_DIR", data.join("uv-cache").into_os_string()),
                // Python autonome telecharge par uv : pas de dependance a un Python systeme
                ("UV_PYTHON_PREFERENCE", "only-managed".into()),
            ];
            return Ok(LaunchPlan {
                args: uv_args(&project, core_args),
                source: format!("uv embarque ({}) + coeur {}", uv.display(), project.display()),
                program: uv,
                env,
                cwd: Some(data),
                isolate_python: true,
            });
        }
    }

    // 3. uv du PATH (a defaut, l'embarque) + depot source
    if let (Some(uv), Some(root)) = (find_program("uv").or_else(bundled_uv), source_root) {
        return Ok(LaunchPlan {
            args: uv_args(&root, core_args),
            source: format!("uv ({}) + depot source {}", uv.display(), root.display()),
            program: uv,
            env: Vec::new(),
            cwd: Some(root),
            isolate_python: true,
        });
    }

    // 4. prophet-studio installe (PATH, ou environnement des scripts d'installation)
    let installed = ["app-venv/Scripts/prophet-studio.exe", "app-venv/bin/prophet-studio"]
        .iter()
        .map(|p| data.join(p))
        .find(|p| p.is_file());
    if let Some(exe) = find_program("prophet-studio").or(installed) {
        return Ok(LaunchPlan {
            source: format!("prophet-studio ({})", exe.display()),
            program: exe,
            args: core_args,
            env: Vec::new(),
            cwd: None,
            isolate_python: true,
        });
    }

    Err("Moteur introuvable : ni uv embarque, ni uv avec le depot source, ni prophet-studio. \
         Installez uv (https://docs.astral.sh/uv/) ou reinstallez Prophet Studio."
        .into())
}

fn uv_args(project: &Path, core_args: Vec<OsString>) -> Vec<OsString> {
    let mut args: Vec<OsString> = vec!["run".into(), "--project".into(), project.as_os_str().to_owned()];
    args.extend(
        ["--extra", "studio", "--python", PYTHON_VERSION, "python", "-m", "prophet_studio"].map(OsString::from),
    );
    args.extend(core_args);
    args
}

fn exe_name(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_owned()
    }
}

/// Sidecar uv, nomme `prophet-uv` : Tauri le depose a cote de l'executable, sans suffixe de cible
/// (Contents/MacOS sur macOS, /usr/bin pour le .deb, d'ou un nom qui n'entre pas en conflit avec un uv systeme).
fn bundled_uv() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let uv = exe.parent()?.join(exe_name("prophet-uv"));
    let len = fs::metadata(&uv).ok()?.len();
    (len >= MIN_SIDECAR_BYTES).then_some(uv)
}

/// Cherche un programme dans le PATH, puis dans les emplacements usuels : une application lancee
/// depuis le Finder ou le menu Demarrer n'herite pas toujours du PATH du terminal.
fn find_program(name: &str) -> Option<PathBuf> {
    let file = exe_name(name);
    let mut dirs: Vec<PathBuf> =
        std::env::var_os("PATH").map(|p| std::env::split_paths(&p).collect()).unwrap_or_default();
    if let Some(home) = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }) {
        let home = PathBuf::from(home);
        dirs.push(home.join(".local").join("bin"));
        dirs.push(home.join(".cargo").join("bin"));
    }
    if cfg!(unix) {
        dirs.extend(["/opt/homebrew/bin", "/usr/local/bin"].map(PathBuf::from));
    }
    dirs.into_iter().map(|d| d.join(&file)).find(|p| p.is_file())
}

/// Variables a corriger pour le coeur (`None` = a retirer). Dans une AppImage, AppRun fait pointer
/// PYTHONHOME, PYTHONPATH, LD_LIBRARY_PATH, PATH, XDG_DATA_DIRS... vers l'image montee : herites par
/// uv, Python et llama-server, ils les cassent. On retire toutes les entrees situees sous $APPDIR.
pub fn appimage_env_fixes() -> Vec<(OsString, Option<OsString>)> {
    match std::env::var_os("APPDIR").filter(|v| !v.is_empty()) {
        Some(appdir) => strip_dir_from_env(Path::new(&appdir), std::env::vars_os()),
        None => Vec::new(),
    }
}

fn strip_dir_from_env(
    dir: &Path,
    vars: impl Iterator<Item = (OsString, OsString)>,
) -> Vec<(OsString, Option<OsString>)> {
    vars.filter_map(|(key, value)| {
        let entries: Vec<PathBuf> = std::env::split_paths(&value).collect();
        if !entries.iter().any(|p| p.starts_with(dir)) {
            return None;
        }
        let kept: Vec<PathBuf> =
            entries.into_iter().filter(|p| !p.as_os_str().is_empty() && !p.starts_with(dir)).collect();
        let value = if kept.is_empty() { None } else { std::env::join_paths(kept).ok() };
        Some((key, value))
    })
    .collect()
}

/// Depot source : on remonte depuis l'executable puis le dossier courant.
fn find_source_root() -> Option<PathBuf> {
    let starts =
        [std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)), std::env::current_dir().ok()];
    starts.into_iter().flatten().find_map(|start| {
        start.ancestors().find_map(|dir| {
            let manifest = fs::read_to_string(dir.join("pyproject.toml")).ok()?;
            (manifest.contains("prophet_studio") && dir.join("prophet_studio").is_dir()).then(|| dir.to_path_buf())
        })
    })
}

/// Recopie le coeur embarque dans un dossier inscriptible : le dossier des ressources peut etre en
/// lecture seule (AppImage, Program Files, .app signee) alors que uv y ecrirait uv.lock et *.egg-info.
/// Une empreinte du contenu evite de recopier a chaque lancement.
fn sync_bundled_core(src: &Path, dst: &Path, version: &str) -> io::Result<()> {
    let mut files = Vec::new();
    list_files(src, src, &mut files)?;
    files.sort();
    let mut hash = Fnv::new();
    hash.write(version.as_bytes());
    for rel in &files {
        hash.write(rel.to_string_lossy().as_bytes());
        hash.write(&fs::read(src.join(rel))?);
    }
    let stamp = format!("{version} {:016x}", hash.0);
    if fs::read_to_string(dst.join(STAMP_FILE)).is_ok_and(|s| s.trim() == stamp) {
        return Ok(());
    }
    match fs::remove_dir_all(dst) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e),
        _ => {}
    }
    for rel in &files {
        let target = dst.join(rel);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(src.join(rel), &target)?;
    }
    fs::write(dst.join(STAMP_FILE), stamp)
}

fn list_files(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        if name == "__pycache__" || path.extension().is_some_and(|e| e == "pyc") {
            continue;
        }
        if entry.file_type()?.is_dir() {
            list_files(root, &path, out)?;
        } else if let Ok(rel) = path.strip_prefix(root) {
            out.push(rel.to_path_buf());
        }
    }
    Ok(())
}

/// FNV-1a 64 bits : empreinte stable d'une version a l'autre de Rust (contrairement a DefaultHasher).
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 ^= u64::from(*b);
            self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
        }
        // separateur : ("ab","c") et ("a","bc") ne donnent pas la meme empreinte
        self.0 ^= 0xff;
        self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv_is_stable_and_separates_fields() {
        let mut a = Fnv::new();
        a.write(b"ab");
        a.write(b"c");
        let mut b = Fnv::new();
        b.write(b"a");
        b.write(b"bc");
        assert_ne!(a.0, b.0);
        let mut c = Fnv::new();
        c.write(b"ab");
        c.write(b"c");
        assert_eq!(a.0, c.0);
    }

    #[test]
    fn sync_copies_once_and_skips_bytecode() {
        let base = std::env::temp_dir().join(format!("prophet-sync-test-{}", std::process::id()));
        let (src, dst) = (base.join("src"), base.join("dst"));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(src.join("pkg/__pycache__")).unwrap();
        fs::write(src.join("pyproject.toml"), "[project]\nname='x'\n").unwrap();
        fs::write(src.join("pkg/mod.py"), "x = 1\n").unwrap();
        fs::write(src.join("pkg/__pycache__/mod.cpython-311.pyc"), "junk").unwrap();

        sync_bundled_core(&src, &dst, "1.0.0").unwrap();
        assert!(dst.join("pkg/mod.py").is_file());
        assert!(!dst.join("pkg/__pycache__").exists());

        // un fichier ecrit par uv (uv.lock) survit tant que le coeur embarque ne change pas
        fs::write(dst.join("uv.lock"), "lock").unwrap();
        sync_bundled_core(&src, &dst, "1.0.0").unwrap();
        assert!(dst.join("uv.lock").is_file());

        // nouvelle version du coeur : recopie propre
        fs::write(src.join("pkg/mod.py"), "x = 2\n").unwrap();
        sync_bundled_core(&src, &dst, "1.0.0").unwrap();
        assert_eq!(fs::read_to_string(dst.join("pkg/mod.py")).unwrap(), "x = 2\n");
        assert!(!dst.join("uv.lock").exists());
        let _ = fs::remove_dir_all(&base);
    }

    #[cfg(unix)]
    #[test]
    fn appimage_entries_are_stripped() {
        let vars = [
            ("PYTHONHOME", "/tmp/.mount_ps/usr/"),
            ("PATH", "/tmp/.mount_ps/usr/bin:/usr/local/bin:/usr/bin"),
            ("LD_LIBRARY_PATH", "/tmp/.mount_ps/usr/lib:"),
            ("HOME", "/home/moi"),
        ]
        .map(|(k, v)| (OsString::from(k), OsString::from(v)));
        let mut fixes = strip_dir_from_env(Path::new("/tmp/.mount_ps"), vars.into_iter());
        fixes.sort();
        assert_eq!(
            fixes,
            vec![
                (OsString::from("LD_LIBRARY_PATH"), None),
                (OsString::from("PATH"), Some(OsString::from("/usr/local/bin:/usr/bin"))),
                (OsString::from("PYTHONHOME"), None),
            ]
        );
    }

    #[test]
    fn vram_saver_defaults_to_true() {
        let dir = std::env::temp_dir().join(format!("prophet-vram-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let _ = fs::remove_file(dir.join("settings.json"));
        assert!(vram_saver(&dir));
        fs::write(dir.join("settings.json"), r#"{"vram_saver": false}"#).unwrap();
        assert!(!vram_saver(&dir));
        fs::write(dir.join("settings.json"), "pas du json").unwrap();
        assert!(vram_saver(&dir));
        let _ = fs::remove_dir_all(&dir);
    }
}
