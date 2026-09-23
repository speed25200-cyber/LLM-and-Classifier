//! Fenetre principale, creee en Rust (et non dans tauri.conf.json) pour choisir les options du
//! moteur web selon `vram_saver` avant sa creation.

use tauri::webview::NewWindowResponse;
use tauri::window::Color;
use tauri::{AppHandle, Manager, Runtime, Url, WebviewUrl, WebviewWindow, WebviewWindowBuilder};

pub const MAIN: &str = "main";
const WIDTH: f64 = 1440.0;
const HEIGHT: f64 = 900.0;
/// #07080B, le fond de l'interface : pas d'eclair blanc au demarrage.
const BACKGROUND: Color = Color(7, 8, 11, 255);
/// Options par defaut de wry pour WebView2, a conserver quand on en ajoute (voir wry/src/webview2/mod.rs).
#[cfg(windows)]
const WEBVIEW2_DEFAULT_ARGS: &str = "--disable-features=msWebOOUI,msPdfOOUI,msSmartScreenProtection";

/// Variables d'environnement a poser avant l'initialisation de GTK / WebKitGTK (Linux uniquement).
/// Tauri n'existe pas encore : le dossier de donnees est deduit comme le fait le coeur.
pub fn prepare_environment() {
    #[cfg(target_os = "linux")]
    {
        use std::path::PathBuf;
        let data_dir = std::env::var_os("PROPHET_HOME").filter(|v| !v.is_empty()).map(PathBuf::from).or_else(|| {
            let share = std::env::var_os("XDG_DATA_HOME")
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
                .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("share")))?;
            Some(share.join("prophet-studio"))
        });
        let vram_saver = data_dir.is_none_or(|d| crate::launch::vram_saver(&d));
        let set_default = |key: &str, value: &str| {
            if std::env::var_os(key).is_none() {
                // Appele au tout debut de `run`, avant la creation de tout autre fil.
                std::env::set_var(key, value);
            }
        };
        // Pilote NVIDIA proprietaire : le rendu DMA-BUF de WebKitGTK donne souvent une fenetre blanche.
        if std::path::Path::new("/proc/driver/nvidia/version").exists() {
            set_default("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
        }
        // Economiseur de VRAM : composition logicielle, la carte reste au modele.
        if vram_saver {
            set_default("WEBKIT_DISABLE_COMPOSITING_MODE", "1");
        }
    }
}

pub fn create_main<R: Runtime, M: Manager<R>>(manager: &M, vram_saver: bool) -> tauri::Result<WebviewWindow<R>> {
    let dev_url = manager.config().build.dev_url.clone();

    let builder = WebviewWindowBuilder::new(manager, MAIN, WebviewUrl::default())
        .title("Prophet Studio")
        .inner_size(WIDTH, HEIGHT)
        .min_inner_size(960.0, 640.0)
        .center()
        .visible(false)
        .background_color(BACKGROUND)
        // Liens externes : navigateur du systeme, jamais dans la fenetre de l'application.
        .on_navigation(move |url| {
            let allowed = is_app_url(url, dev_url.as_ref());
            if !allowed {
                open_external(url);
            }
            allowed
        })
        .on_new_window(move |url, _features| {
            open_external(&url);
            NewWindowResponse::Deny
        });

    // Windows / Linux : pas de cadre systeme, l'interface dessine sa barre de titre (data-tauri-drag-region).
    #[cfg(not(target_os = "macos"))]
    let builder = builder.decorations(false);
    // macOS : barre de titre transparente, les feux tricolores restent en surimpression.
    #[cfg(target_os = "macos")]
    let builder = builder.title_bar_style(tauri::TitleBarStyle::Overlay).hidden_title(true);

    // Windows : economiseur de VRAM = WebView2 sans GPU (rendu et composition logiciels).
    #[cfg(windows)]
    let builder = if vram_saver {
        builder.additional_browser_args(&format!("{WEBVIEW2_DEFAULT_ARGS} --disable-gpu --disable-gpu-compositing"))
    } else {
        builder
    };
    #[cfg(not(windows))]
    let _ = vram_saver;

    let window = builder.build()?;
    fit_to_screen(&window);
    #[cfg(target_os = "linux")]
    enable_linux_microphone(&window);
    window.show()?;
    let _ = window.set_focus();
    Ok(window)
}

/// Sur un petit ecran (portable 1366x768), 1440x900 deborderait : on maximise.
fn fit_to_screen<R: Runtime>(window: &WebviewWindow<R>) {
    if let Ok(Some(monitor)) = window.current_monitor() {
        let size = monitor.size().to_logical::<f64>(monitor.scale_factor());
        if size.width < WIDTH + 40.0 || size.height < HEIGHT + 80.0 {
            let _ = window.maximize();
        }
    }
}

/// Montre, restaure et donne le focus a la fenetre principale.
pub fn reveal<R: Runtime>(app: &AppHandle<R>) {
    if let Some(w) = app.get_webview_window(MAIN) {
        let _ = w.show();
        let _ = w.unminimize();
        let _ = w.set_focus();
    }
}

/// Afficher / Masquer (barre d'etat).
pub fn toggle<R: Runtime>(app: &AppHandle<R>) {
    if let Some(w) = app.get_webview_window(MAIN) {
        if w.is_visible().unwrap_or(false) && !w.is_minimized().unwrap_or(false) {
            let _ = w.hide();
        } else {
            reveal(app);
        }
    }
}

/// Pages de l'application : protocole embarque (tauri://localhost, http(s)://tauri.localhost), serveur
/// de developpement, coeur local (apercus en iframe), about:blank.
fn is_app_url(url: &Url, dev_url: Option<&Url>) -> bool {
    let host = url.host_str().unwrap_or_default();
    match url.scheme() {
        "tauri" | "about" | "data" | "blob" => true,
        "http" | "https" => {
            host == "tauri.localhost"
                || matches!(host, "127.0.0.1" | "localhost" | "[::1]")
                || dev_url.is_some_and(|d| d.origin() == url.origin())
        }
        _ => false,
    }
}

fn open_external(url: &Url) {
    if matches!(url.scheme(), "http" | "https" | "mailto") {
        let _ = tauri_plugin_opener::open_url(url.as_str(), None::<&str>);
    }
}

/// WebKitGTK refuse getUserMedia par defaut : on active le flux media et on n'accorde que le micro.
#[cfg(target_os = "linux")]
fn enable_linux_microphone<R: Runtime>(window: &WebviewWindow<R>) {
    let _ = window.with_webview(|webview| {
        use webkit2gtk::glib::prelude::Cast;
        use webkit2gtk::{
            PermissionRequestExt, SettingsExt, UserMediaPermissionRequest, UserMediaPermissionRequestExt, WebViewExt,
        };

        let view = webview.inner();
        if let Some(settings) = WebViewExt::settings(&view) {
            settings.set_enable_media_stream(true);
        }
        view.connect_permission_request(|_, request| match request.downcast_ref::<UserMediaPermissionRequest>() {
            Some(media) if media.is_for_audio_device() && !media.is_for_video_device() => {
                request.allow();
                true
            }
            _ => false, // traitement par defaut de WebKitGTK : refus
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn url(s: &str) -> Url {
        Url::parse(s).unwrap()
    }

    #[test]
    fn app_urls_stay_in_window() {
        let dev = url("http://localhost:5173");
        assert!(is_app_url(&url("tauri://localhost/index.html"), None));
        assert!(is_app_url(&url("http://tauri.localhost/"), None));
        assert!(is_app_url(&url("https://tauri.localhost/settings"), None));
        assert!(is_app_url(&url("http://127.0.0.1:7878/"), None));
        assert!(is_app_url(&url("http://localhost:5173/src/main.ts"), Some(&dev)));
        assert!(!is_app_url(&url("https://huggingface.co/prism-ml"), Some(&dev)));
        assert!(!is_app_url(&url("https://tauri.localhost.evil.com/"), None));
        assert!(!is_app_url(&url("file:///etc/passwd"), None));
    }
}
