//! Prophet Studio, application de bureau : une fenetre sans cadre qui affiche l'interface (ui/dist),
//! un superviseur du coeur Python (core.rs), une icone de barre d'etat et un raccourci global
//! push-to-talk. Fermer la fenetre quitte l'application et arrete le coeur (et donc les modeles) :
//! aucune VRAM ne reste occupee en arriere-plan.
//!
//! Contrat avec l'interface (window.__TAURI__, withGlobalTauri) :
//! * commandes : `core_info` -> CoreInfo, `restart_core` ;
//! * evenements : `core://status` (CoreInfo), `core://log` ({ line }), `shortcut://ptt` ({ state }).

mod core;
mod launch;
mod process;
mod window;

use std::sync::Arc;

use serde::Serialize;
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{App, AppHandle, Emitter, Manager, RunEvent, State};

use crate::core::{Core, CoreInfo};

/// Ctrl+Shift+Espace (Cmd+Shift+Espace sur macOS).
const PTT_SHORTCUT: &str = "CommandOrControl+Shift+Space";

#[tauri::command]
fn core_info(core: State<'_, Arc<Core>>) -> CoreInfo {
    core.info()
}

#[tauri::command]
fn restart_core(core: State<'_, Arc<Core>>) {
    core.restart();
}

pub fn run() {
    window::prepare_environment();

    let app = tauri::Builder::default()
        // doit rester le premier plugin : une seconde instance ne fait que ramener la fenetre existante
        .plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| window::reveal(app)))
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![core_info, restart_core])
        .setup(|app| {
            let core = Core::new(app.handle().clone());
            app.manage(Arc::clone(&core));
            let vram_saver = launch::vram_saver(&launch::data_dir(app.handle()));
            window::create_main(app, vram_saver)?;
            create_tray(app)?;
            register_ptt_shortcut(app.handle());
            core.start();
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("impossible de demarrer Prophet Studio");

    #[cfg(unix)]
    {
        let handle = app.handle().clone();
        process::on_termination_signal(move || handle.exit(0));
    }

    app.run(|app, event| {
        if let RunEvent::Exit = event {
            if let Some(core) = app.try_state::<Arc<Core>>() {
                core.shutdown();
            }
        }
    });
}

fn create_tray(app: &App) -> tauri::Result<()> {
    let toggle = MenuItem::with_id(app, "toggle", "Afficher / Masquer", true, None::<&str>)?;
    let restart = MenuItem::with_id(app, "restart", "Redémarrer le moteur", true, None::<&str>)?;
    let separator = PredefinedMenuItem::separator(app)?;
    let quit = MenuItem::with_id(app, "quit", "Quitter", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&toggle, &restart, &separator, &quit])?;

    let mut tray =
        TrayIconBuilder::with_id("main").tooltip("Prophet Studio").menu(&menu).on_menu_event(|app, event| match event
            .id()
            .as_ref()
        {
            "toggle" => window::toggle(app),
            "restart" => {
                if let Some(core) = app.try_state::<Arc<Core>>() {
                    core.restart();
                }
            }
            "quit" => app.exit(0),
            _ => {}
        });

    // macOS : icone "template" monochrome, recoloree par le systeme (theme clair / sombre).
    #[cfg(target_os = "macos")]
    {
        tray = tray
            .icon(tauri::image::Image::from_bytes(include_bytes!("../icons/tray-template.png"))?)
            .icon_as_template(true);
    }
    #[cfg(not(target_os = "macos"))]
    if let Some(icon) = app.default_window_icon() {
        tray = tray.icon(icon.clone());
    }

    // Windows : clic gauche = afficher / masquer, clic droit = menu (usage de la zone de notification).
    #[cfg(windows)]
    {
        use tauri::tray::{MouseButton, MouseButtonState, TrayIconEvent};
        tray = tray.show_menu_on_left_click(false).on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click { button: MouseButton::Left, button_state: MouseButtonState::Up, .. } = event {
                window::toggle(tray.app_handle());
            }
        });
    }

    tray.build(app)?;
    Ok(())
}

#[derive(Clone, Serialize)]
struct PttEvent {
    state: &'static str,
}

/// Push-to-talk global. Un echec (raccourci pris par une autre application, Wayland sans XWayland)
/// n'empeche pas l'application de demarrer.
fn register_ptt_shortcut(app: &AppHandle) {
    use tauri_plugin_global_shortcut::{Builder, GlobalShortcutExt, ShortcutState};

    let plugin = Builder::new()
        .with_handler(|app, _shortcut, event| {
            let state = match event.state {
                ShortcutState::Pressed => "pressed",
                ShortcutState::Released => "released",
            };
            if cfg!(debug_assertions) {
                eprintln!("[bureau] shortcut://ptt {state}");
            }
            if event.state == ShortcutState::Pressed {
                window::reveal(app);
            }
            let _ = app.emit_to(window::MAIN, "shortcut://ptt", PttEvent { state });
        })
        .build();
    let result = app
        .plugin(plugin)
        .map_err(|e| e.to_string())
        .and_then(|_| app.global_shortcut().register(PTT_SHORTCUT).map_err(|e| e.to_string()));
    if let (Err(e), Some(core)) = (result, app.try_state::<Arc<Core>>()) {
        core.log(&format!("[bureau] raccourci {PTT_SHORTCUT} indisponible : {e}"));
    }
}
