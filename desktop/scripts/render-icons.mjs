// Regenere le jeu d'icones a partir des SVG (npm run icons).
//   src-tauri/icons/icon.svg          -> icon-1024.png, 32x32.png, 128x128.png, 128x128@2x.png, icon.png, icon.ico, icon.icns
//   src-tauri/icons/tray-template.svg -> tray-template.png (barre des menus macOS)
// Le rendu SVG est fait par le CLI Tauri lui-meme (resvg) : aucune dependance en plus.
import { execFileSync } from "node:child_process";
import { copyFileSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const icons = join(root, "src-tauri", "icons");
const tauri = (...args) =>
  execFileSync(process.platform === "win32" ? "npx.cmd" : "npx", ["tauri", "icon", ...args], {
    cwd: root,
    stdio: "inherit",
    shell: process.platform === "win32",
  });

const tmp = mkdtempSync(join(tmpdir(), "prophet-icons-"));
try {
  // 1. PNG maitre 1024 px, puis toutes les tailles a partir de lui
  tauri(join(icons, "icon.svg"), "-o", join(tmp, "master"), "-p", "1024");
  const master = join(tmp, "master", "1024x1024.png");
  copyFileSync(master, join(icons, "icon-1024.png"));
  tauri(master, "-o", join(tmp, "all"));
  for (const f of ["32x32.png", "128x128.png", "128x128@2x.png", "icon.png", "icon.ico", "icon.icns"]) {
    copyFileSync(join(tmp, "all", f), join(icons, f));
  }
  // 2. icone "template" monochrome de la barre des menus macOS
  tauri(join(icons, "tray-template.svg"), "-o", join(tmp, "tray"), "-p", "64");
  copyFileSync(join(tmp, "tray", "64x64.png"), join(icons, "tray-template.png"));
  console.log("icones regenerees dans", icons);
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
