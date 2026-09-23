// Place le binaire uv (sidecar Tauri) dans src-tauri/binaries/prophet-uv-<cible>[.exe] (npm run sidecar).
// Nom `prophet-uv` : le .deb installe les sidecars dans /usr/bin, ou `uv` entrerait en conflit avec un uv systeme.
//
//   node scripts/fetch-uv.mjs                         # cible = hote (rustc -vV), version UV_VERSION ou defaut
//   node scripts/fetch-uv.mjs --target aarch64-apple-darwin
//   node scripts/fetch-uv.mjs --placeholder           # fichier vide : suffit pour `cargo check` / `cargo build`
//   node scripts/fetch-uv.mjs --force                 # retelecharge meme si le binaire est deja la
//
// Source : release officielle astral-sh/uv sur GitHub, somme SHA-256 verifiee. Extraction avec `tar`
// (present sur Linux, macOS et Windows 10+ ; bsdtar de Windows lit aussi les .zip).
// Derriere un proxy HTTP : NODE_USE_ENV_PROXY=1 (Node >= 22.21) pour que fetch lise HTTPS_PROXY.
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { chmodSync, copyFileSync, existsSync, mkdirSync, mkdtempSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const DEFAULT_UV_VERSION = "0.12.18";
const MIN_REAL_BYTES = 256 * 1024; // en dessous : espace reserve (meme seuil que src/launch.rs)

const args = process.argv.slice(2);
const flag = (name) => args.includes(name);
const option = (name) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : undefined;
};

const hostTriple = () => {
  const out = execFileSync("rustc", ["-vV"], { encoding: "utf8" });
  const line = out.split(/\r?\n/).find((l) => l.startsWith("host:"));
  if (!line) throw new Error("rustc -vV : ligne host introuvable");
  return line.slice(5).trim();
};

const version = option("--version") ?? process.env.UV_VERSION ?? DEFAULT_UV_VERSION;
const target = option("--target") ?? process.env.TAURI_TARGET_TRIPLE ?? hostTriple();
const windows = target.includes("windows");
const binDir = join(dirname(fileURLToPath(import.meta.url)), "..", "src-tauri", "binaries");
const dest = join(binDir, `prophet-uv-${target}${windows ? ".exe" : ""}`);
mkdirSync(binDir, { recursive: true });

if (flag("--placeholder")) {
  if (!existsSync(dest)) writeFileSync(dest, "");
  if (!windows) chmodSync(dest, 0o755);
  console.log(`espace reserve : ${dest}`);
  process.exit(0);
}
if (!flag("--force") && existsSync(dest) && statSync(dest).size >= MIN_REAL_BYTES) {
  console.log(`deja present : ${dest} (--force pour retelecharger)`);
  process.exit(0);
}

const asset = `uv-${target}${windows ? ".zip" : ".tar.gz"}`;
const base = `https://github.com/astral-sh/uv/releases/download/${version}/${asset}`;

async function download(url) {
  const res = await fetch(url, { redirect: "follow" });
  if (!res.ok) throw new Error(`${url} : HTTP ${res.status}`);
  return Buffer.from(await res.arrayBuffer());
}

function findFile(dir, name) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name);
    if (entry.isDirectory()) {
      const found = findFile(p, name);
      if (found) return found;
    } else if (entry.name === name) {
      return p;
    }
  }
  return undefined;
}

const tmp = mkdtempSync(join(tmpdir(), "prophet-uv-"));
try {
  console.log(`telechargement de uv ${version} (${target})`);
  const [archive, sums] = await Promise.all([download(base), download(`${base}.sha256`)]);
  const expected = sums.toString("utf8").trim().split(/\s+/)[0].toLowerCase();
  const actual = createHash("sha256").update(archive).digest("hex");
  if (expected !== actual) throw new Error(`somme SHA-256 invalide pour ${asset} : ${actual} != ${expected}`);

  const file = join(tmp, asset);
  writeFileSync(file, archive);
  const tar = process.platform === "win32" ? join(process.env.SystemRoot ?? "C:\\Windows", "System32", "tar.exe") : "tar";
  execFileSync(tar, ["-xf", file, "-C", tmp], { stdio: "inherit" });
  const exe = findFile(tmp, windows ? "uv.exe" : "uv");
  if (!exe) throw new Error(`uv introuvable dans ${asset}`);
  copyFileSync(exe, dest);
  if (!windows) chmodSync(dest, 0o755);
  console.log(`uv ${version} -> ${dest}`);
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
