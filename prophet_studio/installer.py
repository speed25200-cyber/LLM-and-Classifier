"""Installation en un clic : runtime llama.cpp (fork PrismML) adapte au GPU, modeles GGUF, voix.

  * runtime : l'asset de release est choisi selon l'OS, l'architecture et le GPU. Blackwell (RTX 50xx, sm_120)
    exige CUDA >= 12.8 ; CUDA 13 abandonne les GPU d'avant Turing ; sous Windows, le zip `cudart-*` de la
    meme version CUDA est ajoute (DLL du runtime CUDA) pour qu'aucun toolkit ne soit necessaire.
  * modeles : liste des fichiers via l'API Hugging Face (taille + SHA-256 LFS), telechargement reprenable.
  * registre `installed.json` : ce qui est present, ou, et quand ; les GGUF personnels (clone entraine) s'y importent.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import requests

from prophet_studio.catalog import MODELS, RUNTIME_REPO, VOICE
from prophet_studio.config import Paths, SettingsStore
from prophet_studio.downloads import DownloadJob, Downloader, free_disk_gb
from prophet_studio.hardware import GPU

EXE = ".exe" if sys.platform == "win32" else ""


# ---- choix de l'asset runtime -----------------------------------------------------------------------------------------
def _os_keys(os_name: str) -> tuple[str, ...]:
    o = os_name.lower()
    if o.startswith("win"):
        return ("win",)
    if o == "darwin" or o.startswith("mac"):
        return ("macos",)
    return ("ubuntu", "linux")


def _arch_key(arch: str) -> str:
    return "arm64" if arch.lower() in ("arm64", "aarch64") else "x64"


def _cuda_ver(name: str) -> float | None:
    m = re.search(r"cuda-?(\d+)\.(\d+)", name)
    return float(f"{m.group(1)}.{m.group(2)}") if m else None


def select_runtime_assets(assets: list[dict], os_name: str, arch: str, gpu: GPU | None) -> dict:
    """assets : [{name, browser_download_url, size}] d'une release. Renvoie {backend, cuda, main, extra[]}."""
    osk, ak = _os_keys(os_name), _arch_key(arch)
    mains = [a for a in assets if a["name"].startswith("llama-") and any(f"-{k}-" in a["name"] or f"-{k}." in a["name"] for k in osk)
             and ak in a["name"] and re.search(r"\.(zip|tar\.gz|tgz)$", a["name"])]
    if not mains and osk == ("macos",):
        mains = [a for a in assets if a["name"].startswith("llama-") and "macos" in a["name"] and ak in a["name"]]

    def pick(kind: str) -> list[dict]:
        if kind == "cuda":
            return [a for a in mains if "cuda" in a["name"]]
        if kind in ("vulkan", "rocm", "hip", "sycl"):
            return [a for a in mains if kind in a["name"]]
        return [a for a in mains if not any(b in a["name"] for b in ("cuda", "vulkan", "rocm", "hip", "sycl", "kompute", "opencl"))]

    choice, backend, cuda = None, "cpu", None
    if gpu is not None and gpu.vendor == "nvidia":
        cands = [(v, a) for a in pick("cuda") if (v := _cuda_ver(a["name"])) is not None]
        try:
            cc = float(gpu.compute_cap or 0)
        except ValueError:
            cc = 0.0
        try:
            drv = float(gpu.cuda_version or 99)
        except ValueError:
            drv = 99.0
        ok = [(v, a) for v, a in cands if v <= drv + 1e-6 and not (cc >= 10.0 and v < 12.8) and not (0 < cc < 7.5 and v >= 13.0)]
        if ok:
            cuda, choice = max(ok, key=lambda x: x[0])
            backend = "cuda"
        elif cands:
            # pilote trop ancien pour toutes les builds : on garde la plus basse compatible avec l'architecture et on
            # le signale (mise a jour du pilote conseillee)
            arch_ok = [(v, a) for v, a in cands if not (cc >= 10.0 and v < 12.8)]
            if arch_ok:
                cuda, choice = min(arch_ok, key=lambda x: x[0]); backend = "cuda"
    if choice is None and gpu is not None and gpu.vendor == "apple":
        m = [a for a in mains if "macos" in a["name"]]
        if m:
            choice, backend = m[0], "metal"
    if choice is None and gpu is not None and gpu.vendor in ("amd", "intel", "nvidia"):
        v = pick("vulkan")
        if v:
            choice, backend = v[0], "vulkan"
    if choice is None:
        c = pick("cpu")
        if c:
            choice, backend = sorted(c, key=lambda a: len(a["name"]))[0], "cpu"
    if choice is None:
        raise LookupError(f"aucun binaire llama.cpp pour {os_name}/{arch} dans cette release")
    extra = []
    if backend == "cuda" and osk == ("win",):
        rt = [a for a in assets if a["name"].startswith("cudart") and _cuda_ver(a["name"]) == cuda and "win" in a["name"]]
        extra = rt[:1]
    return {"backend": backend, "cuda": cuda, "main": choice, "extra": extra}


def guessed_assets(tag: str, os_name: str, arch: str) -> list[dict]:
    """Noms d'assets construits selon les conventions du fork (repli si l'API GitHub est injoignable)."""
    base = f"https://github.com/{RUNTIME_REPO}/releases/download/{tag}"
    osk, ak = _os_keys(os_name)[0], _arch_key(arch)
    names = []
    if osk == "win":
        for v in ("12.4", "12.8", "13.3"):
            names += [f"llama-{tag}-bin-win-cuda-{v}-x64.zip", f"cudart-llama-bin-win-cuda-{v}-x64.zip"]
        names += [f"llama-{tag}-bin-win-vulkan-x64.zip", f"llama-{tag}-bin-win-cpu-x64.zip"]
    elif osk == "macos":
        names += [f"llama-{tag}-bin-macos-{ak}.tar.gz"]
    else:
        names += [f"llama-{tag}-bin-linux-cuda-{v}-x64.tar.gz" for v in ("12.4", "12.8", "13.3")]
        names += [f"llama-{tag}-bin-ubuntu-vulkan-x64.tar.gz", f"llama-{tag}-bin-ubuntu-{ak}.tar.gz"]
    return [{"name": n, "browser_download_url": f"{base}/{n}", "size": 0} for n in names]


def find_server_binary(root: Path) -> Path | None:
    for p in sorted(root.rglob(f"llama-server{EXE}"), key=lambda x: len(x.parts)):
        if p.is_file():
            return p
    return None


# ---- installateur -------------------------------------------------------------------------------------------------------
class Installer:
    def __init__(self, paths: Paths, settings: SettingsStore, downloader: Downloader, emit: Callable[[dict], None],
                 http: requests.Session | None = None):
        self.paths, self.settings, self.dl, self.emit = paths, settings, downloader, emit
        self.http = http or requests.Session()
        self._lock = threading.Lock()
        self.registry = self._load()

    # ---- registre ------------------------------------------------------------------------------------------------
    def _load(self) -> dict:
        if self.paths.installed.exists():
            try:
                return json.loads(self.paths.installed.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"models": {}, "runtime": None, "voice": {}}

    def _save(self) -> None:
        tmp = self.paths.installed.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.registry, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.paths.installed)
        self.emit({"type": "install.changed", "installed": self.status()})

    def installed_models(self) -> set[str]:
        return {k for k, v in self.registry["models"].items() if Path(v.get("main", "")).exists()}

    def model_path(self, model_id: str) -> Path | None:
        v = self.registry["models"].get(model_id)
        return Path(v["main"]) if v and Path(v["main"]).exists() else None

    def mmproj_path(self, model_id: str) -> Path | None:
        v = self.registry["models"].get(model_id) or {}
        return Path(v["mmproj"]) if v.get("mmproj") and Path(v["mmproj"]).exists() else None

    def server_binary(self) -> Path | None:
        custom = self.settings.get().llama_server_path
        if custom and Path(custom).exists():
            return Path(custom)
        rt = self.registry.get("runtime") or {}
        if rt.get("server") and Path(rt["server"]).exists():
            return Path(rt["server"])
        return None

    def voice_files(self, voice_id: str) -> dict | None:
        v = self.registry["voice"].get(voice_id)
        return v.get("files") if v else None

    def status(self) -> dict:
        models = {}
        for mid, spec in MODELS.items():
            r = self.registry["models"].get(mid)
            models[mid] = {"installed": bool(r and Path(r["main"]).exists()), "path": r.get("main") if r else None,
                           "mmproj": bool(r and r.get("mmproj")), "size_gb": spec.size_gb}
        for mid, r in self.registry["models"].items():
            if mid not in MODELS:
                models[mid] = {"installed": Path(r["main"]).exists(), "path": r["main"], "custom": True, "role": r.get("role"), "label": r.get("label")}
        rt = self.registry.get("runtime")
        return {"models": models, "runtime": rt if (rt and self.server_binary()) else None,
                "custom_server": bool(self.settings.get().llama_server_path),
                "voice": {vid: (vid in self.registry["voice"]) for vid in VOICE}, "free_disk_gb": round(free_disk_gb(self.paths.root), 1)}

    # ---- Hugging Face ----------------------------------------------------------------------------------------------
    def hf_files(self, repo: str) -> list[dict]:
        ep = self.settings.get().hf_endpoint.rstrip("/")
        r = self.http.get(f"{ep}/api/models/{repo}/tree/main", params={"recursive": "true"}, timeout=20)
        r.raise_for_status()
        return [f for f in r.json() if f.get("type") == "file"]

    def _hf_job(self, repo: str, files: list[dict], pattern: str, dest_dir: Path, group: str, label: str) -> DownloadJob:
        hits = sorted((f for f in files if fnmatch.fnmatch(f["path"].split("/")[-1], pattern)), key=lambda f: (f["path"].count("/"), f["path"]))
        if not hits:
            raise LookupError(f"aucun fichier '{pattern}' dans {repo}")
        f = hits[0]
        lfs = f.get("lfs") or {}
        ep = self.settings.get().hf_endpoint.rstrip("/")
        return DownloadJob(f"{ep}/{repo}/resolve/main/{f['path']}", dest_dir / f["path"].split("/")[-1], label=label, group=group,
                           size=int(lfs.get("size") or f.get("size") or 0), sha256=lfs.get("oid", ""))

    def install_model(self, model_id: str, with_mmproj: bool = True) -> list[dict]:
        spec = MODELS[model_id]
        files = self.hf_files(spec.repo)
        dest = self.paths.models / model_id
        jobs = [self._hf_job(spec.repo, files, spec.pattern, dest, model_id, spec.label)]
        if with_mmproj and spec.mmproj_pattern:
            try:
                jobs.append(self._hf_job(spec.repo, files, spec.mmproj_pattern, dest, model_id, f"{spec.label} · vision"))
            except LookupError:
                pass
        need = sum(j.size for j in jobs) / 1e9
        if need and free_disk_gb(dest.parent) < need + 0.5:
            raise OSError(f"espace disque insuffisant : {need:.1f} Go necessaires")
        pending = {j.id for j in jobs}

        def done(job: DownloadJob) -> None:
            with self._lock:
                pending.discard(job.id)
                if pending:
                    return
                entry = {"main": str(jobs[0].dest), "role": spec.role, "installed_at": time.time(), "repo": spec.repo}
                if len(jobs) > 1 and jobs[1].status == "done":
                    entry["mmproj"] = str(jobs[1].dest)
                self.registry["models"][model_id] = entry
                self._save()
        return [self.dl.add(j, done).public() for j in jobs]

    def import_gguf(self, path: str, role: str, label: str = "") -> str:
        p = Path(path).expanduser().resolve()
        if not p.exists() or p.suffix.lower() != ".gguf":
            raise FileNotFoundError(f"GGUF introuvable : {p}")
        mid = f"custom-{role}-{re.sub(r'[^a-z0-9]+', '-', p.stem.lower()).strip('-')}"
        with self._lock:
            self.registry["models"][mid] = {"main": str(p), "role": role, "label": label or p.stem, "installed_at": time.time(), "custom": True}
            self._save()
        return mid

    def remove(self, item_id: str) -> bool:
        with self._lock:
            if item_id in self.registry["models"]:
                r = self.registry["models"].pop(item_id)
                if not r.get("custom"):
                    for k in ("main", "mmproj"):
                        if r.get(k):
                            Path(r[k]).unlink(missing_ok=True)
                self._save(); return True
            if item_id in self.registry["voice"]:
                self.registry["voice"].pop(item_id); self._save(); return True
        return False

    # ---- runtime -------------------------------------------------------------------------------------------------------
    def release_assets(self, tag: str) -> list[dict]:
        try:
            r = self.http.get(f"https://api.github.com/repos/{RUNTIME_REPO}/releases/tags/{tag}", timeout=20,
                              headers={"Accept": "application/vnd.github+json"})
            r.raise_for_status()
            return r.json().get("assets", [])
        except Exception:
            return []

    def plan_runtime(self, hw) -> dict:
        tag = self.settings.get().runtime_tag
        assets = self.release_assets(tag)
        guessed = not assets
        sel = select_runtime_assets(assets or guessed_assets(tag, hw.os, hw.arch), hw.os, hw.arch, hw.gpu)
        return {"tag": tag, "guessed": guessed, **sel}

    def install_runtime(self, hw) -> list[dict]:
        sel = self.plan_runtime(hw)
        tag, backend = sel["tag"], sel["backend"]
        dest = self.paths.bin / tag / (f"cuda-{sel['cuda']}" if backend == "cuda" else backend)
        jobs = [DownloadJob(a["browser_download_url"], self.paths.bin / "_dl" / a["name"], label=f"llama.cpp {backend}" + (f" {sel['cuda']}" if sel["cuda"] else ""),
                            group="runtime", size=int(a.get("size") or 0), extract_to=dest) for a in [sel["main"], *sel["extra"]]]
        pending = {j.id for j in jobs}

        def done(job: DownloadJob) -> None:
            with self._lock:
                pending.discard(job.id)
                if pending:
                    return
                srv = find_server_binary(dest)
                if srv is None:
                    raise FileNotFoundError("llama-server introuvable dans l'archive")
                if os.name != "nt":
                    srv.chmod(srv.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                self.registry["runtime"] = {"tag": tag, "backend": backend, "cuda": sel["cuda"], "server": str(srv), "dir": str(dest),
                                            "installed_at": time.time(), "version": self.server_version(srv)}
                self._save()
        return [self.dl.add(j, done).public() for j in jobs]

    @staticmethod
    def server_version(binary: Path) -> str:
        kw = {"creationflags": 0x08000000} if sys.platform == "win32" else {}
        env = dict(os.environ)
        if sys.platform.startswith("linux"):
            env["LD_LIBRARY_PATH"] = f"{binary.parent}:{env.get('LD_LIBRARY_PATH', '')}"
        try:
            r = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=20, env=env, **kw)
            return (r.stdout + r.stderr).strip().splitlines()[-1][:200] if (r.stdout + r.stderr).strip() else ""
        except Exception as e:
            return f"? ({str(e)[:80]})"

    # ---- voix ------------------------------------------------------------------------------------------------------------
    def install_voice(self, voice_id: str) -> list[dict]:
        spec = VOICE[voice_id]
        target = self.paths.voice / voice_id
        name = spec.url.rsplit("/", 1)[-1]
        archive = name.endswith((".tar.bz2", ".tar.gz", ".zip"))
        job = DownloadJob(spec.url, (self.paths.voice / "_dl" / name) if archive else (target / name), label=spec.label, group=voice_id,
                          size=spec.size_mb * 1_000_000 if not archive else 0, extract_to=target if archive else None)
        job.size = 0  # tailles de catalogue approximatives : on se fie a Content-Length

        def done(_job: DownloadJob) -> None:
            files = {}
            for role, pat in spec.files.items():
                hits = sorted((p for p in target.rglob("*") if fnmatch.fnmatch(p.name, pat)), key=lambda p: (len(p.parts), p.name))
                if hits:
                    files[role] = str(hits[0])
            with self._lock:
                self.registry["voice"][voice_id] = {"files": files, "dir": str(target), "installed_at": time.time()}
                self._save()
        return [self.dl.add(job, done).public()]

    def recommended(self, plan) -> list[str]:
        """Ce qu'il faut pour demarrer : runtime + modeles du plan + voix par defaut."""
        want = ["runtime", plan.s2.model_id]
        if plan.s1:
            want.append(plan.s1.model_id)
        v = self.settings.get().voice
        want += ["vad-silero", v.stt_model, v.tts_voice]
        have = self.installed_models() | set(self.registry["voice"]) | ({"runtime"} if self.server_binary() else set())
        return [w for w in want if w not in have]
