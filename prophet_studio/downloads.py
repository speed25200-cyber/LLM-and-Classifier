"""Telechargements reprenables avec progression, verification SHA-256 et extraction d'archives sure.

Un telechargement de 6 Go sur une connexion domestique doit survivre a une coupure : fichier `.part`,
en-tete Range, reessais avec attente croissante ; la progression (octets, debit lisse, temps restant) part
vers l'interface par le bus d'evenements, limitee a ~4 messages/s par tache.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import threading
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import requests

USER_AGENT = "ProphetStudio/0.2 (+https://github.com/speed25200-cyber/LLM-and-Classifier)"


class Cancelled(Exception):
    pass


@dataclass
class DownloadJob:
    url: str
    dest: Path
    label: str = ""
    group: str = ""                 # regroupement dans l'interface (ex. "bonsai2-27b-ptq1")
    size: int = 0                   # taille attendue (0 = inconnue)
    sha256: str = ""
    extract_to: Path | None = None  # archive a extraire puis supprimer
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    status: str = "queued"          # queued | running | verifying | extracting | done | error | cancelled
    done_bytes: int = 0
    speed_bps: float = 0.0
    error: str = ""
    started: float = 0.0

    def public(self) -> dict:
        d = asdict(self)
        d["dest"], d["extract_to"] = str(self.dest), (str(self.extract_to) if self.extract_to else None)
        d["eta_s"] = round((self.size - self.done_bytes) / self.speed_bps) if self.speed_bps > 0 and self.size else None
        d["progress"] = round(self.done_bytes / self.size, 4) if self.size else None
        return d


def sha256_file(path: Path, chunk: int = 4 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def safe_extract(archive: Path, dest: Path) -> list[str]:
    """Extraction zip / tar(.gz/.bz2/.xz) en refusant tout chemin qui sortirait de `dest`."""
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    names: list[str] = []

    def check(name: str) -> Path:
        target = (dest / name).resolve()
        if target != dest and dest not in target.parents:
            raise ValueError(f"chemin d'archive dangereux : {name}")
        return target

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                check(info.filename)
                names.append(info.filename)
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            members = []
            for m in t.getmembers():
                check(m.name)
                if m.issym() or m.islnk():
                    check(os.path.join(os.path.dirname(m.name), m.linkname))
                if m.isdev():
                    continue
                members.append(m)
                names.append(m.name)
            t.extractall(dest, members=members)
    return names


class Downloader:
    def __init__(self, on_event: Callable[[dict], None] | None = None, workers: int = 2, retries: int = 6,
                 session: requests.Session | None = None):
        self.on_event = on_event or (lambda e: None)
        self.jobs: dict[str, DownloadJob] = {}
        self._queue: list[str] = []
        self._cancel: set[str] = set()
        self._cv = threading.Condition()
        self.retries = retries
        self.http = session or requests.Session()
        self.http.headers["User-Agent"] = USER_AGENT
        self._done_hooks: dict[str, Callable[[DownloadJob], None]] = {}
        self._last_emit: dict[str, float] = {}
        for _ in range(workers):
            threading.Thread(target=self._worker, daemon=True).start()

    # ---- API ------------------------------------------------------------------------------------------------------
    def add(self, job: DownloadJob, on_done: Callable[[DownloadJob], None] | None = None) -> DownloadJob:
        with self._cv:
            for j in self.jobs.values():   # meme destination deja en cours : on ne double pas
                if j.dest == job.dest and j.status in ("queued", "running", "verifying", "extracting"):
                    return j
            self.jobs[job.id] = job
            if on_done:
                self._done_hooks[job.id] = on_done
            self._queue.append(job.id)
            self._cv.notify()
        self._emit(job, force=True)
        return job

    def cancel(self, job_id: str | None = None, group: str | None = None) -> int:
        n = 0
        with self._cv:
            for j in self.jobs.values():
                if (job_id and j.id == job_id) or (group and j.group == group) or (not job_id and not group):
                    if j.status in ("queued", "running"):
                        self._cancel.add(j.id); n += 1
                        if j.status == "queued":
                            j.status = "cancelled"
                            if j.id in self._queue:
                                self._queue.remove(j.id)
                            self._emit(j, force=True)
        return n

    def snapshot(self) -> list[dict]:
        return [j.public() for j in self.jobs.values()]

    def wait(self, job_id: str, timeout: float = 60) -> DownloadJob:
        t0 = time.time()
        while time.time() - t0 < timeout:
            j = self.jobs[job_id]
            if j.status in ("done", "error", "cancelled"):
                return j
            time.sleep(0.02)
        raise TimeoutError(job_id)

    # ---- interne -----------------------------------------------------------------------------------------------------
    def _emit(self, job: DownloadJob, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_emit.get(job.id, 0) < 0.25:
            return
        self._last_emit[job.id] = now
        try:
            self.on_event({"type": "download.progress", "job": job.public()})
        except Exception:
            pass

    def _worker(self) -> None:
        while True:
            with self._cv:
                while not self._queue:
                    self._cv.wait()
                jid = self._queue.pop(0)
            job = self.jobs[jid]
            if job.status == "cancelled":
                continue
            try:
                self._run(job)
                job.status = "done"
            except Cancelled:
                job.status = "cancelled"
            except Exception as e:
                job.status, job.error = "error", str(e)[:400]
            finally:
                self._cancel.discard(job.id)
            self._emit(job, force=True)
            hook = self._done_hooks.pop(job.id, None)
            if hook and job.status == "done":
                try:
                    hook(job)
                except Exception as e:
                    job.status, job.error = "error", f"post-traitement : {str(e)[:300]}"
                    self._emit(job, force=True)

    def _run(self, job: DownloadJob) -> None:
        job.status, job.started = "running", time.time()
        job.dest.parent.mkdir(parents=True, exist_ok=True)
        part = job.dest.with_name(job.dest.name + ".part")
        if job.dest.exists() and (not job.size or job.dest.stat().st_size == job.size) and not job.extract_to:
            job.done_bytes = job.dest.stat().st_size
            return
        attempt = 0
        while True:
            try:
                self._fetch(job, part)
                break
            except Cancelled:
                raise
            except Exception as e:
                attempt += 1
                if attempt > self.retries:
                    raise RuntimeError(f"echec apres {self.retries} reessais : {e}") from e
                job.error = f"reessai {attempt}/{self.retries} : {str(e)[:120]}"
                self._emit(job, force=True)
                for _ in range(int(min(30, 2 ** attempt) * 10)):
                    if job.id in self._cancel:
                        raise Cancelled()
                    time.sleep(0.1)
        job.error = ""
        if job.sha256:
            job.status = "verifying"; self._emit(job, force=True)
            got = sha256_file(part)
            if got.lower() != job.sha256.lower():
                part.unlink(missing_ok=True)
                raise RuntimeError(f"empreinte SHA-256 inattendue ({got[:12]}... au lieu de {job.sha256[:12]}...)")
        part.replace(job.dest)
        if job.extract_to:
            job.status = "extracting"; self._emit(job, force=True)
            safe_extract(job.dest, job.extract_to)
            job.dest.unlink(missing_ok=True)

    def _fetch(self, job: DownloadJob, part: Path) -> None:
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        with self.http.get(job.url, headers=headers, stream=True, timeout=(15, 60), allow_redirects=True) as r:
            if r.status_code == 416:      # deja complet
                job.done_bytes = have
                return
            r.raise_for_status()
            if have and r.status_code != 206:  # le serveur ignore Range : on repart de zero
                have = 0
            total = int(r.headers.get("Content-Length") or 0) + have
            if total and not job.size:
                job.size = total
            job.done_bytes = have
            mode = "ab" if have else "wb"
            last_t, last_b = time.monotonic(), have
            with open(part, mode) as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if job.id in self._cancel:
                        raise Cancelled()
                    if not chunk:
                        continue
                    f.write(chunk)
                    job.done_bytes += len(chunk)
                    now = time.monotonic()
                    if now - last_t >= 0.5:
                        inst = (job.done_bytes - last_b) / (now - last_t)
                        job.speed_bps = inst if job.speed_bps == 0 else 0.7 * job.speed_bps + 0.3 * inst
                        last_t, last_b = now, job.done_bytes
                    self._emit(job)
        if job.size and job.done_bytes < job.size:
            raise IOError(f"telechargement incomplet ({job.done_bytes}/{job.size})")


def free_disk_gb(path: Path) -> float:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9
