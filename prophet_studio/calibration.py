"""Calibration du classifieur (System One) dans Studio : un fichier par modele, applique au seul S1 qui tourne.

  <donnees>/runs/calibration/<id du modele>.json : temperature par primitive + seuils par question + meta
  * produit par POST /api/calibrate (graines etiquetees livrees avec jev_clone, lues par le S1 en marche) ou par
    `python -m jev_clone.calibrate --studio-model <id>`, ou importe (POST /api/calibration/import) ;
  * charge par AgentService pour le classifieur en marche seulement : jamais en mode mono (Bonsai n'est pas le
    modele calibre), jamais pour un autre S1 ; le cache des moteurs est invalide a chaque changement ;
  * expose par modele dans l'etat (installed.calibration) : l'interface affiche "non calibre" sinon.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Callable

from jev_clone.readout import Calibration

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_running = threading.Lock()


class Busy(RuntimeError):
    pass


def calibration_file(runs: Path, model_id: str | None) -> Path | None:
    if not model_id or not SAFE_ID.match(model_id):
        return None
    return Path(runs) / "calibration" / f"{model_id}.json"


def active_s1_model(runtime) -> str | None:
    """Classifieur reellement en marche ; None en mode mono (Bonsai repond aux questions S1) ou sans plan."""
    p = runtime.plan
    return None if runtime.mono or p is None or p.s1 is None else p.s1.model_id


def forget(runs: Path, model_id: str) -> None:
    """Nouveaux poids sous le meme id (GGUF re-importe) : l'ancienne calibration ne vaut plus rien."""
    f = calibration_file(runs, model_id)
    if f is not None:
        f.unlink(missing_ok=True)


def status(runs: Path) -> dict[str, dict]:
    """Calibrations presentes, par id de modele (un fichier illisible est signale, jamais applique en silence)."""
    d = Path(runs) / "calibration"
    out: dict[str, dict] = {}
    for f in sorted(d.glob("*.json")) if d.is_dir() else []:
        try:
            cal = Calibration.load(f)
        except Exception as e:
            out[f.stem] = {"model_id": f.stem, "error": f"fichier illisible : {str(e)[:160]}"}
            continue
        m = cal.meta
        out[f.stem] = {"model_id": f.stem, "source": m.get("source", "import"), "n": m.get("n"), "ts": m.get("ts") or round(f.stat().st_mtime, 1),
                       "temperature": cal.temperature, "thresholds": cal.thresholds, "report": m.get("report", {})}
    return out


def store(runs: Path, model_id: str, cal: Calibration) -> dict:
    f = calibration_file(runs, model_id)
    if f is None:
        raise ValueError(f"identifiant de modele invalide : {model_id!r}")
    cal.meta = {**cal.meta, "model_id": model_id}
    cal.save(f)
    return {"model_id": model_id, "path": str(f), "n": cal.meta.get("n"), "temperature": cal.temperature,
            "thresholds": cal.thresholds, "report": cal.meta.get("report", {})}


def import_calibration(runs: Path, model_id: str, data: dict | None = None, path: str | None = None, force: bool = False) -> dict:
    """calibration.json (notebook Colab, autre machine...) -> fichier du modele ; refuse un fichier fait pour un autre modele."""
    if data is None:
        p = Path(str(path or "")).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"calibration introuvable : {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
    cal = Calibration.from_dict(data)
    made_for = cal.meta.get("model_id")
    if made_for and made_for != model_id and not force:
        raise ValueError(f"cette calibration a ete faite pour {made_for}, pas pour {model_id} (force pour l'importer quand meme)")
    cal.meta = {**cal.meta, "source": "import", **({"calibrated_for": made_for} if made_for and made_for != model_id else {})}
    return store(runs, model_id, cal)


def run(s1_url: str, model_id: str, emit: Callable[[dict], None], target_precision: float = 0.95,
        examples: list[dict] | None = None, engine=None) -> Calibration:
    """Lit les graines etiquetees avec le classifieur en marche (lecture brute, T = 1) et ajuste temperature + seuils.
    Evenements `s1.calibration` (running, puis error) ; une seule calibration a la fois (Busy) ; n'ecrit rien."""
    from jev_clone.calibrate import calibrate, load_examples
    if not _running.acquire(blocking=False):
        raise Busy("une calibration est deja en cours")
    try:
        exs = load_examples() if examples is None else examples
        if engine is None:
            from jev_clone.backend_llamacpp import LlamaCppBackend
            from jev_clone.engine import SystemOneEngine
            engine = SystemOneEngine(LlamaCppBackend(s1_url, max_workers=4, timeout=120), calibration=Calibration(), model_name=model_id)

        def progress(done: int, total: int) -> None:
            emit({"type": "s1.calibration", "status": "running", "model": model_id, "done": done, "total": total})
        progress(0, len(exs))
        cal, _ = calibrate(engine, exs, target_precision, progress, source="studio", model_id=model_id)
        return cal
    except Exception as e:
        emit({"type": "s1.calibration", "status": "error", "model": model_id, "error": f"{type(e).__name__}: {str(e)[:300]}"})
        raise
    finally:
        _running.release()
