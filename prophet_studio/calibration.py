"""Calibration du classifieur (System One) dans Studio : un fichier par modele, applique au seul S1 qui tourne.

  <donnees>/runs/calibration/<id du modele>.json : temperature et seuil par question calibree + meta
  * produit par POST /api/calibrate (graines etiquetees livrees avec jev_clone, lues par le S1 en marche) ou par
    `python -m jev_clone.calibrate --studio-model <id>`, ou importe (POST /api/calibration/import) ;
  * lie aux poids sur lesquels il a ete ajuste (meta.weights : chemin, taille, date, empreinte du debut et de la fin du
    GGUF) : un GGUF remplace sur place rend la calibration perimee, jamais appliquee (l'etat le dit) ;
  * charge par AgentService pour le classifieur en marche seulement : jamais en mode mono (Bonsai n'est pas le
    modele calibre), jamais pour un autre S1 ; le cache des moteurs est invalide a chaque changement ;
  * une temperature ne vaut que pour les questions du pre-tour sur lesquelles elle a ete ajustee : le garde-fou, la
    voix et le computer use restent lus bruts ; une calibration generique (temperature seule, sans question connue)
    n'est pas appliquee par Studio ;
  * expose par modele dans l'etat (installed.calibration) : l'interface affiche "non calibre" sinon.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Callable

from jev_clone.readout import Calibration

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_running = threading.Lock()
_HASH_SPAN = 1 << 20          # empreinte des poids : 1 Mio au debut + 1 Mio a la fin (un GGUF de 2 Go se relit en ~1 ms)
_hash_cache: dict[tuple, str] = {}


class Busy(RuntimeError):
    pass


def calibration_file(runs: Path, model_id: str | None) -> Path | None:
    if not model_id or not SAFE_ID.match(model_id):
        return None
    return Path(runs) / "calibration" / f"{model_id}.json"


def id_problem(model_id: str | None) -> str | None:
    """Message clair si ce modele ne peut pas recevoir de calibration (identifiant trop long ou invalide)."""
    if calibration_file(Path("."), model_id) is not None:
        return None
    return (f"identifiant de classifieur non calibrable ({len(model_id or '')} caracteres, 128 au plus, lettres, chiffres, . _ -) : "
            f"{str(model_id)[:60]}... ; renommez le GGUF plus court et re-importez-le")


# ---- lien calibration <-> poids -------------------------------------------------------------------------------------------
def _head_tail(p: Path, size: int) -> str:
    h = hashlib.sha256(str(size).encode())
    with open(p, "rb") as f:
        h.update(f.read(_HASH_SPAN))
        if size > 2 * _HASH_SPAN:
            f.seek(size - _HASH_SPAN)
            h.update(f.read(_HASH_SPAN))
        elif size > _HASH_SPAN:
            h.update(f.read())
    return h.hexdigest()[:24]


def weights_id(path: Path | str) -> dict:
    """Identite des poids : chemin, taille, date (verification rapide) et empreinte debut + fin (copie, meme date)."""
    p = Path(path)
    st = p.stat()
    return {"path": str(p), "size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha": _head_tail(p, st.st_size)}


def weights_check(cal: Calibration, current: Path | str | None = None) -> tuple[bool | None, str]:
    """(True, "") : memes poids ; (False, raison) : poids changes ou absents ; (None, "") : ancienne calibration, non liee.
    current : GGUF actuellement enregistre pour ce modele (s'il differe du chemin calibre, la calibration est perimee)."""
    w = cal.meta.get("weights")
    if not isinstance(w, dict) or not w.get("path"):
        return None, ""
    p = Path(w["path"])
    if current is not None and Path(current) != p:
        return False, "le modele pointe vers un autre GGUF que celui calibre"
    try:
        st = p.stat()
    except OSError:
        return False, "GGUF calibre introuvable"
    if st.st_size != w.get("size"):
        return False, "poids modifies depuis la calibration (taille)"
    if st.st_mtime_ns == w.get("mtime_ns"):
        return True, ""
    key = (str(p), st.st_size, st.st_mtime_ns)   # date changee : on compare le contenu, une fois par etat du fichier
    if key not in _hash_cache:
        try:
            _hash_cache[key] = _head_tail(p, st.st_size)
        except OSError:
            return False, "GGUF calibre illisible"
    return (True, "") if _hash_cache[key] == w.get("sha") else (False, "poids modifies depuis la calibration")


def usable(cal: Calibration) -> tuple[bool, str]:
    """Studio n'applique une calibration qu'aux questions sur lesquelles elle a ete ajustee et aux poids calibres."""
    ok, why = weights_check(cal)
    if ok is False:
        return False, f"{why} : recalibrez ce classifieur (bouton Calibrer)"
    if cal.generic and cal.is_active():
        return False, ("calibration generique (temperature seule, sans question ajustee) : non appliquee, elle toucherait "
                       "aussi le garde-fou et la voix ; recalibrez ce classifieur (bouton Calibrer)")
    return True, ""


def for_studio(cal: Calibration) -> Calibration:
    """Ancien fichier (temperature par primitive + seuils) : Studio ne l'applique qu'aux questions du pre-tour de Prophet
    de ses seuils (reconnues a leur empreinte), jamais a un autre nom (tool_risk d'un jeu de validation, par ex.)."""
    if cal.questions or not cal.thresholds:
        return cal
    keep = {q: {**e, "T": cal.t(e["kind"])} for q, e in cal.scope().items() if e.get("fp")}
    return Calibration(temperature=dict(cal.temperature), thresholds={q: cal.thresholds[q] for q in keep}, meta=dict(cal.meta), questions=keep)


def load_for_engine(path: str | Path | None) -> tuple[Calibration, str, str | None]:
    """Calibration a appliquer au S1 en marche, ou lecture brute (T = 1) et la raison (illisible, perimee, generique) ;
    plus le GGUF calibre a surveiller (un remplacement sur place invalide le moteur)."""
    if path is None:
        return Calibration(), "", None
    try:
        cal = Calibration.load(path)
    except Exception as e:
        return Calibration(), f"fichier illisible : {str(e)[:160]}", None
    ok, why = usable(cal)
    w = (cal.meta.get("weights") or {}).get("path") if isinstance(cal.meta.get("weights"), dict) else None
    return (for_studio(cal), "", w) if ok else (Calibration(), why, w)


def registered_weights(installed: Path, model_id: str) -> Path | None:
    """GGUF d'un modele d'apres le registre de Studio (installed.json), s'il existe."""
    try:
        main = json.loads(Path(installed).read_text(encoding="utf-8"))["models"][model_id]["main"]
    except Exception:
        return None
    return Path(main) if Path(main).is_file() else None


def active_s1_model(runtime) -> str | None:
    """Classifieur reellement en marche ; None en mode mono (Bonsai repond aux questions S1) ou sans plan."""
    p = runtime.plan
    return None if runtime.mono or p is None or p.s1 is None else p.s1.model_id


def forget(runs: Path, model_id: str) -> None:
    """Nouveaux poids sous le meme id (GGUF re-importe) : l'ancienne calibration ne vaut plus rien."""
    f = calibration_file(runs, model_id)
    if f is not None:
        f.unlink(missing_ok=True)


def status(runs: Path, model_path: Callable[[str], Path | None] | None = None) -> dict[str, dict]:
    """Calibrations presentes, par id de modele. Un fichier illisible, perime (poids remplaces) ou generique porte `error` :
    signale, jamais applique en silence. model_path : GGUF enregistre de chaque modele (autre chemin = perimee)."""
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
                       "temperature": cal.temperature, "thresholds": cal.thresholds, "report": m.get("report", {}),
                       "questions": {q: e.get("T") for q, e in for_studio(cal).questions.items()}}   # celles que Studio applique
        current = model_path(f.stem) if model_path is not None else None
        ok, why = weights_check(cal, current)
        if ok is False:
            out[f.stem].update(error=f"calibration perimee : {why} ; recalibrez ce classifieur (bouton Calibrer)", stale=True)
        elif ok is None:
            out[f.stem]["unverified"] = True          # ancienne calibration, non liee aux poids : appliquee telle quelle
        good, why = usable(cal)
        if not good and "error" not in out[f.stem]:
            out[f.stem]["error"] = why
    return out


def store(runs: Path, model_id: str, cal: Calibration, weights: Path | str | None = None) -> dict:
    """Ecrit la calibration du modele ; weights : GGUF calibre (la calibration ne vaudra que pour ces poids-la)."""
    f = calibration_file(runs, model_id)
    if f is None:
        raise ValueError(id_problem(model_id))
    cal.meta = {**cal.meta, "model_id": model_id}
    if weights is not None:
        cal.meta["weights"] = weights_id(weights)
    cal.save(f)
    return {"model_id": model_id, "path": str(f), "n": cal.meta.get("n"), "temperature": cal.temperature,
            "thresholds": cal.thresholds, "report": cal.meta.get("report", {})}


def import_calibration(runs: Path, model_id: str, data: dict | None = None, path: str | None = None, force: bool = False,
                       weights: Path | str | None = None) -> dict:
    """calibration.json (notebook Colab, autre machine...) -> fichier du modele, lie a ses poids ; refuse un fichier fait
    pour un autre modele (`force`, option de l'API seulement, passe outre)."""
    if data is None:
        p = Path(str(path or "")).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"calibration introuvable : {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
    cal = Calibration.from_dict(data)
    made_for = cal.meta.get("model_id")
    if made_for and made_for != model_id and not force:
        raise ValueError(f"cette calibration a ete faite pour {made_for}, pas pour {model_id} : recalibrez ce classifieur "
                         f"(bouton Calibrer quand il tourne), ou retirez meta.model_id du fichier s'il s'agit du meme GGUF")
    cal.meta = {k: v for k, v in cal.meta.items() if k != "weights"}   # les poids d'une autre machine ne se verifient pas ici
    cal.meta = {**cal.meta, "source": "import", **({"calibrated_for": made_for} if made_for and made_for != model_id else {})}
    return store(runs, model_id, cal, weights)


def run(s1_url: str, model_id: str, emit: Callable[[dict], None], target_precision: float = 0.95,
        examples: list[dict] | None = None, engine=None) -> Calibration:
    """Lit les graines etiquetees avec le classifieur en marche (lecture brute, T = 1) et ajuste temperature + seuils.
    Evenements `s1.calibration` (running, puis error) ; une seule calibration a la fois (Busy) ; n'ecrit rien."""
    from jev_clone.calibrate import calibrate, load_examples
    if id_problem(model_id):             # avant la lecture des graines : rien ne pourrait etre ecrit a la fin
        raise ValueError(id_problem(model_id))
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
