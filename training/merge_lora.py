"""Adaptateur LoRA / QLoRA -> modele HF fusionne -> GGUF (f16) -> GGUF quantifies (Q8_0, Q4_K_M...) pour Prophet Studio.

    # apres train_lora_rlcd.py --qlora (ou LoRA) : la base est rechargee en bf16 (jamais en 4-bit) puis fusionnee
    python training/merge_lora.py --adapter runs/jev-clone --llama-cpp ../llama.cpp --quant Q8_0,Q4_K_M
    # apres --full (deja complet) : conversion seulement
    python training/merge_lora.py --merged runs/jev-clone/merged --llama-cpp ../llama.cpp --quant Q8_0
    # verifier les outils et le plan sans rien calculer
    python training/merge_lora.py --adapter runs/jev-clone --llama-cpp ../llama.cpp --check

Sorties (defaut --gguf-dir = dossier parent de l'adaptateur) : <nom>-f16.gguf, <nom>-Q8_0.gguf, <nom>-Q4_K_M.gguf et
<nom>.manifest.json (base, adaptateur, tailles). Studio : Modeles > Importer un GGUF, role Classifieur (s1), puis Calibrer.

Outils : torch + transformers + peft (fusion) ; convert_hf_to_gguf.py d'un clone de llama.cpp, de preference le fork PrismML
au tag du runtime de Studio (--llama-cpp) ; llama-quantize (--quantize-bin, sinon bin/*/ du depot, le runtime de Studio,
<llama.cpp>/build/bin, le PATH). Tout est verifie AVANT la fusion : un outil manquant arrete le script tout de suite.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LLAMA_TAG = "prism-b10683-d8f26ee"          # runtime de Prophet Studio (prophet_studio/config.py)
LLAMA_REPO = "https://github.com/PrismML-Eng/llama.cpp"
EXE = ".exe" if os.name == "nt" else ""
QUANTS = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0", "IQ4_NL")
OUTTYPES = ("f16", "bf16", "f32")


class ToolMissing(SystemExit):
    """Outil ou dependance absent : arret immediat avec la commande d'installation."""

    def __init__(self, msg: str):
        super().__init__(f"merge_lora : {msg}")


def need_modules(mods: list[str], why: str) -> None:
    missing = [m for m in mods if importlib.util.find_spec(m) is None]
    if missing:
        raise ToolMissing(f"{', '.join(missing)} manquant(s) pour {why} : pip install -e \".[train]\" (ou pip install "
                          + " ".join(missing) + ")")


def base_of(adapter: Path) -> str:
    cfg = adapter / "adapter_config.json"
    if not cfg.is_file():
        raise ToolMissing(f"{adapter} n'est pas un adaptateur LoRA (adapter_config.json absent) ; pour un modele deja complet "
                          f"(train_lora_rlcd.py --full) utilisez --merged {adapter / 'merged'}")
    base = json.loads(cfg.read_text(encoding="utf-8")).get("base_model_name_or_path")
    if not base:
        raise ToolMissing(f"{cfg} ne nomme pas le modele de base : passez --base <depot HF ou dossier>")
    return base


def find_convert(llama_cpp: str | None) -> Path:
    cands = [Path(llama_cpp)] if llama_cpp else []
    cands += [Path(p) for p in (os.environ.get("LLAMA_CPP_DIR"),) if p] + [ROOT / "llama.cpp", ROOT.parent / "llama.cpp", Path("/content/llama.cpp-src")]
    for d in cands:
        f = d if d.name == "convert_hf_to_gguf.py" else d / "convert_hf_to_gguf.py"
        if f.is_file():
            return f
    # pas requirements-convert_hf_to_gguf.txt : il epingle un torch CPU et remplacerait le torch CUDA de l'entrainement
    raise ToolMissing("convert_hf_to_gguf.py introuvable (--llama-cpp <dossier de llama.cpp>). Pour l'obtenir :\n"
                      f"    git clone --depth 1 -b {LLAMA_TAG} {LLAMA_REPO} llama.cpp\n"
                      "    pip install sentencepiece protobuf   (gguf-py est lu dans le clone)")


def find_quantize(explicit: str | None, llama_cpp: str | None) -> Path:
    name = f"llama-quantize{EXE}"
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        raise ToolMissing(f"--quantize-bin {explicit} introuvable")
    cands = [Path(os.environ["LLAMA_QUANTIZE"])] if os.environ.get("LLAMA_QUANTIZE") else []
    cands += [ROOT / "bin" / b / name for b in ("cuda", "cpu", "vulkan", "rocm")]
    if llama_cpp:
        d = Path(llama_cpp)
        d = d.parent if d.suffix == ".py" else d
        cands += [d / "build" / "bin" / name, d / "build" / "bin" / "Release" / name]
    try:   # runtime installe par Prophet Studio (Modeles > Runtime)
        from prophet_studio.config import Paths
        b = Paths().bin
        if b.is_dir():
            cands += sorted(b.rglob(name), key=lambda x: len(x.parts))
    except Exception:
        pass
    for p in cands:
        if p.is_file():
            return p
    w = shutil.which("llama-quantize")
    if w:
        return Path(w)
    raise ToolMissing("llama-quantize introuvable (--quantize-bin). Il est dans l'archive du runtime llama.cpp : "
                      f"{LLAMA_REPO}/releases/tag/{LLAMA_TAG} (ou ./scripts/setup.sh -> bin/<backend>/, ou Prophet Studio, "
                      "Modeles > Runtime), ou compilez-le : cmake -B build && cmake --build build --target llama-quantize")


def plan(args) -> dict:
    """Verifie outils et entrees (rien n'est calcule) ; -> etapes et chemins."""
    if bool(args.adapter) == bool(args.merged):
        raise SystemExit("merge_lora : donnez --adapter (LoRA / QLoRA) ou --merged (modele deja complet), pas les deux")
    quants = [q.strip() for q in (args.quant or "").split(",") if q.strip()]
    unknown = [q for q in quants if q not in QUANTS]
    if unknown:
        raise SystemExit(f"merge_lora : quantification inconnue {unknown} (parmi {', '.join(QUANTS)})")
    src = Path(args.adapter or args.merged)
    if not src.is_dir():
        raise SystemExit(f"merge_lora : dossier introuvable : {src}")
    p: dict = {"quants": quants}
    if args.adapter:
        p["base"] = args.base or base_of(src)
        p["merged"] = Path(args.out) if args.out else src / "merged"
        need_modules(["torch", "transformers", "peft"], "la fusion de l'adaptateur")
    else:
        if not (src / "config.json").is_file():
            raise SystemExit(f"merge_lora : {src} n'est pas un modele HF (config.json absent)")
        p["merged"] = src
    if not args.no_gguf:
        need_modules(["torch", "transformers", "numpy"], "convert_hf_to_gguf.py")
        p["convert"] = find_convert(args.llama_cpp)
        if quants:
            p["quantize"] = find_quantize(args.quantize_bin, args.llama_cpp)
        run_dir = src if args.adapter or src.name != "merged" else src.parent   # runs/jev-clone (adaptateur ou .../merged)
        gdir = Path(args.gguf_dir) if args.gguf_dir else run_dir.parent
        name = args.name or run_dir.name
        p["gguf"] = {"dir": gdir, "f16": gdir / f"{name}-{args.outtype}.gguf", **{q: gdir / f"{name}-{q}.gguf" for q in quants}}
        p["name"] = name
    return p


def merge(base: str, adapter: Path, out: Path, dtype: str = "bfloat16", device: str = "cpu") -> Path:
    """Base rechargee en pleine precision (jamais 4-bit), adaptateur applique puis fusionne, modele + tokenizer sauvegardes."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(base, dtype=getattr(torch, dtype), device_map={"": device} if device != "auto" else "auto",
                                                 low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
    tok_src = adapter if (adapter / "tokenizer_config.json").is_file() else base
    AutoTokenizer.from_pretrained(tok_src).save_pretrained(out)
    print(f"modele fusionne dans {out} ({time.perf_counter() - t0:.0f} s)")
    return out


def run(cmd: list, what: str) -> None:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode != 0:
        raise SystemExit(f"merge_lora : echec de {what} (code {r.returncode}) ; voir la sortie ci-dessus")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fusion d'un adaptateur LoRA / QLoRA et export GGUF pour Prophet Studio")
    ap.add_argument("--adapter", help="dossier de l'adaptateur (sortie de train_lora_rlcd.py sans --full)")
    ap.add_argument("--merged", help="modele HF deja complet (sortie de --full) : conversion GGUF seulement")
    ap.add_argument("--base", help="modele de base (defaut : adapter_config.json)")
    ap.add_argument("--out", help="dossier du modele fusionne (defaut : <adaptateur>/merged)")
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--device", default="cpu", help="cpu (defaut, RAM ~2x la taille du modele), cuda ou auto")
    ap.add_argument("--llama-cpp", help="dossier d'un clone de llama.cpp (convert_hf_to_gguf.py)")
    ap.add_argument("--quantize-bin", help="chemin de llama-quantize")
    ap.add_argument("--gguf-dir", help="dossier des GGUF (defaut : parent de l'adaptateur ou du modele)")
    ap.add_argument("--name", help="prefixe des fichiers GGUF (defaut : nom du dossier d'entrainement)")
    ap.add_argument("--outtype", choices=OUTTYPES, default="f16", help="GGUF de conversion (avant quantification)")
    ap.add_argument("--quant", default="Q8_0,Q4_K_M", help="quantifications, separees par des virgules ('' = aucune)")
    ap.add_argument("--no-gguf", action="store_true", help="fusionner seulement (pas de conversion)")
    ap.add_argument("--check", action="store_true", help="verifier les outils et afficher le plan, sans rien calculer")
    args = ap.parse_args(argv)

    p = plan(args)
    print("plan :", json.dumps({k: (str(v) if isinstance(v, Path) else {kk: str(vv) for kk, vv in v.items()} if isinstance(v, dict) else v)
                                for k, v in p.items()}, ensure_ascii=False, indent=1))
    if args.check:
        return p
    if args.adapter:
        merge(p["base"], Path(args.adapter), p["merged"], args.dtype, args.device)
    if args.no_gguf:
        return p
    g = p["gguf"]
    g["dir"].mkdir(parents=True, exist_ok=True)
    run([sys.executable, p["convert"], p["merged"], "--outfile", g["f16"], "--outtype", args.outtype], "la conversion GGUF")
    for q in p["quants"]:
        run([p["quantize"], g["f16"], g[q], q], f"la quantification {q}")
    files = {k: {"path": str(v), "bytes": v.stat().st_size} for k, v in g.items() if k != "dir" and v.is_file()}
    manifest = {"name": p["name"], "base": p.get("base"), "adapter": args.adapter, "merged": str(p["merged"]), "files": files,
                "llama_cpp": str(p["convert"].parent), "ts": round(time.time(), 1)}
    (g["dir"] / f"{p['name']}.manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    best = next((g[q] for q in p["quants"]), g["f16"])
    print(f"\nGGUF prets dans {g['dir']} : " + ", ".join(Path(f["path"]).name for f in files.values()))
    print(f"Studio : Modeles > Importer un GGUF -> {best.name}, role Classifieur (s1) ; puis, quand il tourne, Calibrer.")
    return p


if __name__ == "__main__":
    main()
