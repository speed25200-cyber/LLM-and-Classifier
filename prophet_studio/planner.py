"""Planificateur VRAM : du materiel detecte a une configuration de lancement des deux serveurs.

Principe (cible RTX 5060 8 Go, valable pour toute carte) :
  1. Budget = VRAM totale - ce qu'utilisent deja le bureau et les autres applications - marge de securite.
  2. System Two (Bonsai) doit tenir entierement sur le GPU : on prend le meilleur modele qui y tient.
  3. On maximise le contexte, puis on place le classifieur (System One) sur le GPU s'il reste de la place,
     sinon sur le CPU (un 1.7B ternaire y decide encore en ~0,1-0,3 s).
  4. Chaque plan porte la ventilation memoire (barre VRAM de l'interface), les debits attendus et ses raisons.
  5. `degrade()` donne le cran suivant de l'echelle anti-OOM (utilisee automatiquement par le superviseur).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

from prophet_studio.catalog import MODELS, ModelSpec, kv_mib
from prophet_studio.hardware import GPU, HardwareInfo

CTX_STEPS = [131072, 98304, 65536, 49152, 32768, 24576, 16384, 12288, 8192, 6144, 4096]
PRIORITY_CTX_CAP = {"equilibre": 32768, "contexte": 131072, "vitesse": 16384}
S1_CTX, S1_SLOTS, S1_KV = 8192, 4, "q8_0"
S1_SLOTS_CPU = 1        # classifieur sur CPU : un seul slot, branches en sequence sur le meme cache (pas de re-prefill par slot)
SAFETY_MIB = 256
S2_LAYERS = 65          # 64 blocs + sortie (Bonsai 2 27B / Qwen3.8) : repere pour le dechargement partiel


@dataclass
class ServerPlan:
    role: str
    model_id: str
    device: str                  # "gpu" | "cpu" | "partial"
    ngl: int
    ctx: int
    np: int = 1
    kv_type: str = "f16"
    mmproj: str = "off"          # off | cpu | gpu (System Two seulement)
    reasoning_budget: int = -1
    threads: int | None = None


@dataclass
class Plan:
    backend: str                 # cuda | metal | cpu
    priority: str
    s2: ServerPlan
    s1: ServerPlan | None
    budget: dict
    fits: bool
    expected: dict
    notes: list[str] = field(default_factory=list)
    rung: int = 0
    title: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _gib(x: float) -> float:
    return x * 1024.0


MEASURED: dict[str, float] = {}   # surcout runtime mesure sur cette machine (Mio), par modele : voir calibrated_overhead()


def calibrated_overhead(model_id: str, kv_kib: float, weights_gib: float, ctx: int, kv_type: str, mmproj_mib: float, used_mib: float) -> float:
    """Surcout runtime deduit d'une mesure : VRAM prise par le serveur - poids - KV - vision, borne a [200, 3000] Mio."""
    from prophet_studio.catalog import KV_FACTOR
    kv = kv_kib * KV_FACTOR.get(kv_type, 1.0) * ctx / 1024.0
    return max(200.0, min(3000.0, used_mib - weights_gib * 1024.0 - kv - mmproj_mib))


def _overhead_mib(m: ModelSpec) -> float:
    if m.id in MEASURED:
        return MEASURED[m.id] + 64.0   # petite marge au-dessus de la mesure
    return _gib(m.overhead_gib)


def s2_mib(m: ModelSpec, ctx: int, kv: str, mmproj_gpu: bool) -> dict:
    return {"s2_weights": _gib(m.weights_gib), "s2_runtime": _overhead_mib(m), "s2_kv": kv_mib(m, ctx, kv),
            "mmproj": _gib(m.mmproj_gib) if mmproj_gpu else 0.0}


def s1_mib(m: ModelSpec) -> dict:
    return {"s1_weights": _gib(m.weights_gib), "s1_runtime": _overhead_mib(m), "s1_kv": kv_mib(m, S1_CTX, S1_KV)}


def expected_speed(g: GPU | None, m: ModelSpec, device: str) -> dict:
    """Debit de generation estime : bande passante / taille des poids x efficacite observee (docs/03)."""
    if g is None or device == "cpu" or not g.bandwidth_gbs:
        return {"tok_s": None, "basis": "CPU : a mesurer (bench)"}
    gb = m.weights_gib * 1.0737  # Gio -> Go
    lo, hi = (0.45, 0.55) if m.id.endswith("-q1") else (0.62, 0.78)   # 1-bit : noyaux moins amortis (docs/03)
    if device == "partial":
        lo, hi = lo * 0.4, hi * 0.6
    return {"tok_s": [round(g.bandwidth_gbs / gb * lo), round(g.bandwidth_gbs / gb * hi)],
            "basis": f"{g.bandwidth_gbs} Go/s / {gb:.2f} Go x efficacite {lo:.2f}-{hi:.2f}"}


def _pick_s1(total_mib: float, override: str) -> ModelSpec:
    if override and override != "auto" and override in MODELS:
        return MODELS[override]
    if total_mib >= 20000:
        return MODELS["ternary-8b"]
    if total_mib >= 11500:
        return MODELS["ternary-4b"]
    return MODELS["ternary-1.7b"]


def _s2_candidates(avail_mib: float, priority: str, override: str) -> list[ModelSpec]:
    if override and override != "auto" and override in MODELS:
        return [MODELS[override]]
    if priority == "vitesse":
        order = ["bonsai-27b-q1", "bonsai-8b-q1"]
    elif avail_mib >= 10500:
        order = ["bonsai2-27b-pq2", "bonsai2-27b-ptq1", "bonsai-27b-q1", "bonsai-8b-q1"]
    else:
        order = ["bonsai2-27b-ptq1", "bonsai-27b-q1", "bonsai-8b-q1"]
    return [MODELS[i] for i in order]


def _kv_type(total_mib: float) -> str:
    return "q4_0" if total_mib < 11500 else ("q8_0" if total_mib < 20000 else "f16")


def cpu_plan(hw: HardwareInfo, priority: str, s2_override: str = "auto", s1_override: str = "auto") -> Plan:
    ram = hw.ram_total_gib
    if s2_override != "auto" and s2_override in MODELS:
        s2m = MODELS[s2_override]
    elif priority != "vitesse" and ram >= 24:
        s2m = MODELS["bonsai-27b-q1"]
    else:
        s2m = MODELS["bonsai-8b-q1"]
    s1m = MODELS[s1_override] if s1_override in MODELS else MODELS["ternary-1.7b"]
    threads = max(1, min(hw.cpu_cores, 8))
    ctx = 8192 if ram >= 12 else 4096
    notes = [f"Pas de GPU NVIDIA exploitable : tout tourne sur le CPU ({hw.cpu_cores} coeurs, {ram:.0f} Gio de RAM).",
             "Les deux modeles vivent en RAM ; comptez ~15-20 tok/s pour le 8B 1-bit sur 8 coeurs AVX2.",
             "Classifieur sur CPU (un slot) : ~0,15-0,4 s par decision une fois l'etat lu, plus le prefill d'un etat neuf (estimation)."]
    if s2m.id != "bonsai-8b-q1":
        notes.append("27B sur CPU : puissant mais lent (~3-6 tok/s) ; la priorite 'vitesse' repasse au 8B.")
    need = (s2m.weights_gib + s2m.overhead_gib + s1m.weights_gib + s1m.overhead_gib) * 1024 + kv_mib(s2m, ctx) + kv_mib(s1m, S1_CTX, S1_KV)
    return Plan("cpu", priority, ServerPlan("s2", s2m.id, "cpu", 0, ctx, 1, "q8_0", "off", 1024, threads),
                ServerPlan("s1", s1m.id, "cpu", 0, S1_CTX, S1_SLOTS_CPU, S1_KV, threads=threads),
                {"ram_needed_mib": round(need), "ram_total_mib": round(ram * 1024)}, need < ram * 1024 * 0.85,
                {"s2": {"tok_s": None, "basis": "CPU"}, "s1_ms": [150, 400]}, notes, title=f"CPU · {s2m.label}")


def make_plan(hw: HardwareInfo, priority: str = "equilibre", s2_override: str = "auto", s1_override: str = "auto",
              ctx_override: int = 0, other_used_mib: float | None = None) -> Plan:
    g = hw.gpu
    if g is None or g.vendor not in ("nvidia", "apple") or g.vram_total_mib < 3000:
        return cpu_plan(hw, priority, s2_override, s1_override)
    backend = "cuda" if g.vendor == "nvidia" else "metal"
    total = float(g.vram_total_mib)
    other = float(g.vram_used_mib if other_used_mib is None else other_used_mib)
    avail = total - other - SAFETY_MIB
    kv = _kv_type(total)
    cap = PRIORITY_CTX_CAP.get(priority, 32768)
    ctxs = [c for c in CTX_STEPS if c <= cap]
    if ctx_override:
        ctxs = [ctx_override]
    s1m = _pick_s1(total, s1_override)
    s1_need = sum(s1_mib(s1m).values())
    mmproj_gpu = total >= 11500
    notes: list[str] = []
    if g.display_active:
        notes.append(f"L'ecran utilise la {g.name} ({other:.0f} Mio deja occupes) ; le brancher sur l'iGPU liberera cette memoire.")

    chosen = None
    for m in _s2_candidates(avail, priority, s2_override):
        fits_ctx = [c for c in ctxs if sum(s2_mib(m, c, kv, mmproj_gpu).values()) <= avail]
        if not fits_ctx:
            continue
        best_alone = fits_ctx[0]
        with_s1 = [c for c in fits_ctx if sum(s2_mib(m, c, kv, mmproj_gpu).values()) + s1_need <= avail]
        # classifieur sur GPU si le contexte reste confortable (>= 16 k, >= 8 k en priorite vitesse) ; en priorite
        # contexte, le classifieur passe sur CPU des que cela libere un palier de contexte
        min_ok = 8192 if priority == "vitesse" else 16384
        s1_gpu = bool(with_s1) and with_s1[0] >= min(min_ok, best_alone) and not (priority == "contexte" and with_s1[0] < best_alone)
        ctx = with_s1[0] if s1_gpu else best_alone
        chosen = (m, ctx, s1_gpu)
        break

    if chosen is None:
        # rien ne tient entierement : dechargement partiel du plus petit candidat pertinent
        m = _s2_candidates(avail, priority, s2_override)[-1]
        per_layer = _gib(m.weights_gib) / S2_LAYERS
        room = avail - _gib(m.overhead_gib) - kv_mib(m, 4096, kv)
        ngl = max(0, min(99, int(room / per_layer)))
        notes.append(f"VRAM insuffisante pour {m.label} entier : {ngl} couches sur GPU, le reste sur CPU (plus lent).")
        s2 = ServerPlan("s2", m.id, "partial" if ngl > 0 else "cpu", ngl, 4096, 1, kv, "off", 1024, threads=hw.cpu_cores)
        s1 = ServerPlan("s1", s1m.id, "cpu", 0, S1_CTX, S1_SLOTS_CPU, S1_KV, threads=hw.cpu_cores)
        budget = {"total": total, "other": other, **s2_mib(m, 4096, kv, False), "s2_weights": per_layer * ngl, "free": 0.0}
        return Plan(backend, priority, s2, s1, budget, False, {"s2": expected_speed(g, m, s2.device), "s1_ms": [150, 400]}, notes,
                    title=f"{g.name} · {m.label} (partiel)")

    m, ctx, s1_gpu = chosen
    mm = "gpu" if mmproj_gpu else "cpu"
    s2 = ServerPlan("s2", m.id, "gpu", 99, ctx, 1, kv, mm if m.mmproj_pattern else "off", 2048 if m.thinking else -1)
    s1 = ServerPlan("s1", s1m.id, "gpu" if s1_gpu else "cpu", 99 if s1_gpu else 0, S1_CTX, S1_SLOTS if s1_gpu else S1_SLOTS_CPU, S1_KV,
                    threads=None if s1_gpu else max(2, min(hw.cpu_cores - 1, 8)))
    parts = s2_mib(m, ctx, kv, mmproj_gpu)
    if s1_gpu:
        parts.update(s1_mib(s1m))
    used = sum(parts.values())
    budget = {"total": total, "other": other, **{k: round(v) for k, v in parts.items()}, "reserve": SAFETY_MIB,
              "free": round(max(0.0, total - other - SAFETY_MIB - used))}
    if m.id.startswith("bonsai2"):
        notes.append(f"{m.label} entierement sur GPU, cache KV {kv} ({ctx // 1024} k tokens de contexte).")
    else:
        notes.append(f"{m.label} : Bonsai 2 ne tient pas avec ce budget ({avail:.0f} Mio) ou la priorite 'vitesse' est choisie.")
    notes.append("Classifieur (System One) sur GPU : ~0,05-0,15 s par decision (estimation GPU, 4 slots a KV unifie)." if s1_gpu else
                 "Classifieur (System One) sur CPU pour laisser le contexte au 27B : un seul slot, questions en sequence sur le meme cache ; "
                 "~0,1-0,3 s par decision une fois l'etat lu, plus la lecture d'un etat neuf (prefill CPU, ~0,5-1 s pour 2 k tokens). "
                 "Estimations : mesurez avec le banc.")
    if mm == "cpu" and m.mmproj_pattern:
        notes.append("Vision : projecteur en RAM (--no-mmproj-offload) : -0,6 Gio de VRAM, images un peu plus lentes a lire.")
    if g.is_blackwell:
        notes.append("Blackwell (sm_120) : runtime CUDA 12.8+ selectionne automatiquement.")
    if m.id in MEASURED:
        notes.append(f"Budget calibre sur cette machine : surcout mesure de {MEASURED[m.id]:.0f} Mio pour {m.label}.")
    title = f"{g.name} · {m.label} · {ctx // 1024}k"
    return Plan(backend, priority, s2, s1, budget, True,
                {"s2": expected_speed(g, m, "gpu"), "s1_ms": [40, 150] if s1_gpu else [100, 350]}, notes, title=title)


def degrade(plan: Plan, installed: set[str] | None = None) -> Plan | None:
    """Cran suivant de l'echelle anti-OOM (docs/03 §4), applique automatiquement par le superviseur."""
    p = replace(plan, notes=list(plan.notes), rung=plan.rung + 1, fits=False)
    s2, s1 = replace(plan.s2), (replace(plan.s1) if plan.s1 else None)
    step = None
    if s2.np > 1:
        s2.np, step = 1, "Bonsai repasse a un seul slot (mode mono)"
    elif s2.mmproj == "gpu":
        s2.mmproj, step = "cpu", "projecteur vision deplace en RAM"
    elif s1 and s1.device == "gpu":
        s1.device, s1.ngl, s1.np, s1.threads, step = "cpu", 0, S1_SLOTS_CPU, 4, "classifieur deplace sur CPU"
    elif s2.ctx > 8192:
        s2.ctx = next(c for c in CTX_STEPS if c < s2.ctx)
        step = f"contexte reduit a {s2.ctx // 1024} k"
    elif s2.kv_type != "q4_0":
        s2.kv_type, step = "q4_0", "cache KV en q4_0"
    elif s2.ctx > 4096:
        s2.ctx, step = 4096, "contexte reduit a 4 k"
    elif s2.model_id.startswith("bonsai2") and (installed is None or "bonsai-27b-q1" in installed):
        s2.model_id, s2.ctx, step = "bonsai-27b-q1", 8192, "passage a Bonsai 27B 1-bit (3,5 Gio)"
    elif s2.ngl > 8:
        s2.ngl = min(s2.ngl, S2_LAYERS) - 8
        s2.device, step = "partial", f"{s2.ngl} couches sur GPU (dechargement partiel)"
    else:
        return None
    p.s2, p.s1 = s2, s1
    p.notes.append(f"Memoire insuffisante au lancement -> {step}.")
    p.title = f"{plan.title.split(' · ')[0]} · {MODELS[s2.model_id].label} · {s2.ctx // 1024}k (ajuste)"
    return p


def s1_on_cpu(plan: Plan) -> Plan:
    """OOM du classifieur sur GPU : il passe directement sur CPU (un slot), sans consommer un cran de degrade() sur Bonsai
    (deja charge : son plan ne change pas)."""
    s1 = replace(plan.s1, device="cpu", ngl=0, np=S1_SLOTS_CPU, threads=plan.s1.threads or 4)
    return replace(plan, s1=s1, budget={k: v for k, v in plan.budget.items() if not k.startswith("s1_")}, fits=False,
                   expected={**plan.expected, "s1_ms": [100, 350]}, rung=plan.rung + 1,
                   notes=[*plan.notes, "Memoire insuffisante pour le classifieur sur GPU -> classifieur deplace sur CPU (Bonsai inchange)."])


def mono_plan(plan: Plan) -> Plan:
    """Classifieur absent au lancement : Bonsai repond aussi aux questions System One (mode mono). Un 2e slot (KV unifie :
    contexte entier par slot) evite que ces lectures evincent le cache de la conversation ou fassent la queue, si la memoire
    le permet : RAM (Bonsai sur CPU) ou VRAM rendue par un classifieur prevu sur GPU. Sur 8 Go avec le classifieur prevu sur
    CPU, aucune marge n'est prevue pour l'etat recurrent d'une 2e sequence du 27B hybride : on garde un slot."""
    if plan.s2.np > 1 or not (plan.s2.device == "cpu" or (plan.s1 is not None and plan.s1.device == "gpu")):
        return plan
    return replace(plan, s2=replace(plan.s2, np=2),
                   notes=[*plan.notes, "Classifieur absent : mode mono, Bonsai prend un 2e slot pour les decisions System One."])
