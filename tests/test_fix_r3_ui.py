"""Deuxieme revue (interface, computer use) : regressions corrigees.

Pas refuse puis done(achieved=true) : confirme par le clone, jamais une etiquette done / atteint ; reponse de Bonsai coupee
apres un vrai progres ; appels de Bonsai des outils browse / desktop comptes ; apercu francais d'un pas ; RAM des plans GPU
(dechargement partiel, gros classifieur sur CPU) ; premiere reponse a une autorisation ; tour en cours relu en direct
(rechargement, changement de session) ; interface pilotee par Chromium : carte d'autorisation restauree, titre de la barre
laterale, modes d'autorisation, RAM, mode mono, decimales a la francaise."""

import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from jev_clone.computer_use import describe_action_fr, trajectory_to_examples
from jev_clone.desktop_use import DesktopSession, SimulatedDesktop, make_desktop_factory
from prophet_studio.catalog import CUSTOM, custom_spec
from prophet_studio.hardware import HardwareInfo, fake_gpus
from prophet_studio.planner import make_plan
from prophet_studio.sessions import AgentService, PendingPermission, SessionStore, reduce_event
from tests.conftest import MockS2
from tests.test_fix_computer_use_voice import CLEAR, EQUALS, FIVE, ONE, PLUS, act, agent, guard_on, s1, tc
from tests.test_fix_r2_cu_ui import ROOT, free_port, studio_page, ui_web, wait_until  # noqa: F401 (fixture ui_web)


class Scripted:
    """Bonsai factice : (message, finish_reason) par appel, le dernier en boucle ; timings comme llama-server."""

    def __init__(self, seq, predicted=40):
        self.seq, self.calls, self.predicted = seq, [], predicted

    def chat(self, messages, **kw):
        self.calls.append(kw)
        m, fin = self.seq[min(len(self.calls) - 1, len(self.seq) - 1)]
        return {"choices": [{"message": {"role": "assistant", **m}, "finish_reason": fin}],
                "timings": {"predicted_n": self.predicted + len(self.calls), "prompt_ms": 25.0}}


def click(i, cid="c"):
    return {"content": "", "tool_calls": [MockS2.tool_call("click", {"index": i}, cid)]}, "tool_calls"


CUT = ({"content": "", "reasoning_content": "hmm " * 50}, "length")


# ---- 1. pas refuse puis done(achieved=true) ----------------------------------------------------------------------------------
def _declined(tmp_path, display, decide, replies):
    desk = SimulatedDesktop()
    desk.display = display
    ledger = tmp_path / f"traj-{len(list(tmp_path.glob('traj-*')))}.jsonl"
    eng, _ = s1(decide, guard=guard_on("Effacer"))
    r = make_desktop_factory(eng, MockS2(replies), lambda: desk, confirm=lambda d, j: False, ledger=ledger)()("remets la calculatrice a zero")
    return desk, r, json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])


def test_a_declined_step_then_done_achieved_is_checked_by_the_clone_and_never_a_done_label(tmp_path):
    done = tc("done", {"summary": "La calculatrice est remise a zero.", "achieved": True}, "c2")
    no_done = lambda ex: not any(e["labels"]["action"] == "done" or e["labels"]["achieved"] for e in ex)   # noqa: E731
    # 42 toujours affiche, le clone ne voit pas l'objectif atteint : pas un succes, Bonsai en est averti
    desk, r, traj = _declined(tmp_path, "42", [act("escalate")], [tc("click", {"index": CLEAR}), done])
    assert desk.display == "42" and (r["ok"], r["status"]) == (False, "blocked") and "not accepted" in r["note"]
    assert traj["status"] == "blocked" and traj["records"][0]["done_check"] < 0.6
    ex = trajectory_to_examples(traj)
    assert ex and no_done(ex) and ex[0]["declined"]
    # clic refuse et done dans la meme reponse de Bonsai : meme verdict
    both = {"content": "", "tool_calls": [MockS2.tool_call("click", {"index": CLEAR}, "a"),
                                          MockS2.tool_call("done", {"summary": "remise a zero", "achieved": True}, "b")]}
    _, r2, _ = _declined(tmp_path, "42", [act("escalate")], [both])
    assert (r2["ok"], r2["status"]) == (False, "blocked")
    # objectif deja atteint (0 affiche) et confirme par le clone : succes ; l'etiquette reste escalate (conclu apres un refus)
    _, r3, traj3 = _declined(tmp_path, "0", [act("escalate", achieved=0.9)], [tc("click", {"index": CLEAR}), done])
    assert (r3["ok"], r3["status"]) == (True, "done") and "note" not in r3 and traj3["records"][0]["done_check"] >= 0.6
    assert no_done(trajectory_to_examples(traj3))
    # anciens journaux : done(achieved=true) explicite apres un refus
    st = {"goal": "g", "available_slots": {}, "last_actions": [], "url": "app://calc.exe", "title": "Calculatrice", "page": "",
          "elements": ['[0] button "Effacer"']}
    old = {"records": [{"step": 0, "path": "escalated", "state": st, "slow": {"actions": [
        {"type": "click", "target": 0, "blocked": True}, {"type": "done", "summary": "ok", "achieved": True}]}}]}
    lab = trajectory_to_examples(old)[0]["labels"]
    assert lab["action"] == "escalate" and lab["achieved"] is False


# ---- 2. reponse coupee par max_tokens apres un vrai progres ---------------------------------------------------------------------
def test_a_reply_cut_after_real_progress_is_not_a_bonsai_failure_but_stays_bounded():
    desk = SimulatedDesktop()
    done = ({"content": "", "tool_calls": [MockS2.tool_call("done", {"summary": "6", "achieved": True}, "d")]}, "tool_calls")
    s2 = Scripted([click(ONE, "a"), CUT, click(PLUS, "b"), CUT, click(FIVE, "c"), CUT, click(EQUALS, "e"), done])
    out = agent(desk, s1([act("escalate")])[0], s2, max_steps=20).run("calcule 1+5")
    assert out["status"] == "done" and desk.display == "6" and out["escalations"] == 4
    # coupee sans rien changer a l'ecran (Effacer sur 0) : un echec, arret apres deux reprises comme avant
    out2 = agent(SimulatedDesktop(), s1([act("escalate")])[0], Scripted([click(CLEAR), CUT]), max_steps=20).run("remets a zero")
    assert out2["status"] == "s2_error" and out2["escalations"] == 2
    # Effacer sur 42 (l'ecran change une fois) puis coupee sans progres : borne a trois reprises
    d3 = SimulatedDesktop()
    d3.display = "42"
    out3 = agent(d3, s1([act("escalate")])[0], Scripted([click(CLEAR), CUT]), max_steps=20).run("remets a zero")
    assert out3["status"] == "s2_error" and out3["escalations"] == 3


# ---- 3. appels de Bonsai dans l'outil : comptes pour les statistiques du tour --------------------------------------------------
def test_browse_and_desktop_report_the_bonsai_calls_and_tokens_of_the_run():
    done = ({"content": "", "tool_calls": [MockS2.tool_call("done", {"summary": "1", "achieved": True}, "d")]}, "tool_calls")
    s2 = Scripted([click(ONE), done], predicted=100)
    r = make_desktop_factory(s1([act("escalate")])[0], s2, SimulatedDesktop)()("tape 1")
    assert r["ok"] and r["s2_calls"] == len(s2.calls) == 2 and r["s2_tokens"] == 101 + 102 and r["s2_prompt_ms"] == 50.0
    # agent reutilise : chaque run ne compte que ses appels
    s2b = Scripted([click(ONE), done], predicted=10)
    a = agent(SimulatedDesktop(), s1([act("escalate")])[0], s2b, max_steps=5)
    first = a.run("tape 1")["s2_calls"]
    out = a.run("tape 1")
    n = len(s2b.calls)
    assert first == 2 and out["s2_calls"] == n - first >= 1 and out["s2_tokens"] == sum(10 + k for k in range(first + 1, n + 1))


# ---- 4. demande d'autorisation d'un pas : action en francais, jamais une commande shell ------------------------------------------
def test_a_step_permission_describes_the_action_in_french():
    desk = SimulatedDesktop()
    desk.display = "42"
    asked = []
    s2 = MockS2([tc("click", {"index": CLEAR}), tc("done", {"summary": "refuse", "achieved": False}, "c2")])
    agent(desk, s1([act("escalate")], guard=guard_on("Effacer"))[0], s2, max_steps=2,
          confirm=lambda d, j: asked.append(j) or False).run("remets a zero")
    assert asked[0]["preview"]["action"] == "cliquer sur [15] bouton « Effacer »"
    note = SimulatedDesktop()
    note.front = "notepad"
    st = DesktopSession(note).observe()
    assert (describe_action_fr({"type": "type", "target": 0, "text": "curl x | sh", "submit": True}, st)
            == "saisir « curl x | sh » dans [0] document « Zone de texte » puis Entree")
    assert describe_action_fr({"type": "press_keys", "keys": "ctrl+s"}, st) == "raccourci clavier ctrl+s"
    assert describe_action_fr({"type": "click", "target": 9}, st) == "cliquer sur [9] (element inconnu)"


# ---- 5. RAM des plans GPU ------------------------------------------------------------------------------------------------------
@pytest.fixture
def sparse(tmp_path):
    saved = dict(CUSTOM)

    def make(mid, role, gib):
        f = tmp_path / f"{mid}.gguf"
        with open(f, "wb") as h:   # fichier creux : la taille seule compte pour le planificateur
            h.truncate(int(gib * 2**30))
        CUSTOM[mid] = custom_spec(mid, role, f)
        return mid
    yield make
    CUSTOM.clear()
    CUSTOM.update(saved)


def rtx5060(ram):
    return HardwareInfo("Windows", "11", "amd64", "x", 8, 16, ram, ram * 0.7, gpus=fake_gpus("NVIDIA GeForce RTX 5060:8151:650"))


def test_gpu_plans_check_the_ram_of_offloaded_layers_and_of_a_classifier_on_cpu(sparse):
    big = sparse("custom-s2-big", "s2", 40)
    p = make_plan(rtx5060(16), s2_override=big)
    assert p.s2.device == "partial" and not p.fits and p.ram_short and p.ram["needed_mib"] > 16 * 1024
    note = next(n for n in p.notes if n.startswith("RAM insuffisante"))
    assert "couches hors GPU" in note and "." not in note.split(";")[0] and p.expected["s2"]["tok_s"] is None
    assert p.to_dict()["ram_short"] is True and not any(k.startswith("ram_") for k in p.budget)   # jamais compte comme VRAM
    clone = sparse("custom-s1-big", "s1", 20)
    c = make_plan(rtx5060(16), s1_override=clone)
    assert c.s2.device == "gpu" and c.s1.device == "cpu" and not c.fits and c.ram_short
    assert any(n.startswith("RAM insuffisante") and "classifieur sur CPU" in n for n in c.notes)
    for ram in (16, 32):   # plan de reference (RTX 5060) : jamais de fausse alerte
        d = make_plan(rtx5060(ram))
        assert d.fits and not d.ram_short and d.ram["needed_mib"] < 4096 and not any(n.startswith("RAM insuffisante") for n in d.notes)
    ok = make_plan(rtx5060(64), s2_override=big)   # partiel mais la RAM suffit : lent, pas bloque
    assert ok.s2.device == "partial" and not ok.fits and not ok.ram_short and ok.expected["s2"]["tok_s"]


# ---- 6. autorisation : la premiere reponse fait foi --------------------------------------------------------------------------
def test_the_first_answer_to_a_permission_wins(tmp_path):
    svc = AgentService(SessionStore(tmp_path / "s"), lambda e: None, urls=lambda: ("", ""), settings=lambda: None, ctx=lambda: 8192)
    pp = PendingPermission("sid", "write_file")
    svc.permissions[pp.id] = pp
    assert svc.respond(pp.id, True, True) and not svc.respond(pp.id, False)   # 2e fenetre, double clic, voix + clavier
    assert pp.event.is_set() and (pp.allow, pp.remember) == (True, True) and svc.pending_permissions() == []
    q = PendingPermission("sid", "python")   # Stop deja passe : la demande a expire
    svc.permissions[q.id] = q
    q.event.set()
    assert not svc.respond(q.id, True) and q.allow is False


# ---- 7. tour en cours relu en direct -------------------------------------------------------------------------------------------
@pytest.fixture
def live_studio(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from prophet_studio.config import Paths
    from prophet_studio.server import Studio, build_app
    monkeypatch.setenv("PROPHET_FAKE_GPU", "NVIDIA GeForce RTX 5060:8151:650:display")
    monkeypatch.setenv("FAKE_LLAMA_TPS", "0")
    paths = Paths(tmp_path / "home")
    paths.settings.write_text(json.dumps({"workspace": str(tmp_path / "ws"), "s2_port": free_port(), "s1_port": free_port(),
                                          "voice": {"enabled": False}}))
    st = Studio(paths, "tok-live", 7878, demo=True)
    published, pub = [], st.agent.publish
    st.agent.publish = lambda e: (published.append(e), pub(e))
    try:
        with TestClient(build_app(st), base_url="http://127.0.0.1:7878") as c:
            assert wait_until(lambda: st.runtime.state == "ready", 40)
            yield st, c, published
    finally:
        st.runtime.stop()


def test_a_running_turn_is_served_live_with_its_blocks_and_pending_permission(live_studio):
    st, c, published = live_studio
    H = {"Authorization": "Bearer tok-live"}
    sid = c.post("/api/sessions", headers=H, json={}).json()["id"]
    assert c.get(f"/api/sessions/{sid}/live", headers=H).status_code == 404
    c.post(f"/api/sessions/{sid}/turn", headers=H, json={"text": "Cree une app chrono", "permission_mode": "ask"})
    live = wait_until(lambda: (lambda r: r.status_code == 200 and any(b["type"] == "permission" for b in r.json()["transcript"][-1]["blocks"])
                               and r.json())(c.get(f"/api/sessions/{sid}/live", headers=H)), 30)
    assert live, "aucune demande d'autorisation dans la copie vivante"
    disk = c.get(f"/api/sessions/{sid}", headers=H).json()
    assert disk["running"] and disk["transcript"][-1]["blocks"] == []          # la copie sur disque n'a que la demande
    item = live["transcript"][-1]
    perm = [b for b in item["blocks"] if b["type"] == "permission"]
    pend = st.agent.pending_permissions()
    assert live["running"] and perm[-1]["id"] == pend[0]["id"] and perm[-1]["decision"] is None and perm[-1]["preview"]["diff"]
    assert any(b["type"] == "tool" and b["name"] == "glob" for b in item["blocks"])
    seqs = [e["tseq"] for e in published if e.get("session_id") == sid and "tseq" in e]
    assert seqs == list(range(1, len(seqs) + 1)) and item["tseq"] == seqs[-1]   # l'interface reprend apres item.tseq
    t0 = time.time()
    while sid in st.agent.running and time.time() - t0 < 30:
        for p in st.agent.pending_permissions():
            c.post(f"/api/permissions/{p['id']}", headers=H, json={"allow": False})
        time.sleep(0.05)
    assert c.get(f"/api/sessions/{sid}/live", headers=H).status_code == 404


# ---- 8. aides de l'interface (Node, types effaces) : raisons et decimales en francais, evenements rejoues sans doublon ------------
def _node(work: Path, body: str):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node absent")
    tmp = Path(tempfile.mkdtemp(dir=work))
    lib = tmp / "lib"
    lib.mkdir()
    for f in (ROOT / "ui" / "src" / "lib").glob("*.ts"):   # specificateurs relatifs completes en .ts (resolution ESM de Node)
        src = f.read_text(encoding="utf-8")
        (lib / f.name).write_text(re.sub(r'(from\s+"\./[\w.-]+?)(?<!\.ts)"', r'\1.ts"', src), encoding="utf-8")
    script = tmp / "run.mjs"
    script.write_text('import { pathToFileURL } from "node:url";\n'
                      'const lib = (n) => import(pathToFileURL(`${process.argv[2]}/${n}.ts`).href);\n' + body, encoding="utf-8")
    for flags in ([], ["--experimental-strip-types"]):
        p = subprocess.run([node, *flags, str(script), str(lib)], capture_output=True, text=True, encoding="utf-8", timeout=60)
        if p.returncode == 0:
            return json.loads(p.stdout.strip().splitlines()[-1])
    if "ERR_UNKNOWN_FILE_EXTENSION" in p.stderr or "bad option" in p.stderr:
        pytest.skip(f"Node ne sait pas executer du TypeScript : {p.stderr[-300:]}")
    pytest.fail(p.stderr[-1500:])   # erreur du script (aide absente ou fausse) : un echec, pas un saut


def test_ui_helpers_translate_reasons_and_numbers_and_skip_replayed_events(tmp_path):
    out = _node(tmp_path, r'''
const f = await lib("format");
const t = await lib("transcript");
const item = t.newAssistant("t1");
const evs = [{ type: "text.delta", text: "a", tseq: 1 }, { type: "text.delta", text: "b", tseq: 2 }];
const applied = [...evs, ...evs, { type: "text.delta", text: "c", tseq: 3 }].map((e) => t.reduce(item, e));
console.log(JSON.stringify({
  why: f.cuWhy("action not decisive: done p=0.31, margin 0.01; done not confirmed: goal achieved p=0.34"),
  target: f.cuWhy("target uncertain: None p=0.00, margin 0.00"),
  risky: f.cuWhy('risky step proposed by the fast policy (click [3] button "OK"): double-check it'),
  unknown: f.cuWhy("something new"),
  disk: f.frText("espace disque insuffisant : 12.3 Go necessaires (qwen2.5-7b, 99.5 %)"),
  text: item.blocks.map((b) => b.text).join(""), applied, tseq: item.tseq,
}));
''')
    assert out["why"] == "action indecise : done 31 %, ecart 1 pt · fin non confirmee : objectif atteint 34 %"
    assert out["target"] == "cible incertaine : aucune 0 %, ecart 0 pt"
    assert out["risky"] == "pas risque propose par le classifieur : reexamine par Bonsai" and out["unknown"] == "something new"
    assert out["disk"] == "espace disque insuffisant : 12,3 Go necessaires (qwen2.5-7b, 99,5 %)"
    assert out["text"] == "abc" and out["applied"] == [True, True, False, False, True] and out["tseq"] == 3


# ---- 9. interface (construction temporaire, jamais prophet_studio/web) -----------------------------------------------------------
CARD = ".perm:has(> .head)"   # carte d'autorisation (la pastille du mode, dans la zone de saisie, porte aussi la classe perm)


class Soft(list):
    """Verifications independantes d'un meme ecran : toutes evaluees, un seul echec a la fin qui les liste."""

    def __call__(self, ok, what: str) -> None:
        if not ok:
            self.append(what)


def _toast(page, title, timeout=20000) -> str:
    t = page.locator(".toast", has_text=title).first
    try:
        t.wait_for(timeout=timeout)
    except Exception:
        return ""
    return t.inner_text()


def test_ui_a_reload_or_a_session_switch_mid_turn_keeps_the_blocks_and_the_pending_permission(tmp_path, monkeypatch, ui_web):
    bad = Soft()
    with studio_page(tmp_path, monkeypatch, ui_web, settings={"autostart_models": True, "permission_mode": "ask"}, size=(1440, 900)) as (st, page):
        assert wait_until(lambda: st.runtime.state == "ready", 40)
        page.locator("textarea").wait_for(timeout=15000)
        time.sleep(1.2)   # etat "pret" relaye a l'interface (evenements runtime / metrics)
        page.locator("textarea").fill("Cree une app chrono")
        page.keyboard.press("Enter")
        page.locator(".perm.pending").wait_for(timeout=30000)
        # titre de la barre laterale pendant le 1er tour (et non « Nouvelle session » jusqu'a la fin)
        bad(wait_until(lambda: "Cree une app chrono" in page.locator(".sessions").inner_text()), "titre de la barre laterale")
        tools = page.locator(".tool").count()
        page.reload()
        page.locator(".perm.pending").wait_for(timeout=15000)
        assert page.locator(".perm.pending").count() == 1 and page.locator(".tool").count() == tools
        page.keyboard.press("y")   # la zone de saisie est vide : le raccourci repond a la carte restauree
        assert wait_until(lambda: any(p["tool"] == "python" for p in st.agent.pending_permissions()), 30)
        # changement de session puis retour : la carte en attente revient et repond au clavier
        page.get_by_role("button", name="Nouvelle session").first.click()
        assert wait_until(lambda: page.locator(CARD).count() == 0)
        page.locator(".sessions .row button.open").filter(has=page.locator(".spinner")).click()   # la session du tour en cours
        page.locator(".perm.pending").wait_for(timeout=15000)
        assert page.locator(CARD).count() == 2 and "Autorise" in page.locator(CARD).first.inner_text()
        page.keyboard.press("n")
        assert wait_until(lambda: not st.agent.running, 30)
        assert wait_until(lambda: page.locator(".perm.pending").count() == 0 and page.locator(".perm.denied").count() == 1)
        # mesure : notification en decimales francaises
        page.get_by_role("button", name="Modeles", exact=True).click()
        page.get_by_role("button", name="Mesurer").click()
        body = _toast(page, "Mesure terminee")
        bad(body and not re.search(r"\d\.\d", body), f"mesure en decimales francaises : {body!r}")
        # mode mono : jamais le port ni le temps de chargement du classifieur arrete
        rt = st.runtime.public()
        rt["mono"], rt["state"] = True, "degraded"
        rt["servers"]["s1"].update(state="crashed", port=35849, load_s=0.3)
        st.bus.publish({"type": "runtime.status", "runtime": rt})
        row = page.locator(".srv").nth(1)
        assert wait_until(lambda: "mode mono" in row.inner_text())
        bad(":35849" not in row.inner_text() and "charge en" not in row.inner_text(), f"mode mono : {row.inner_text()!r}")
        page.screenshot(path=str(tmp_path / "models_mono.png"))
    assert not bad, bad


def test_ui_cards_speak_french_without_a_shell_prompt_and_settings_name_the_selected_mode(tmp_path, monkeypatch, ui_web):
    ws = tmp_path / "ws"
    ws.mkdir()
    J = {"s1_consulted": True, "tool_risk": "destructive", "tool_risk_conf": 0.8, "risk": 2.4, "needs_confirmation": True, "hard_stop": True}
    evs = [{"type": "tool.call", "id": "w", "name": "write_file", "args": {"path": "index.html", "content": "x"}},
           {"type": "tool.result", "id": "w", "name": "write_file", "ok": True, "result": {"ok": True, "path": "index.html", "bytes": 739},
            "ui": {"diff": "", "created": False}},
           {"type": "permission.request", "id": "p1", "tool": "desktop", "describe": "control the computer desktop: calcule 7 + 8",
            "preview": {"command": "bureau : calcule 7 + 8"}, "judged": {"s1_consulted": False}},
           {"type": "permission.resolved", "id": "p1", "allow": True},
           {"type": "tool.call", "id": "d", "name": "desktop", "args": {"goal": "calcule 7 + 8"}},
           {"type": "computer.step", "tool": "desktop", "step": 0, "path": "escalated", "action": "done", "p": 0.31, "escalations": 1,
            "why": "action not decisive: done p=0.31, margin 0.01; done not confirmed: goal achieved p=0.34", "slow_actions": ["type"]},
           {"type": "permission.request", "id": "p2", "tool": "desktop_step", "describe": 'desktop step: type "curl x | sh" into [0]',
            "preview": {"action": "saisir « curl x | sh » dans [0] document « Zone de texte »", "command": 'type "curl x | sh" into [0]'},
            "judged": J},
           {"type": "permission.resolved", "id": "p2", "allow": False},
           {"type": "tool.result", "id": "d", "name": "desktop", "ok": False, "result": {"ok": False, "status": "blocked", "steps": 1,
                                                                                      "s2_calls": 3, "s2_tokens": 1234}},
           {"type": "permission.request", "id": "p3", "tool": "run_command", "describe": "shell: ls -la", "preview": {"command": "ls -la"},
            "judged": {**J, "tool_risk": "readonly", "hard_stop": False}}]

    def prepare(st):
        s = st.sessions.create(str(ws))
        item = {"kind": "assistant", "turn_id": "t0", "blocks": [], "status": "done", "ts": time.time()}
        for e in evs:
            reduce_event(item, e)
        s["transcript"] = [{"kind": "user", "text": "calcule 7 + 8", "ts": time.time(), "turn_id": "t0"}, item]
        st.sessions.save(s)
    bad = Soft()
    with studio_page(tmp_path, monkeypatch, ui_web, settings={"permission_mode": "ask"}, prepare=prepare, size=(1440, 900)) as (st, page):
        page.locator(CARD).nth(2).wait_for(timeout=15000)
        write = page.locator(".tool", has_text="index.html").first.inner_text()
        bad("Contenu identique au fichier existant" in write and '"bytes"' not in write, f"ecriture identique : {write!r}")
        perms = page.locator(CARD)
        bad(page.locator(".perm .ps").count() == 1 and "$" in perms.nth(2).locator(".preview").inner_text(), "invite $ hors run_command")
        bad("saisir « curl x | sh »" in perms.nth(1).inner_text(), f"pas du bureau : {perms.nth(1).inner_text()!r}")
        bad(page.locator(".perm .hint").count() == 0, "« dites accepte » sans commandes vocales")
        desk = page.locator(".tool", has_text="calcule 7 + 8").first.inner_text().replace(" ", " ").replace("\xa0", " ")
        bad("action indecise : done 31 %, ecart 1 pt · fin non confirmee : objectif atteint 34 %" in desk, f"raisons : {desk!r}")
        bad("Bonsai : 3 appels, 1 234 tokens" in desk, "appels de Bonsai dans l'outil")
        bad("Toujours demander" in page.locator(".composer .perm .pill").inner_text(), "nom du mode dans la zone de saisie")
        page.get_by_role("button", name="Reglages", exact=True).click()
        row = page.locator(".row", has_text="Autorisations").first
        row.wait_for(timeout=10000)
        bad("Chaque ecriture, commande et lancement du computer use vous est demande" in row.inner_text(), f"mode decrit : {row.inner_text()!r}")
        bad([b.inner_text() for b in row.locator(".seg button").all()] == ["Smart", "Toujours demander", "Jamais demander"], "noms des modes")
        page.screenshot(path=str(tmp_path / "settings.png"))
    assert not bad, bad


def test_ui_gpu_plans_short_of_ram_are_never_applied_blindly_and_install_errors_are_french(tmp_path, monkeypatch, ui_web):
    huge, mid = tmp_path / "huge-brain.gguf", tmp_path / "mid-brain.gguf"
    ids = {}

    def prepare(st):
        st.hw.ram_total_gib = 16.0
        for f, gib in ((huge, 40), (mid, 10)):
            with open(f, "wb") as h:   # fichiers creux
                h.truncate(gib * 2**30)
        ids["huge"] = st.installer.import_gguf(str(huge), "s2")
        ids["mid"] = st.installer.import_gguf(str(mid), "s2")

        def full(*a, **kw):
            raise OSError("espace disque insuffisant : 12.3 Go necessaires")
        st.installer.install_model = full

    def choose(st, mid_):   # meme chemin que PUT /api/settings : nouveau plan publie
        st.settings.update({"s2_model": mid_})
        st.bus.publish({"type": "settings.changed", "settings": st.settings.get().model_dump(), "plan": st.plan().to_dict()})
    bad = Soft()
    with studio_page(tmp_path, monkeypatch, ui_web, settings={"autostart_models": True}, prepare=prepare) as (st, page):
        assert wait_until(lambda: st.runtime.state == "ready", 40)   # plan automatique en marche
        page.get_by_role("button", name="Modeles", exact=True).click()
        page.get_by_role("button", name="Redemarrer").wait_for(timeout=15000)
        # cerveau importe de 40 Gio : partiel sur la RTX 5060, ~36 Gio hors GPU pour 16 Gio de RAM
        choose(st, ids["huge"])
        assert st.plan().s2.device == "partial"
        warn = page.locator(".warnbox.ram")
        bad(wait_until(lambda: warn.count() == 1, 5) and "RAM insuffisante" in warn.inner_text()
            and warn.get_by_role("button", name="Lancer quand meme").count() == 1, "plan partiel sans assez de RAM : pas d'avertissement")
        bad(page.get_by_role("button", name="Appliquer et redemarrer").count() == 0, "« Appliquer et redemarrer » propose malgre la RAM")
        bad(page.get_by_role("button", name="Redemarrer", exact=True).is_disabled(), "Redemarrer actif malgre la RAM")
        # partiel mais la RAM suffit : dit clairement, jamais « Appliquer et redemarrer » sans reserve
        choose(st, ids["mid"])
        vram = page.locator(".warnbox.vram")
        bad(wait_until(lambda: vram.count() == 1, 5) and "VRAM insuffisante" in vram.inner_text() and page.locator(".warnbox.ram").count() == 0,
            "plan partiel : VRAM insuffisante non dite")
        bad(page.get_by_role("button", name="Appliquer et redemarrer").count() == 0
            and page.get_by_role("button", name="Appliquer quand meme (plus lent)").count() == 1, "plan partiel applique sans reserve")
        # erreur d'installation : decimales a la francaise
        page.locator(".cat .model").get_by_role("button", name="Installer").first.click()
        body = _toast(page, "impossible")
        bad("12,3 Go" in body and "12.3" not in body, f"erreur d'installation : {body!r}")
    assert not bad, bad
