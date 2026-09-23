"""Computer use sur le bureau : la meme boucle a deux vitesses que le navigateur (computer_use.py), appliquee aux
fenetres de l'ordinateur via l'arbre d'accessibilite.

  observe() -> PageState : titre de la fenetre active, arbre de controles (role + nom), controles interactifs
               numerotes avec leur rectangle, capture d'ecran optionnelle (pour la vision de Bonsai 2)
  act()     -> clic au centre d'un controle, saisie (ValuePattern, sinon presse-papiers), molette, touches,
               ouverture d'une application
  FastPolicy (clone de Jev) choisit l'action et le controle en une passe (~0,1 s) ; SlowPolicy (Bonsai) reprend
  la main en cas de doute, avec en plus les outils `press_keys` et `open_app`. Chaque clic, saisie, raccourci ou
  lancement est juge par le clone avant execution (StepGuard) ; un pas risque demande votre confirmation.

Backends : Windows UI Automation (paquet `uiautomation`, charge seulement sous Windows) ; `SimulatedDesktop`
pour les tests et les demonstrations. L'arbre d'accessibilite est au bureau ce que l'arbre ARIA est au web : le
clone decide sur du texte, sans lire de pixels, d'ou la vitesse.
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from jev_clone.computer_use import ComputerUseAgent, Element, FastPolicy, PageState, SlowPolicy, run_result
from jev_clone.tools import SystemOneToolbox, _tool

INTERACTIVE_TYPES = {"ButtonControl", "EditControl", "HyperlinkControl", "MenuItemControl", "ListItemControl", "TabItemControl",
                     "CheckBoxControl", "RadioButtonControl", "ComboBoxControl", "TreeItemControl", "SplitButtonControl",
                     "DataItemControl", "SpinnerControl", "SliderControl", "DocumentControl"}
ROLE = {"ButtonControl": "button", "EditControl": "textbox", "HyperlinkControl": "link", "MenuItemControl": "menuitem",
        "ListItemControl": "listitem", "TabItemControl": "tab", "CheckBoxControl": "checkbox", "RadioButtonControl": "radio",
        "ComboBoxControl": "combobox", "TreeItemControl": "treeitem", "SplitButtonControl": "button", "DataItemControl": "cell",
        "SpinnerControl": "spinbutton", "SliderControl": "slider", "DocumentControl": "document", "TextControl": "text",
        "WindowControl": "window", "PaneControl": "pane", "GroupControl": "group", "ToolBarControl": "toolbar",
        "MenuBarControl": "menubar", "ListControl": "list", "TreeControl": "tree", "TabControl": "tablist"}
KEY_NAMES = {"ctrl": "{Ctrl}", "control": "{Ctrl}", "alt": "{Alt}", "shift": "{Shift}", "win": "{Win}", "enter": "{Enter}",
             "return": "{Enter}", "esc": "{Esc}", "escape": "{Esc}", "tab": "{Tab}", "space": "{Space}", "backspace": "{Back}",
             "delete": "{Del}", "del": "{Del}", "up": "{Up}", "down": "{Down}", "left": "{Left}", "right": "{Right}",
             "home": "{Home}", "end": "{End}", "pageup": "{PageUp}", "pagedown": "{PageDown}",
             **{f"f{i}": f"{{F{i}}}" for i in range(1, 13)}}


def keys_to_sendkeys(combo: str) -> str:
    """'ctrl+shift+s' -> '{Ctrl}{Shift}s' (syntaxe SendKeys de uiautomation)."""
    out = []
    for part in combo.lower().replace(" ", "").split("+"):
        if not part:
            continue
        out.append(KEY_NAMES.get(part, part if len(part) == 1 else "{" + part.capitalize() + "}"))
    return "".join(out)


@dataclass
class UINode:
    """Controle d'interface, independant du backend."""
    role: str
    name: str
    value: str = ""
    rect: tuple[int, int, int, int] | None = None   # x, y, largeur, hauteur
    enabled: bool = True
    interactive: bool = False
    depth: int = 0
    handle: Any = None


def walk_uia(root, max_depth: int = 12, max_nodes: int = 400) -> list[UINode]:
    """Parcours en profondeur d'un arbre UI Automation (objets `uiautomation.Control` ou equivalents)."""
    out: list[UINode] = []

    def visit(c, depth: int) -> None:
        if len(out) >= max_nodes or depth > max_depth:
            return
        try:
            if getattr(c, "IsOffscreen", False):
                return
            t = c.ControlTypeName
            r = c.BoundingRectangle
            rect = (int(r.left), int(r.top), int(r.right - r.left), int(r.bottom - r.top)) if r is not None else None
            value = ""
            if t in ("EditControl", "ComboBoxControl", "DocumentControl"):
                try:
                    value = str(c.GetValuePattern().Value or "")[:200]
                except Exception:
                    value = ""
            name = (c.Name or "").strip()
            node = UINode(ROLE.get(t, t.replace("Control", "").lower()), name[:120], value, rect, bool(getattr(c, "IsEnabled", True)),
                          t in INTERACTIVE_TYPES and bool(rect and rect[2] > 0 and rect[3] > 0), depth, c)
            if name or node.interactive or t in ("WindowControl", "PaneControl", "GroupControl", "ToolBarControl", "MenuBarControl"):
                out.append(node)
            for ch in c.GetChildren():
                visit(ch, depth + 1)
        except Exception:
            return

    visit(root, 0)
    return out


def nodes_to_state(title: str, app: str, nodes: list[UINode], max_elements: int = 40, screenshot_b64: str | None = None) -> tuple[PageState, list[UINode]]:
    outline = "\n".join(f"{'  ' * min(n.depth, 8)}- {n.role} \"{n.name}\"" + (f" = {n.value[:60]!r}" if n.value else "") for n in nodes[:180])
    inter = [n for n in nodes if n.interactive and n.enabled][:max_elements]
    elements = [Element(index=i, role=n.role, name=n.name or "(sans nom)", value=n.value, bbox=n.rect) for i, n in enumerate(inter)]
    return PageState(url=f"app://{app}", title=title, aria=outline, elements=elements, screenshot_b64=screenshot_b64), inter


class DesktopSession:
    """Interface commune aux sessions (meme contrat que BrowserSession : observe / act / goto)."""

    def __init__(self, backend: "DesktopBackend", max_elements: int = 40, screenshot: bool = False):
        self.backend, self.max_elements, self.screenshot = backend, max_elements, screenshot
        self._targets: list[UINode] = []
        self.last: PageState | None = None   # derniere observation : les indices des actions s'y rapportent

    def goto(self, target: str) -> None:        # "url" d'un agent de bureau = application a ouvrir
        self.backend.open_app(target)

    def observe(self) -> PageState:
        title, app, nodes = self.backend.snapshot()
        shot = self.backend.screenshot_b64() if self.screenshot else None
        state, self._targets = nodes_to_state(title, app, nodes, self.max_elements, shot)
        self.last = state
        return state

    def act(self, action: dict) -> dict:
        t = action.get("type")
        try:
            if t == "click":
                self.backend.click(self._targets[int(action["target"])])
            elif t == "type":
                self.backend.type_into(self._targets[int(action["target"])], str(action.get("text", "")), bool(action.get("submit")))
            elif t == "scroll_down":
                self.backend.scroll(-5)
            elif t == "go_back":
                self.backend.press("alt+left")
            elif t == "press_keys":
                self.backend.press(str(action["keys"]))
            elif t == "open_app":
                self.backend.open_app(str(action["name"]))
            elif t in ("done", "escalate"):
                return {"ok": True}
            else:
                return {"ok": False, "error": f"action inconnue {t}"}
            self.backend.settle()
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)[:200]}

    def close(self) -> None:
        self.backend.close()


class DesktopBackend:
    def snapshot(self) -> tuple[str, str, list[UINode]]: ...
    def screenshot_b64(self) -> str | None: return None
    def click(self, node: UINode) -> None: ...
    def type_into(self, node: UINode, text: str, submit: bool) -> None: ...
    def scroll(self, amount: int) -> None: ...
    def press(self, combo: str) -> None: ...
    def open_app(self, name: str) -> None: ...
    def settle(self) -> None: ...
    def close(self) -> None: ...


class WindowsUIABackend(DesktopBackend):
    """Windows UI Automation via `uiautomation` (pip install uiautomation). A utiliser dans un seul fil, entoure de
    `uiautomation.UIAutomationInitializerInThread()` (COM est initialise par fil)."""

    def __init__(self):
        if sys.platform != "win32":
            raise RuntimeError("le controle du bureau via UI Automation n'existe que sous Windows")
        import uiautomation as auto  # import tardif : Windows seulement
        self.auto = auto

    def snapshot(self):
        fg = self.auto.GetForegroundControl()
        win = fg.GetTopLevelControl() if hasattr(fg, "GetTopLevelControl") else fg
        name = ""
        try:
            import psutil
            name = psutil.Process(win.ProcessId).name()
        except Exception:
            pass
        return win.Name or "", name, walk_uia(win)

    def screenshot_b64(self):
        try:
            import io
            import mss
            from PIL import Image
            with mss.mss() as s:
                raw = s.grab(s.monitors[1])
                img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
                img.thumbnail((1280, 1280))
                buf = io.BytesIO(); img.save(buf, "JPEG", quality=60)
                return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            return None

    def click(self, node):
        x, y, w, h = node.rect
        self.auto.Click(x + w // 2, y + h // 2)

    def type_into(self, node, text, submit):
        try:
            node.handle.GetValuePattern().SetValue(text)
        except Exception:
            self.click(node)
            self.auto.SetClipboardText(text)
            self.auto.SendKeys("{Ctrl}v", waitTime=0.05)
        if submit:
            self.auto.SendKeys("{Enter}", waitTime=0.05)

    def scroll(self, amount):
        (self.auto.WheelDown if amount < 0 else self.auto.WheelUp)(wheelTimes=abs(amount))

    def press(self, combo):
        self.auto.SendKeys(keys_to_sendkeys(combo), waitTime=0.05)

    def open_app(self, name):
        # jamais de shell : `start "" "{name}"` en shell=True laissait injecter une commande dans le nom
        name = str(name).strip()
        if not name or any(c in name for c in "\r\n\x00"):
            raise ValueError("nom d'application invalide")
        try:
            os.startfile(name)   # type: ignore[attr-defined]  # ShellExecute : applications enregistrees, fichiers, URL
        except OSError:
            subprocess.Popen([name])   # argv sans shell (programme du PATH)
        import time
        time.sleep(1.2)

    def settle(self):
        import time
        time.sleep(0.25)


# ---- bureau simule (tests, demonstrations) ------------------------------------------------------------------------------
@dataclass
class SimWindow:
    title: str
    app: str
    controls: list[dict] = field(default_factory=list)   # {"role", "name", "value", "on_click": callable, "editable"}


class SimulatedDesktop(DesktopBackend):
    """Une calculatrice et un bloc-notes simules : assez pour verifier la boucle observe -> decide -> agit -> verifie."""

    def __init__(self):
        self.display = "0"
        self.notes = ""
        self.saved = False
        self.front = "calc"
        self.log: list[str] = []

    def _windows(self) -> dict[str, SimWindow]:
        def press(ch):
            def f():
                self.display = ch if self.display == "0" else self.display + ch
            return f

        def equals():
            try:
                self.display = str(eval(self.display.replace("×", "*").replace("÷", "/"), {"__builtins__": {}}))  # noqa: S307 (bac a sable de test)
            except Exception:
                self.display = "Erreur"
        calc = SimWindow("Calculatrice", "calc.exe", [{"role": "text", "name": "Affichage", "value": self.display}] +
                         [{"role": "button", "name": ch, "on_click": press(ch)} for ch in "7894561230+-×÷"] +
                         [{"role": "button", "name": "Egal", "on_click": equals},
                          {"role": "button", "name": "Effacer", "on_click": lambda: setattr(self, "display", "0")}])
        notepad = SimWindow("Sans titre - Bloc-notes" if not self.saved else "notes.txt - Bloc-notes", "notepad.exe",
                            [{"role": "document", "name": "Zone de texte", "value": self.notes, "editable": True},
                             {"role": "menuitem", "name": "Fichier"}, {"role": "menuitem", "name": "Enregistrer",
                                                                       "on_click": lambda: setattr(self, "saved", True)}])
        return {"calc": calc, "notepad": notepad}

    def snapshot(self):
        w = self._windows()[self.front]
        nodes = [UINode("window", w.title, depth=0, rect=(0, 0, 800, 600))]
        for i, c in enumerate(w.controls):
            interactive = c["role"] in ("button", "menuitem", "document") and (c.get("on_click") or c.get("editable"))
            nodes.append(UINode(c["role"], c["name"], c.get("value", ""), (10 + 60 * (i % 8), 40 + 50 * (i // 8), 50, 40), True,
                                bool(interactive), 1, c))
        return w.title, w.app, nodes

    def click(self, node):
        self.log.append(f"click:{node.name}")
        if node.handle.get("on_click"):
            node.handle["on_click"]()

    def type_into(self, node, text, submit):
        self.log.append(f"type:{node.name}:{text}")
        if node.handle.get("editable"):
            self.notes += text

    def scroll(self, amount):
        self.log.append(f"scroll:{amount}")

    def press(self, combo):
        self.log.append(f"keys:{combo}")
        if combo.lower() == "ctrl+s" and self.front == "notepad":
            self.saved = True

    def open_app(self, name):
        self.log.append(f"open:{name}")
        self.front = "notepad" if "note" in name.lower() else "calc"

    def settle(self):
        pass


class DesktopSlowPolicy(SlowPolicy):
    """Bonsai pour le bureau : les outils du navigateur + raccourcis clavier + ouverture d'applications."""

    def __init__(self, s2_backend, session: DesktopSession, toolbox: SystemOneToolbox | None = None, **kw):
        super().__init__(s2_backend, session, toolbox, **kw)

        # ces actions peuvent changer de fenetre : on renvoie la nouvelle observation (et on rafraichit les indices),
        # sinon Bonsai viserait les controles de l'ancienne fenetre ; comme clic et saisie, elles sont jugees avant execution
        self.loop.extra["press_keys"] = (_tool("press_keys", "Press a keyboard shortcut in the active window, e.g. 'ctrl+s', 'alt+f4', 'enter'.",
                                               {"keys": {"type": "string"}}, ["keys"]),
                                         lambda a: self.act({"type": "press_keys", "keys": a["keys"]}, observe=True))
        self.loop.extra["open_app"] = (_tool("open_app", "Open an application or a file by name (e.g. 'notepad', 'calc', 'C:/path/file.txt').",
                                             {"name": {"type": "string"}}, ["name"]),
                                       lambda a: self.act({"type": "open_app", "name": a["name"]}, observe=True))


def make_desktop_factory(s1_engine, s2_backend, backend_factory: Callable[[], DesktopBackend] | None = None, vision: bool = False,
                         max_steps: int = 24, confirm: Callable[[str, dict], bool] | None = None,
                         on_event: Callable[[dict], None] | None = None, should_stop: Callable[[], bool] | None = None,
                         ledger: str | Path | None = None):
    """Fabrique l'outil `desktop` de Prophet : run(goal, app=None, slots=None) -> resultat compact.
    confirm : autorisation des pas risques (sans elle, ils sont refuses) ; on_event / should_stop : progression par pas et
    annulation ; ledger : journal des trajectoires (re-entrainement de la politique rapide, training/make_from_trajectories.py)."""
    def factory():
        def run(goal: str, app: str | None = None, slots: dict | None = None) -> dict:
            def go() -> dict:
                backend = backend_factory() if backend_factory else WindowsUIABackend()
                session = DesktopSession(backend, screenshot=vision)
                agent = ComputerUseAgent(session, FastPolicy(s1_engine),
                                         DesktopSlowPolicy(s2_backend, session, SystemOneToolbox(s1_engine), vision=vision), max_steps=max_steps,
                                         ledger=ledger, confirm=confirm, on_event=on_event, should_stop=should_stop, kind="desktop")
                out = agent.run(goal, url=app, slots=slots or {})
                final = session.observe()
                return {**run_result(out), "window": final.title, "app": final.url, "screen": final.aria[:2500]}
            if backend_factory is None and sys.platform == "win32":
                import uiautomation as auto
                with auto.UIAutomationInitializerInThread():
                    return go()
            return go()
        return run
    return factory


def desktop_available() -> tuple[bool, str]:
    if sys.platform != "win32":
        return False, "controle du bureau disponible sous Windows (UI Automation)"
    try:
        import uiautomation  # noqa: F401
        return True, ""
    except Exception:
        return False, "paquet `uiautomation` manquant : pip install uiautomation"


__all__ = ["DesktopSession", "SimulatedDesktop", "WindowsUIABackend", "DesktopSlowPolicy", "make_desktop_factory", "walk_uia",
           "nodes_to_state", "keys_to_sendkeys", "desktop_available"]
