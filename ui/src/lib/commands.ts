// Commandes : barre oblique (/), palette (Ctrl+K) et commandes vocales partagent les memes actions.
import { app } from "./store.svelte";
import { speaker, toggleHandsfree } from "./voice";

export interface Command {
  id: string;
  title: string;
  hint?: string;
  slash?: string;
  keys?: string;
  group: "Session" | "Agent" | "Voix" | "Moteur" | "Navigation";
  run: (arg?: string) => unknown;
}

export const commands: Command[] = [
  { id: "new_session", title: "Nouvelle session", slash: "/nouveau", keys: "Ctrl N", group: "Session", run: () => app.newSession() },
  { id: "stop", title: "Arreter la generation", slash: "/stop", keys: "Echap", group: "Agent", run: () => (speaker.stop(), app.cancel()) },
  { id: "plan_on", title: "Mode plan (lecture seule, propose un plan)", slash: "/plan", keys: "Maj Tab", group: "Agent", run: () => (app.planMode = !app.planMode) },
  { id: "effort_deep", title: "Reflexion profonde pour le prochain message", slash: "/profond", group: "Agent", run: () => (app.effortOnce = "deep") },
  { id: "effort_fast", title: "Reponse rapide pour le prochain message", slash: "/rapide", group: "Agent", run: () => (app.effortOnce = "fast") },
  { id: "perm_smart", title: "Autorisations : le classifieur decide (smart)", slash: "/smart", group: "Agent", run: () => app.updateSettings({ permission_mode: "smart" }) },
  { id: "perm_ask", title: "Autorisations : toujours demander", slash: "/demander", group: "Agent", run: () => app.updateSettings({ permission_mode: "ask" }) },
  { id: "perm_auto", title: "Autorisations : ne jamais demander", slash: "/auto", group: "Agent", run: () => app.updateSettings({ permission_mode: "auto" }) },
  { id: "files", title: "Afficher les fichiers de l'espace de travail", slash: "/fichiers", keys: "Ctrl B", group: "Navigation", run: () => (app.inspectorOpen = !app.inspectorOpen) },
  { id: "open_models", title: "Modeles et materiel", slash: "/modeles", group: "Navigation", run: () => (app.view = "models") },
  { id: "open_settings", title: "Reglages", slash: "/reglages", keys: "Ctrl ,", group: "Navigation", run: () => (app.view = "settings") },
  { id: "chat", title: "Retour a la conversation", group: "Navigation", run: () => (app.view = "chat") },
  { id: "wizard", title: "Assistant d'installation", group: "Moteur", run: () => (app.wizardOpen = true) },
  { id: "bench", title: "Mesurer les performances (tok/s, latence du classifieur)", slash: "/bench", group: "Moteur", run: () => app.runBench() },
  { id: "restart", title: "Redemarrer les modeles", slash: "/redemarrer", group: "Moteur", run: () => app.startRuntime() },
  { id: "read_last", title: "Lire la derniere reponse a voix haute", slash: "/lire", group: "Voix", run: () => readLast() },
  { id: "handsfree", title: "Mains libres (mot d'eveil)", slash: "/mainslibres", group: "Voix", run: () => toggleHandsfree() },
  { id: "mute", title: "Couper le micro et la voix", group: "Voix", run: () => (toggleHandsfree(false), speaker.stop()) },
  { id: "clear_input", title: "Effacer la saisie", slash: "/effacer", group: "Session", run: () => (app.composerText = "") },
];

export function readLast() {
  const a = app.lastAssistant;
  const text = a?.response || a?.blocks.filter((b) => b.type === "text").map((b) => (b as { text: string }).text).join(" ");
  if (text) speaker.speak(text);
}

export async function runVoiceCommand(cmd: string) {
  switch (cmd) {
    case "approve":
    case "deny": {
      const p = app.pendingPermission;
      if (p) await app.respondPermission(p.id, cmd === "approve");
      else app.toast("info", "Aucune action en attente d'autorisation");
      return;
    }
    case "confirm_approve":
    case "confirm_deny": {
      // le classifieur (et non la grammaire exacte) a compris une reponse a l'autorisation : on n'agit pas, on demande la phrase exacte
      const word = cmd === "confirm_approve" ? "accepte" : "refuse";
      if (app.pendingPermission) app.toast("warn", "Confirmation vocale requise", `Dites exactement « ${word} » ou utilisez les boutons de la demande.`);
      else app.toast("info", "Aucune action en attente d'autorisation");
      return;
    }
    case "send":
      if (app.composerText.trim()) {
        const t = app.composerText;
        app.composerText = "";
        await app.send(t);
      }
      return;
    case "plan_on":
      app.planMode = true;
      return;
    case "plan_off":
      app.planMode = false;
      return;
  }
  await commands.find((c) => c.id === cmd)?.run();
}

export function matchSlash(text: string): Command[] {
  const q = text.slice(1).toLowerCase();
  return commands.filter((c) => c.slash && (c.slash.slice(1).startsWith(q) || c.title.toLowerCase().includes(q))).slice(0, 8);
}

export function fuzzy(q: string, s: string): number {
  q = q.toLowerCase();
  s = s.toLowerCase();
  if (!q) return 1;
  if (s.includes(q)) return 2 + q.length / s.length;
  let i = 0;
  for (const ch of s) if (ch === q[i]) i++;
  return i === q.length ? 1 : 0;
}
