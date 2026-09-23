const nf1 = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 1 });
const nf0 = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 0 });

export const num = (x: number, digits = 0) => (digits ? nf1 : nf0).format(x);

export function bytes(b: number): string {
  if (!b) return "0 o";
  const u = ["o", "Ko", "Mo", "Go", "To"];
  const i = Math.min(u.length - 1, Math.floor(Math.log(b) / Math.log(1000)));
  return `${nf1.format(b / 1000 ** i)} ${u[i]}`;
}

export const gb = (x: number) => `${nf1.format(x)} Go`;
export const mibToGib = (m: number) => `${nf1.format(m / 1024)} Gio`;

export function ms(x: number | null | undefined): string {
  if (x == null) return "—";
  if (x < 1000) return `${nf0.format(x)} ms`;
  if (x < 60_000) return `${nf1.format(x / 1000)} s`;
  const m = Math.floor(x / 60_000);
  return `${m} min ${nf0.format((x % 60_000) / 1000)} s`;
}

export function eta(s: number | null | undefined): string {
  if (s == null || !isFinite(s)) return "";
  if (s < 60) return `${Math.round(s)} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
  return `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min`;
}

export const pct = (x: number | null | undefined) => (x == null ? "—" : `${nf0.format(x * 100)} %`);

export function ago(ts: number): string {
  const d = Date.now() / 1000 - ts;
  if (d < 60) return "a l'instant";
  if (d < 3600) return `il y a ${Math.floor(d / 60)} min`;
  if (d < 86400) return `il y a ${Math.floor(d / 3600)} h`;
  const days = Math.floor(d / 86400);
  return days === 1 ? "hier" : `il y a ${days} j`;
}

export const ctxLabel = (c: number) => (c >= 1024 ? `${Math.round(c / 1024)} k` : `${c}`);

const nfd = new Map<number, Intl.NumberFormat>();
/** Exactement `d` decimales, virgule francaise (num() s'arrete a une decimale). */
export function fixed(x: number, d: number): string {
  let f = nfd.get(d);
  if (!f) nfd.set(d, (f = new Intl.NumberFormat("fr-FR", { minimumFractionDigits: d, maximumFractionDigits: d })));
  return f.format(x);
}

// ---- computer use (browse / desktop) : voie d'un pas et statut de fin, en clair ----
const CU_PATH: Record<string, string> = { fast: "voie rapide", escalated: "Bonsai", escalated_after_verify: "Bonsai apres verification", blocked: "refuse" };
export const cuPath = (p: string | null | undefined) => (p ? (CU_PATH[p] ?? p) : "—");
const CU_STATUS: Record<string, string> = {
  done: "objectif atteint",
  not_achieved: "objectif non atteint",
  blocked: "pas refuse : objectif non atteint",
  cancelled: "annule",
  max_steps: "limite de pas atteinte",
  max_escalations: "limite d'escalades atteinte",
  s2_error: "Bonsai en echec",
  needs_reasoning: "raisonnement requis (Bonsai absent)",
};
export const cuStatus = (s: string | null | undefined) => (s ? (CU_STATUS[s] ?? s) : "");

const pts = (m: number) => {
  const v = Math.round(m * 100);
  return `${num(v)} pt${Math.abs(v) > 1 ? "s" : ""}`;
};
/** Raison d'une escalade (texte anglais de jev_clone/computer_use.py, lu par Bonsai et le journal) -> francais, nombres au
 *  format francais ; les noms d'actions (click, done...) restent ceux des pas. Une partie inconnue (ou coupee a 200
 *  caracteres) reste telle quelle. */
export function cuWhy(why: string | null | undefined): string {
  if (!why) return "";
  if (why.startsWith("risky step proposed by the fast policy")) return "pas risque propose par le classifieur : reexamine par Bonsai";
  if (why.startsWith("fast policy unavailable for verification")) return "classifieur indisponible pour la verification";
  if (why.startsWith("fast policy unavailable")) return "classifieur indisponible : Bonsai prend le pas";
  const P = "p=([\\d.]+), margin (-?[\\d.]+)";
  return why.split("; ").map((part) => {
    let m: RegExpExecArray | null;
    if (part === "the fast policy asked for help") return "le classifieur demande de l'aide";
    if (part === "two consecutive low-confidence verifications") return "deux verifications peu sures d'affilee";
    if ((m = new RegExp(`^action not decisive: (\\S+) ${P}$`).exec(part))) return `action indecise : ${m[1]} ${pct(+m[2])}, ecart ${pts(+m[3])}`;
    if ((m = new RegExp(`^target uncertain: (\\S+) ${P}$`).exec(part)))
      return `cible incertaine : ${m[1] === "None" ? "aucune" : `[${m[1]}]`} ${pct(+m[2])}, ecart ${pts(+m[3])}`;
    if ((m = new RegExp(`^no confident slot to type \\((.*) ${P}\\)$`).exec(part)))
      return `valeur a saisir incertaine : ${m[1] === "None" || m[1] === "none" ? "aucune" : m[1]} ${pct(+m[2])}, ecart ${pts(+m[3])}`;
    if ((m = /^done not confirmed: goal achieved p=([\d.]+)$/.exec(part))) return `fin non confirmee : objectif atteint ${pct(+m[1])}`;
    return part;
  }).join(" · ");
}

/** Messages du coeur (erreurs d'installation, de telechargement) : decimales a la francaise pour un nombre suivi d'une unite
 *  (« 12.3 Go » -> « 12,3 Go »), meme nombre de decimales. Un identifiant ou une version sans unite reste intact. */
export const frText = (s: string | null | undefined): string =>
  (s ?? "").replace(/(\d+)\.(\d+)(?=\s?(?:(?:[GMKT]i?[oB]|ms|s|min|h|tok\/s)(?![A-Za-z])|%))/g, (_, a: string, b: string) => fixed(Number(`${a}.${b}`), b.length));

export function shortPath(p: string, max = 42): string {
  if (!p || p.length <= max) return p;
  const parts = p.split(/[\\/]/);
  let out = parts.pop() ?? "";
  while (parts.length && out.length < max - 6) out = parts.pop() + "/" + out;
  return "…/" + out;
}
