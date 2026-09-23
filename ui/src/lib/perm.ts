// Modes d'autorisation : memes noms et descriptions dans les Reglages, la zone de saisie et les commandes (/smart...).
import type { PermissionMode } from "./types";

export const PERM: Record<PermissionMode, { label: string; desc: string }> = {
  smart: { label: "Smart", desc: "Le classifieur juge chaque action (~100 ms) et ne demande que si elle est risquee ou incertaine." },
  ask: {
    label: "Toujours demander",
    desc: "Chaque ecriture, commande et lancement du computer use vous est demande ; dans le computer use, seuls les pas juges risques le sont.",
  },
  auto: { label: "Jamais demander", desc: "Aucune question, pas du computer use compris : tout s'execute sans confirmation (a vos risques)." },
};

export const PERM_MODES = Object.keys(PERM) as PermissionMode[];
