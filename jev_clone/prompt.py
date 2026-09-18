"""Rendu (etat, question) -> texte, avec un prefixe d'etat strictement partage.

Disposition d'une requete (ChatML, format d'entrainement des modeles Qwen / Bonsai) :

    prefixe  = <|im_start|>system ... <|im_end|>
               <|im_start|>user
               # State
               <etat>

    branche_i = # Question
                <instructions>
                # Options
                A. ...
                B. ...
                Respond with only the letter.<|im_end|>
                <|im_start|>assistant
                <think>\n\n</think>\n\n        (prefixe "sans reflexion" des modeles Qwen3.x hybrides)

Le prefixe est identique octet pour octet entre les branches : llama-server reutilise son cache
(cache_prompt) et ne traite que la branche. Le modele ne genere jamais : on lit les probabilites
du token suivant, restreintes aux etiquettes (A/B/C..., Yes/No) par une grammaire.
"""

from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass, field
from typing import Any

from jev_clone.schema import MAX_LETTER_OPTIONS, ChoiceQuestion, NoulQuestion, ScoreQuestion, Text

SYSTEM_PROMPT = (
    "You are a System One decision model. You read the State and answer each Question "
    "by choosing exactly one of the listed options. You never explain. You answer with "
    "the single option label only."
)

LETTERS = string.ascii_uppercase
YES, NO = "Yes", "No"


def render_text(x: Text) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False, indent=2)


@dataclass
class Branch:
    qid: str
    kind: str                 # "noul" | "choice" | "score"
    text: str                 # texte de la branche (apres le prefixe)
    labels: list[str]         # etiquettes dont on lit la probabilite, dans l'ordre du prompt
    keys: list[Any] = field(default_factory=list)  # keys[i] = option / niveau / booleen designe par labels[i]
    terminator: str = ""      # "" : etiquettes mono-token (lettres, Yes/No) ; '"' : noms d'options entre guillemets


@dataclass
class PromptFormat:
    chat: bool = True          # ChatML (modeles instruct) ; False = texte brut (modeles base)
    no_think: bool = True      # Qwen3.x hybrides : bloc <think> vide pour couper la reflexion
    system_prompt: str = SYSTEM_PROMPT

    def prefix(self, state: Text) -> str:
        body = f"# State\n{render_text(state)}\n\n"
        if not self.chat:
            return f"{self.system_prompt}\n\n{body}"
        return f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n{body}"

    def branch(self, body: str, opening: str = "") -> str:
        """`opening` est ajoute apres l'amorce assistant (ex. un guillemet ouvrant en mode valeurs)."""
        if not self.chat:
            return f"{body}Answer:{opening}"
        tail = "<|im_end|>\n<|im_start|>assistant\n"
        if self.no_think:
            tail += "<think>\n\n</think>\n\n"
        return f"{body}{tail}{opening}"


def _options_block(instructions: Text, labelled: list[tuple[str, str]], ask: str) -> str:
    lines = [f"# Question\n{render_text(instructions)}\n", "# Options"]
    for label, desc in labelled:
        lines.append(f"{label}. {desc}" if desc else f"{label}.")
    lines.append(f"\n{ask}\n")
    return "\n".join(lines)


def _value_branch(qid: str, q: ChoiceQuestion, fmt: PromptFormat, order: list[str]) -> Branch:
    """Mode valeurs (> 26 options, ou label_mode="values") : on lit la probabilite du NOM de chaque option,
    ecrit entre guillemets ; la grammaire n'autorise que `nom"`, et les options qui partagent un prefixe sont
    departagees par des lectures de continuation (backend). Meme principe que Jev (255 options)."""
    opts = q.options()
    lines = [f"# Question\n{render_text(q.instructions)}\n", "# Options (answer with the exact option name, in quotes)"]
    for k in order:
        d = render_text(opts[k])
        lines.append(f'- "{k}"' + (f": {d}" if d else ""))
    lines.append("\nRespond with only the option name, in quotes.\n")
    return Branch(qid, "choice", fmt.branch("\n".join(lines), opening='"'), list(order), list(order), terminator='"')


def build_branches(qid: str, q, fmt: PromptFormat, permutations: int = 1,
                   rng: random.Random | None = None, label_mode: str = "auto") -> list[Branch]:
    """1..permutations branches pour une question (noul n'est jamais permute)."""
    rng = rng or random.Random(0)

    if isinstance(q, NoulQuestion):
        t = render_text(q.criteria.true) if q.criteria else ""
        f = render_text(q.criteria.false) if q.criteria else ""
        labelled = [(YES, t or "The statement is true."), (NO, f or "The statement is false.")]
        body = _options_block(q.instructions, labelled, "Respond with only Yes or No.")
        return [Branch(qid, "noul", fmt.branch(body), [YES, NO], [True, False])]

    if isinstance(q, ChoiceQuestion):
        opts = q.options()
        keys = list(opts.keys())
        kind, ask = "choice", "Respond with only the letter of the best option."

        def desc_of(k):
            d = render_text(opts[k])
            return f"{k}: {d}" if d else k
    else:
        assert isinstance(q, ScoreQuestion)
        keys = list(range(len(q.criteria)))
        kind, ask = "score", "Respond with only the letter of the level that best matches."

        def desc_of(k):
            return f"(level {k} of {len(keys) - 1}) {render_text(q.criteria[k])}"

    values = isinstance(q, ChoiceQuestion) and (label_mode == "values" or (label_mode == "auto" and len(keys) > MAX_LETTER_OPTIONS))
    out: list[Branch] = []
    for p in range(permutations):
        order = list(keys)
        if p > 0:
            rng.shuffle(order)
        if values:
            out.append(_value_branch(qid, q, fmt, order))
            continue
        labels = [LETTERS[i] for i in range(len(order))]
        labelled = [(lab, desc_of(k)) for lab, k in zip(labels, order)]
        out.append(Branch(qid, kind, fmt.branch(_options_block(q.instructions, labelled, ask)), labels, order))
    return out


def label_grammar(labels: list[str], terminator: str = "") -> str:
    """Grammaire GBNF qui n'autorise que les etiquettes (suivies du terminateur eventuel) : le sampler de
    llama.cpp masque tout le reste et `post_sampling_probs` renvoie la distribution renormalisee."""
    return "root ::= " + " | ".join(json.dumps(l + terminator) for l in labels)
