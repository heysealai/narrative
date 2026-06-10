# The BFS interrogation test (June 10, 2026)

> **Terminology note (post-dates this report):** on June 11, 2026 the node
> formerly called a *midpoint* was renamed **distillant**, and the node's
> cached one-line summary — which this report calls its *distillant* — is now
> its **line**. This report keeps its original vocabulary.

The one recall miss from the Crusoe tracking run — "who do I trust most in the
world?" → `(nothing recalled)` — was the *designed* miss: zero lexical hooks,
the case the `open_memory` BFS tier exists for. The first-draft handoff made
running it the gate on the embeddings decision. Run on June 10, 2026 against a
/tmp copy of `examples/crusoe` (78 leaves; the fixture untouched).

**Verdict: pass. Skeleton + descent answers it in 2 of 6 tool rounds.
Embeddings stay deferred.**

## Protocol

A Claude session played the agent role through the CLI, holding *only* what
`agent.rs::build_system` actually assembles — `narrative profile` +
`narrative map` inline, `narrative project "<question>"` as the would-be
`<recall>` injection, `narrative open <id>` as the `open_memory` tool —
and choosing each descent target only from text already rendered. (Caveat
inherent to the method: the driver model knows the novel. The discipline is
that navigation used only rendered distillants, and the answer content below
is all graph text. The navigation itself required nothing deep — one obvious
top-level pick, then reading.)

## Probe 1 — "who do I trust most?" (the designed case)

- `project` → `(nothing recalled)`, reproducing the report's miss.
- The question is comparative across people, and the skeleton shows
  `people — Family, friends, contacts, counterparties` with five named
  children. **Round 1:** `open_memory(people)`.
- The child-midpoint listing *is* the answer's shortlist — the trust
  vocabulary lives in the distillants: friday "*sworn to serve for life*",
  portuguese-captain "*scrupulously honest benefactor and standing friend*",
  xury "*swore faithfulness, proves loyal*". **Round 2:** open all three in
  parallel (the tool loop handles multiple `tool_uses` per round).
- What surfaces is more than enough to answer, with texture:
  - **Friday**: "a fast friend proved beyond jealousy — 'take kill Friday, no
    send Friday away'" — and the supersession history shows the trust
    *trajectory* (was: "honest, merry, diligent — genuinely loved"). Episodes
    carry the evidence (carried the Spaniard on his back; shot the wolf off
    the guide).
  - **Portuguese captain**: a formal will names him universal heir — trust in
    the strongest legal sense — plus "honest steward and 'old patron'",
    offered his last moidores.
  - **Xury**: loyal, but parted — gone to the captain's service, "a standing
    regret."

An agent answering "Friday — proved beyond jealousy; and the Portuguese
captain holds my whole estate on trust" is fully grounded in retrieved text.

## Probe 2 — "do I have any regrets?" (second zero-hook paraphrase)

- `project` → `(nothing recalled)` again.
- The pinned profile already half-answers inline (`shame-over-repentance`
  axis; temperament distillant "an impulsive runaway hardened into…").
  Descent via `life/origins` + `life/voyages` surfaces "resolved at eighteen
  to go to sea against father's command and mother's entreaties" — the
  original-sin regret, answerable in one round.
- **The instructive partial:** the *Xury parting* regret exists in the graph
  (`people/xury`, "parting with him is a standing regret") but no distillant
  on any path to it mentions regret — `people/xury`'s says "proves loyal."
  Descent can't smell it; only an agent opening xury for other reasons finds
  it.

## What this settles, and the real boundary

The BFS tier does what it was designed to do: **distillants are the scent**,
and where the relevant vocabulary made it into a distillant on the path, a
paraphrase question routes in 1–2 rounds with no vectors anywhere.

The boundary is now precise: content that no distillant on its path
advertises is invisible to descent. That is a *write-time distillation*
problem before it is a retrieval problem — the cheap lever is harvester/
redistill guidance nudging distillants to carry evaluative facets (trust,
regret, fear, pride) alongside topical ones, not an embeddings index. If a
future corpus shows misses even with good distillants, that reopens the
vector question; this test closes it for now.

One honest residual: the driver here was a frontier model. The navigation it
needed (pick `people` for a people question; read five labelled children) is
shallow, but a belt-and-braces run with a small model (haiku) over a real key
would confirm tool-call discipline holds at the low end.
