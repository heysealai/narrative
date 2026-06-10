# Haiku BFS re-run (June 10, 2026)

> **Terminology note (post-dates this report):** on June 11, 2026 the node
> formerly called a *midpoint* was renamed **distillant**, and the node's
> cached one-line summary — which this report calls its *distillant* — is now
> its **line**. This report keeps its original vocabulary.

The belt-and-braces item from the BFS interrogation test: its driver was a
frontier model, so tool-call discipline at the low end stayed unverified.
This run put **claude-haiku-4-5** in the driver's seat — via Claude Code
subagents rather than a raw API key (user-approved substitution), which
shapes what it can and cannot prove (see *Boundaries* below).

**Verdict: the BFS gate holds at the low end. Navigation discipline and
schema discipline both pass. Where the small model visibly costs is
write-time judgment — entity linking and rule-13 facet followthrough — which
is an argument for spending model quality on the harvester, not the reader.**

## Protocol

Three independent haiku subagents, clean contexts, on a fresh /tmp copy of
`examples/crusoe` (pre-facet-fix fixture, as the original BFS run had it).
Each held only what the real role holds:

- **Agent role** (probes 1 and 2): the `agent.rs::build_system` text with
  profile + map inline, the probe as the user message, no recall block
  (`project` reproduced `(nothing recalled)` for both first), and
  `narrative open <id>` as the only permitted command, max 6 — the
  `open_memory` contract verbatim, Bash-mediated.
- **Harvester role**: the eleven rules verbatim, the section/field spec in
  prose, the real `harvest-prompt` output for a novel turn (tobacco blight;
  partner trust), output to a JSON file then applied with the real
  `deny_unknown_fields` parser.

Nothing in the prompts named targets, the test's purpose, or the expected
answers. Every claim in the answers was verified afterwards against the
exact `open` renders the agent saw.

## Probe 1 — "who do I trust most in the world?"

4 opens: `people` → `people/friday`, `people/portuguese-captain`,
`people/parents`. Same shape as the frontier run (top-level pick, then the
trust-bearing children; it chose parents where the frontier chose xury).
Answer: **Friday first, the Portuguese captain second** — matching the
frontier run — with the will/universal-heir, "proved beyond jealousy",
wolf-off-the-guide and Stoic-smile texture all present. Every checked
detail traced to rendered text, including the bear episode I suspected was
a training-data leak (it is a real episode under `people/friday`). Fully
grounded; zero leaks detected.

## Probe 2 — "do I have any regrets?"

2 opens: `temperament`, `life/voyages`. The interesting one. Opening
`temperament` (a profile midpoint — legal, it is rendered inline) was
actually a *good* move the frontier run didn't make: its episode list is
where the Friday-jealousy regret lives ("weeks of jealousy... wronged 'the
poor honest creature' utterly"), and haiku surfaced it as a genuine
regret — grounded, not fabricated. Weaknesses against the frontier run:

- It read `seafaring-resolve` ("resolved at eighteen to go to sea against
  father's command") in the `life/voyages` render but did not name the
  father-defiance regret explicitly — only alluded ("failures and
  shipwrecks... integrated"). The frontier run led with it.
- One mild conflation: the "saved from something worse" providence pattern
  (the refused homeward ships / "secret hint vindicated" episode, rendered
  in both its opens) got attached to the Guinea wreck instead.
- The Xury regret stayed invisible — as designed on this pre-facet fixture
  (`people/xury` still says "proves loyal"); reproduces the boundary the
  facet guidance now fixes at write time.

## Harvester role — schema pass, judgment findings

The composed ops: valid JSON, all eleven sections, correct field shapes and
enums; **applied with zero errors**. Two episodes, three states (one a
`supports` with a sensible target), proper nulls. The judgment findings,
all low-end-flavored:

- **Entity misattribution**: tagged the Brazil *partner* as
  `people/portuguese-captain` — a different person. Plausible-but-wrong
  linking of the kind a frontier harvester is unlikely to make.
- **Rule 3 miss**: no midpoint for the partner (a durable participant of
  thirty years — the session-3 frontier run created `people/widow` in the
  analogous situation).
- **Rule 13 miss**: an explicit trust fact produced no facet-bearing
  distill; the facet guidance went unheeded.
- **Alias noise**: "honest dealing", "thirty years" are not referring
  expressions; they landed as routing terms on the (wrong) midpoint.

## Boundaries of this run

Subagent-mediated, so two things remain genuinely unverified on haiku:
the raw API `tool_use` loop (native tool blocks, thinking interplay,
MAX_TOOL_ROUNDS behavior) and **grammar-constrained decoding** against the
sectioned schema — the path where `anyOf` degeneration was originally
observed. Those need a real key; `agent.rs` now prints
`↳ open_memory(<id>)` descent traces so that run is observable when it
happens. And as ever: haiku knows the novel — the defense was post-hoc
verification of every claim against rendered text, which it survived.

## What this settles

The retrieval side of the design is robust down-market: a small model given
skeleton + distillants navigates correctly, stays in budget, grounds its
answers, and even finds paths the frontier driver didn't (episode tags via
a profile midpoint). The write side is where model quality buys real
correctness — entity linking, midpoint judgment, facet followthrough. If a
host ever splits model assignments, put the cheap model on the agent role
and the good one on the harvester, not the other way around.
