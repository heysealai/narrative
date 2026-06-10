# Crusoe rebuild: the full book through the drift-aware engine

**June 11, 2026.** The original tracking experiment (`crusoe-tracking-report.md`)
was re-run from scratch on the current engine — post-rename (distillant/line),
with derived facet routing and drift detection live. This is the run that
produced the checked-in `examples/crusoe/memory.json`.

## Method

Same shape as the original: the full Gutenberg text (120,920 words), split at
paragraph boundaries into **66 chunks of ~1,830 words**, fed in order through
the keyless CLI. The harvester role was played by Claude Sonnet subagents (six
chunks per agent, sequential — each harvest's comparanda depend on the graph
the previous one left). Per chunk: `harvest-prompt` → compose ops → `apply`;
when apply hinted `⚙ consolidation due`, the agent performed exactly one
redistill cycle — the engine's own one-step-per-turn pacing. After the last
chunk, **8 settling steps** drained the remaining queue (the idle-host
stand-in). 66 harvests, 75 redistill cycles total. `occurred_at` stayed null
throughout (story time is the 1660s; u64 seconds can't represent it).

## Resulting graph

| | original run | rebuild |
|---|---|---|
| distillants | 22 (19 registry / 3 profile) | **50 (44 / 6)** |
| leaves | 78 | **294** |
| episodes | 172 | **339** |

The richer graph is mostly harvester yield (sonnet wrote more, smaller leaves)
plus consolidation actually keeping up: the tree went three levels deep where
the material demanded it (`life/island/savages/rescues`,
`life/island/escape/boats`, `people/friday/faith`,
`temperament/{practical,religion,wandering}`).

## Drift in the wild

This was the first sustained run with drift live, and it behaved as designed:

- Drift appeared on the very first batch (`people: drift 1` — a child line
  changed after the parent's pass) and surfaced continuously after
  supersessions and child rewrites; redistills consumed it.
- **Children-before-parents starves a fat parent under constant inflow.**
  `life/island` reached 48 direct leaves / pressure ~40 while the queue kept
  consolidating its *due children* first — new material landed in the children
  every chunk, so they kept outranking the parent. The big split (34 moves
  into `savages` + `escape`) only fired once the children quieted (chunk 051),
  and the remainder (`milestones` + `mutiny`) at chunk 056. Correct by design
  (a parent rewrite must not consume stale child lines), but it means a
  sustained-ingest host should expect fat parents to resolve late, when
  inflow pauses. The settling phase exists for exactly this.

## Interrogation: 20 questions

Two-stage probe per question on a copy of the final graph: mechanical
`project` first, then the agent role (map + BFS `open` descent, ≤5 opens),
with the hard rule that any claim not seen in a render is contamination.
Sonnet subagents; 8 answer claims spot-verified verbatim against the stored
graph afterwards (all 8 present — e.g. "unspotted integrity" on
`captain-widow-relief`, "twenty-eight years, two months" on
`island-departure-date`).

**Score: 17 GROUNDED-FULL, 3 GROUNDED-PARTIAL, 0 MISS.**

| # | probe | projection | descent verdict |
|---|---|---|---|
| 1 | whatever happened to Xury? | hit (people/xury) | FULL — sale, terms, the standing regret |
| 2 | tell me about Friday | hit (people/friday) | FULL — rescue, loyalty, faith, wolf/bear epilogue |
| 3 | the Portuguese captain? | hit | FULL — rescue, stewardship, annuity |
| 4 | Brazil plantation? | hit | FULL — arc through the 32,800-pieces sale |
| 5 | how did I end up on the island? | hit | FULL — 1659 voyage, wreck, sole survivor |
| 6 | my home over time? | hit (camp) | FULL — tent → cave → double wall → hidden grove |
| 7 | what animals did I keep? | hit (animals) | FULL — dog, cats, Poll (26 years), goat herds |
| 8 | the ship's wreck? | hit (stores) | FULL — 11 trips, breakup, Spanish wreck |
| 9 | how did I get off? | hit (escape/mutiny) | FULL — pact, shore campaign, 19 Dec 1686 |
| 10 | my financial situation? | **(nothing)** | PARTIAL — BFS into `money` recovered all of it |
| 11 | do I have any regrets? | **hit via facet** | FULL — Xury + the "original sin" self-blame |
| 12 | who do I trust most? | **hit via facet** | FULL — Portuguese captain + the widow |
| 13 | what am I afraid of? | **hit via facet** | FULL — cannibals, anticipatory dread, Inquisition |
| 14 | what am I proud of? | **(nothing)** | PARTIAL — achievements found, no explicit pride |
| 15 | am I a religious man? | hit (religion) | FULL — the earned-not-native arc |
| 16 | how long on the island? | hit | FULL — 28y 2m 19d, exact leaf |
| 17 | when/how did I leave? | hit | FULL — mutiny, recovered ship, the date |
| 18 | my father's advice? | hit (people/father) | FULL — middle station, the prophecy |
| 19 | am I content? | hit | PARTIAL — island contentment vs the 1694 relapse |
| 20 | learned my lesson about wandering? | hit (temperament/wandering) | FULL — "behaviourally unlearned", axis +0.71/9 |

Readings:

- **Derived facet routing carried the evaluative probes** (11, 12, 13) on a
  fresh corpus with zero hand-written facet vocabulary — regret, trust, and
  fear questions all routed mechanically because the lines carry those
  families. This is the auto-mirror doing on first contact what round two
  needed hand-tuning for.
- **The two projection misses are both vocabulary, not retrieval.**
  "Financial situation" shares no exact token with `money`'s routing
  (stemmingless matching, the standing gotcha); "proud" found nothing because
  no line advertises the pride family — the corpus-honest reading is that
  Crusoe narrates achievement without naming pride, and the fix (if wanted)
  is line content, not machinery. BFS descent recovered both — the designed
  fallback doing its job.
- **The profile recovered the arc again**, finer-grained than round two:
  `wandering-restlessness` +0.71 over 9 observations with `original-sin-named`
  as its leaf; `survival-presence` +0.76/5; religion's
  `religious-awakening-under-extremity` sits at −0.19/4 *next to* the settled
  `temperament/religion` median +0.33/6 — the axis structure kept the
  "turns to God only in extremity" pattern separate from the earned faith
  instead of averaging them.

## Caveats

- Harvester and interrogator were both Sonnet; the grounding rule plus
  spot-verification bounds contamination but doesn't eliminate the harvester
  knowing the novel.
- One agent occasionally ran two redistills in a chunk (pacing drift);
  harmless here, but a scripted driver would hold the line better.
- 66 × ~1,830-word chunks vs the original 72 × ~1,700 — paragraph-boundary
  accumulation overshoots; not material.
