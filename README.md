# narrative

A structured long-term memory system for LLM agents. Not markdown notes, not
RAG: a per-person **mind map** — a layer of *distillants* (distilled judgment
nodes) over three stores with different lifecycles — maintained ambiently by
a harvester model and recalled mechanically with no LLM in the read hot path.

Full design rationale: [DESIGN.md](DESIGN.md) (the contract).

## The intuition

Someone who knows you well doesn't grep transcripts of every conversation
you've ever had. They hold a small, settled understanding — *she tends to
overspend late in the month; the lease is a standing regret* — and reach for
specifics only when something prompts them. The understanding is cheap to
carry; the details are fetched.

Existing agent memory fails on one side or the other. Notes-file memory keeps
summaries but lets them rot — nothing re-checks a summary once written, and
whole-doc rewrites destroy as much as they fix. RAG keeps every detail but
never settles anything — retrieval confidence degrades with breadth, and no
one place *understands*.

Narrative's answer is distillation with a maintenance loop. A model with
judgment (the harvester) watches each finished turn and writes small, atomic
facts into a graph. Consolidation — the same judgment, off the hot path —
keeps compressing what accumulates into **distillants**: nodes that are
simultaneously *index* (routing vocabulary that catches future mentions) and
*understanding* (one cached line, the behavioral median of everything below).
Reads are then purely mechanical: lexical matching against vocabulary the
model compiled at write time. Write-time intelligence, compiled down.

The cached line is a judgment, and cached judgment rots — so staleness itself
is detected mechanically (drift, below) and queued for re-distillation. The
graph doesn't just remember; it notices when what it believed stopped being
true.

## How it works

Three stores, one routing table, two motions:

| Store | Topology | Lifecycle |
|---|---|---|
| **Stream** | time-ordered episodes | immutable, accumulate, digest, fade |
| **Registry** | noun-shaped tree of state facts | current value + supersession history (*states switch*) |
| **Profile** | one apex (the character estimate) over a trait-shaped tree of dispositions | scored axes with nudge trajectories (*dispositions drift*); the apex line is distilled from the axis lines |

**Write path (ambient, no remember-tool).** Every finished turn goes to the
harvester, which emits structured ops — new leaves, supports / contradicts /
supersedes classifications against existing facts, aliases, distillant
creation, line rewrites, and **forgets**: when the user asks to forget or
stop remembering something, the harvester names the ids that hold it and
the content leaves every store — the leaf, and the stream episodes that were
evidence for nothing else. Forgetting is the user's authority, applied last
in the batch so nothing the same harvest wrote about the content survives
it. The runtime does all arithmetic: belief strength is
derived from countable events, never model-assigned. Cross-matching sees
the **whole profile every time** (pinned for the harvester exactly as it is
pinned for recall), so an observation about a tracked tendency lands as a
nudge on that axis — belief strength can only accumulate on an axis the
harvester can see. Facet routing is derived
too — whatever evaluative families (trust, regret, fear…) a distillant's line
carries are mirrored into its routing automatically; models write honest
lines, the runtime compiles them.

**Read path (mechanical, two motions).** The profile rides pinned in every
request, opening with the `character` line — the engine's standing estimate of
who this person is, redistilled as the axes beneath it move. Per turn, the message is lexically matched against the routing
table; activated distillants project their best leaves (ranked by match
score, then by the leaf: the message words it shares, then salience) under a
fixed budget. For interrogation beyond what
projection catches, the agent walks the map itself: an `open_memory` tool
descends the tree BFS-style, and a question can only descend where some line
on the path advertises the relevant vocabulary — lines are retrieval scent.

**Maintenance (one step per turn).** Every distillant accrues pressure:
leaves past the fat threshold, residual facts parked with nowhere better to
go, and **drift** — derived evidence that the cached line no longer follows
from its children (leaves that moved *against their own text* via
contradiction, supersession, or a disposition flip; child lines that
materially changed). A distillant a fact named before any pass wrote its line is **bare**, and
bare is the trigger's worth of pressure on its own. The worst offender over
the trigger gets one redistill
pass: rewrite the line grounded in what's actually below (the old line is a
style reference, not a source), **replace** the routing set (routing only ever
grew at harvest; the pass returns the complete set it should carry), split
fat nodes, merge duplicates — and merge a distillant *away* into a same-named
one elsewhere in the tree, which the pass is shown so a duplicate branch is a
choice it can make. Parent sets are antichains: a node never lists an ancestor
beside that ancestor's own descendant. A due
child consolidates before its due parent, and an identical rewrite stamps
nothing — cascades die where the summary absorbed the churn. The stream has
its own species: past a soft cap, the oldest episodes get distilled into a
digest that cuts at a natural period boundary, while the texts are still
alive to read.

## What it aims for

- **Memory that survives forgetting.** Distill-before-forget (hippocampus →
  cortex consolidation, as systems design): by the time detail fades, its
  meaning has been absorbed upward.
- **Auditable belief.** "Why did this belief move?" always has an answer like
  *four contradicting episodes in sixty days* — arithmetic over classified
  events, never model vibes.
- **No LLM, no embeddings, in the read hot path.** Recall is lexical matching
  against compiled vocabulary — cheap, fast, deterministic, debuggable.
  Embeddings stay deferred until paraphrase recall measurably needs them.
- **Cached judgment that can't quietly rot.** Derived facet routing, the
  grounding contract on rewrites, and drift-as-pressure exist so that a
  distillant's line either tracks its evidence or gets re-distilled.
- **Host-agnostic.** The engine externalizes every model role through the
  keyless CLI — any intelligence (an API key, a Claude session, a subagent)
  can play harvester, consolidator, or agent. Host integration seams are
  documented in DESIGN.md.

## Run the chat-box simulator

```sh
export ANTHROPIC_API_KEY=...   # or: cargo run -- --mock (offline, canned model)
cargo run
```

```
you ❯ my rent went up to $2,350, due the 1st — landlord is Marcus, marcus.eth
  ✎ ✚ distillant money/rent — Rent obligation: amount, due date, landlord.
  ✎ + state [rent-amount] Rent is $2,350 per month. (under money/rent)
  ✎ ≈ alias people/marcus ← marcus.eth, my landlord
narrative ❯ Got it — $2,350 on the 1st, to Marcus. ...
```

Memory persists in `data/memory.json` (`NARRATIVE_DATA` overrides the dir);
chat history deliberately does not — restart the sim and ask "when's rent due?"
to watch cross-session recall work.

REPL commands: `/help` `/profile` `/map` `/tree` `/open <id>` `/stream [n]`
`/project <text>` (dry-run recall) `/feed <text>` (ingest user-voice text,
harvest only) `/distill <id>` (consolidation step) `/digest` (stream digest
step) `/stats` (residual + drift report) `/forget <id>` `/reset-chat` `/quit`.

Env: `NARRATIVE_MODEL` (default `claude-opus-4-8`; the harvester works well on
`claude-haiku-4-5`), `NARRATIVE_DATA` (default `./data`), `NARRATIVE_DEBUG=1`
(raw harvester output).

## Tests

```sh
cargo test            # unit + end-to-end loop against the mock model
```

## Drive it without a key

The CLI externalizes the model roles so any intelligence can drive the engine
(this is how the Crusoe experiment ran — no API key in the loop):

```sh
narrative project "<message>"        # mechanical recall (touches salience)
narrative harvest-prompt "<turn>"    # the harvester contract: system + schema + input
narrative apply ops.json             # apply harvester ops (sectioned schema)
narrative digest-prompt              # input for a due stream digest pass
narrative digest "<text>" [--take n] # apply digest text (n = episodes covered)
narrative redistill-prompt <distillant>      # that distillant's full contract
narrative redistill <distillant> @out.json   # apply the redistill response
narrative open <distillant> | profile | map | stream | stats | forget <id>
narrative replay turns.json <full|compact|selective> report.json  # harvester over recorded turns
```

`forget` takes a leaf, or a distillant with its whole subtree, and the stream
episodes that were evidence for nothing else — the same removal the
harvester's `forgets` op performs when the user asks in conversation.

`replay` drives the harvester (the configured model, or the mock) over
recorded turns — `{"turns": [{seq, at, at_epoch, user, assistant}, …]}` —
from the stored graph, under one directory scope: `selective` is the
harvester's own directory (every distillant by id and label; the branches
the turn is on with line and routing, the branch tips' children with their
line); `full` renders every distillant with its line and routing, and
`compact` every distillant with its routing and no line — the two the
design was measured against. It writes the graph after every turn and a
per-turn report of tokens, ops, the turn's neighborhood, and what was
minted: [docs/harvest-scope-replay.md](docs/harvest-scope-replay.md).

## Module map

| | |
|---|---|
| `model.rs` | the three stores, leaves/distillants, seed crown |
| `belief.rs` | belief & salience arithmetic, EMA axis nudges, supersession |
| `routing.rs` | routing table + lexical matching (read hot path) |
| `projection.rs` | per-turn recall, pinned/skeleton/open rendering |
| `harvest.rs` | post-turn harvester: structured-output ops + application |
| `consolidate.rs` | redistill (descent step: rewrite line + split/merge), pressure + drift trigger, residual stats |
| `agent.rs` | system prompt assembly, open_memory tool loop |
| `llm.rs` | Messages API client (raw HTTP) + scripted mock |
| `sim.rs` | the REPL |
| `replay.rs` | the harvester over recorded turns under one directory scope, with a per-turn report |

## The corpus experiments

Three full first-person books have been fed through the engine end to end —
each chosen to stress a different part of the design, each checked in under
[examples/](examples/) with a run report, and each interrogated afterwards
under a strict grounding rule (any claim not seen in a render counts as
contamination):

- ***Robinson Crusoe*** (Defoe, ~121k words) — the self-model and state
  supersession. Run twice: the original session
  ([docs/crusoe-tracking-report.md](docs/crusoe-tracking-report.md))
  recovered the character arc as drift arithmetic (piety: −0.58 → +0.87
  across 25 nudges, flipping exactly at the conversion); the checked-in
  graph is the June 2026 rebuild on the drift-aware engine — **50
  distillants, 294 leaves, 339 episodes**, 20-question interrogation 17
  full / 3 partial / 0 miss, with regret/trust/fear probes routing
  mechanically off derived facet families:
  [docs/crusoe-rebuild-run.md](docs/crusoe-rebuild-run.md).
- ***The Autobiography of Benjamin Franklin*** (~65k words) — the profile
  under a real life: the 13-virtues project lands as
  `temperament/virtue-project`, the civic career builds an 81-node
  registry, and Franklin's "errata" route as regret. **92 distillants, 278
  leaves, 237 episodes**; 10 questions, 8 full / 2 partial / 0 miss:
  [docs/franklin-run.md](docs/franklin-run.md).
- ***My Ántonia*** (Cather, ~81k words) — the people registry when the
  user's memory is mostly about someone else: `people/antonia` grows a
  four-level subtree organized by epoch and relationship. **54 distillants,
  311 leaves, 205 episodes**; 10 questions, 8 full / 2 partial / 0 miss —
  and the facet probes that routed on Crusoe correctly *don't* route here,
  because Jim Burden's voice never names regret or trust; descent recovers
  them: [docs/antonia-run.md](docs/antonia-run.md).

> Reports in `docs/` that predate the June 2026 rename use *midpoint* for
> today's *distillant* (and *distillant* for its *line*); each carries a
> terminology note.

## Status

Third pass — consolidation schedules itself and audits its own cache:

- **Drift — staleness is pressure** (derived, never stored): when leaves
  under a distillant move against their own text (contradiction,
  supersession, disposition flip — one `last_against_at` stamp, weight 2)
  or a child's line materially changes (`line_changed_at`, weight 1) after
  the last pass, the distillant accrues drift and competes in the same
  one-step-per-turn queue. An identical rewrite stamps nothing, so rollup
  cascades die where the summary absorbed the churn; a due child
  consolidates before its due parent; the redistill contract treats the old
  line as style reference, not source — unsupported claims drop instead of
  being carried forward. `/stats` lists drifted distillants.
- **Automatic consolidation**: pressure = leaves past the fat threshold +
  residual count + drift; after each harvest the sim runs one descent step
  on the worst offender over the trigger — the sim-side stand-in for the
  host's eviction-watermark coupling. A `consolidated_at` stamp keeps a
  distillant quiet until something actually happens under it, so a declined
  split can't thrash. `/stats` and the CLI `apply` both name the next
  distillant due (the keyless CLI can't run a model, so the driving session
  is told to do it).
- **Facet routing is derived, not model-written** (measured, adopted,
  re-measured: `docs/facet-routing-discovery.md`): whenever a line is set
  or rewritten, the runtime syncs routing's facet vocabulary to exactly the
  families the line carries (`routing.rs::FACET_FAMILIES`; whole families,
  so a "fear" line routes "afraid" and "dread") and prunes what it dropped.
  Stray model-written facet terms are gated against the line. Projection
  ranks matches (lexical score, then the best leaf underneath) so the
  12-leaf budget feeds the strongest matches instead of table order, and
  ranks each match's leaves by the message words they share before
  salience, so the leaf the message names survives the per-node cap
  instead of losing its slot to a more salient sibling.
- **Stream digests are distilled, not concatenated**: past the soft cap
  (1000) the stream is *due* — the next consolidation step shows the model
  the oldest 150 episodes, full texts intact, and the model returns a real
  period summary plus the cut: how many it covers (50–150), so digests end
  at a natural boundary (a voyage, a year, a move) instead of an arbitrary
  count. Only past the hard cap (1100) does the old lossy mechanical fold
  fire as a keyless backstop, and any such `[digest of ...]` fragments get
  polished by a later pass (the marker is the trigger; rewriting removes
  it). One stream step per turn, distillant pressure first.
- **Lines advertise evaluative facets**: the BFS interrogation test showed
  descent can only find what some line on the path mentions ("a standing
  regret" routes a regret question; "proves loyal" doesn't). Harvester and
  redistill prompts push evaluative vocabulary into lines alongside topical
  summaries — write-time fix, embeddings still deferred.

Second pass — the Crusoe-run priorities are in:

- **Event-time vs write-time**: episodes and states carry harvester-settable
  `occurred_at`; rendered ages ("noted 2mo ago", "changed 3mo ago") prefer
  story time, while belief/salience arithmetic stays on write time.
- **Structural ops**: `moves`, `reparents`, `merge_leaves`,
  `merge_distillants` joined the op vocabulary (cycle-guarded; merges
  combine evidence, belief counts, and routing vocabulary). `redistill` can
  split a fat distillant into children and merge duplicate leaves, not just
  rewrite the line.
- **Episode retrieval**: episodes route through the same table via their tags —
  recall gains a recency-windowed "related events" section, `open_memory` a
  "recent events here" list.
- Lighter `reinforces` op (no more dummy-text supports); the harvester
  cross-matches a wider comparanda window than the agent recall caps; stream
  overflow compresses the oldest episodes into a digest (leaf evidence
  pointers remapped) instead of silently dropping.

Remaining deliberate gaps: paraphrase recall beyond BFS — the designed-miss
interrogation ("who do I trust most?") now **passes via `open_memory` descent**
in 2 of 6 rounds ([docs/bfs-interrogation-test.md](docs/bfs-interrogation-test.md)),
so embeddings stay deferred; the live boundary is line coverage, not
retrieval. Fabrication on a *quiet* distillant (no event ever fires against
it) is the known drift gap — it would need a slow longest-untouched sweep.
Eviction-coupled timing is a host-integration concern the sim approximates by
harvesting every turn and running at most one consolidation step right after.
