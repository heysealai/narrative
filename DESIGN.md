# Narrative — a structured memory system for LLM agents

Local-first, per-person memory that distills observed behavior into a small, always-legible
structure instead of accumulating notes. Designed June 2026; this document is the converged
output of the original design conversation and is self-contained — a fresh session can pick
up from here.

## Why existing systems fail

Two failure modes motivated the design; every choice below traces back to one of them.

**Markdown/file memory** (Claude's default memory, Obsidian-style vaults): overloads as it
deepens. The model selectively chooses which files to read, so recall is stale and
non-contextual — the memory exists but isn't *in play* when it matters.

**RAG / vector memory**: works, but retrieval confidence is vibes. Similarity scores degrade
as breadth widens, the threshold has no semantics, and "why did you recall this?" has no
answer. Confidence-by-cosine is unauditable.

## Core idea: mind-map distillation

A **distillant-first tree**. The middle layer — the categories — is not a filing system; it
is made of **distillants**: nodes that *are* the system's learned model of the person ("how
they handle rent", "what they obsess about", "how they like being spoken to"). Each
distillant carries a **line** — one cached sentence of judgment, the behavioral median of
everything below it — and the routing vocabulary that makes it findable. Leaves are small,
boring, atomic facts that serve as evidence and detail under those concepts. A mind map:
the middle is the understanding, the edges are the receipts.

This attacks both failure modes structurally:

- **Staleness dies** because the thing that must always be in context — the category
  skeleton (labels + lines) — is *small by construction*. The layer compresses as it
  grows instead of accumulating. Nothing is selectively read; only leaf-opening is
  selective.
- **Confidence-vibes die** because there is no retrieval scorer. Recall is the model reading
  a legible map in-context and deciding, with judgment, which branch matters. Confidence
  comes from comprehension of labels, not a similarity threshold.

### The taxonomy is learned — gradient descent with a model as the optimizer

Categories cannot be pre-encoded (people differ), but the span of plausible *top-level*
categories is shallow and quasi-universal (money, work, people, habits, taste...). The
personal signature lives one or two levels down. So:

- **Near-universal crown, personalized middle, factual leaves.**
- The structure is found iteratively. The loss is *explanatory fit*: a fact that lands
  cleanly in an existing distillant is low-residual; facts piling up in "misc", a category
  whose leaves stopped agreeing with its line, two distillants claiming the same facts —
  that's the gradient. A **descent step = a consolidation pass restructuring**: split,
  merge, relabel, rehome, re-distill.
- A distillant is a **behavioral median**: the central tendency of observed behavior. One-off
  facts don't move the structure; repeated patterns do. That's why it stabilizes instead of
  thrashing.
- The tree grows recursively but stays shallow (~3 levels) because distillation pressure is
  constitutive, not cosmetic — see "tree health = retrieval health" below.

## The three stores

Dispositions and states want different shapes, and forcing one taxonomy over both produces
heterogeneous children ("money" parenting both `rent: $2,200` and `tends to overspend
late-month`). Split by species, each in its native topology:

| Store | Topology | Species | Lifecycle |
|---|---|---|---|
| **Stream** | time-ordered log | episodes | immutable, accumulate, fade/compress |
| **Registry** | noun-shaped tree (people, accounts, obligations, work...) | state facts | current value + supersession history |
| **Profile** | one apex — the character estimate — over a trait-shaped shallow tree (spend discipline, risk appetite, communication style...) | dispositions | scored axes that drift, with trajectories; the apex line is distilled from the axis lines |

The stream is the **shared evidence pool**: one episode ("paid rent late in May") supports a
registry history and nudges a profile axis. Both trees hold pointers into it; episodes are
never duplicated. (This is the episodic/semantic split from cognitive science, with the
profile as a distilled self-model on top.)

Both trees are DAGs — a leaf may hang under multiple distillants ("rent lateness" under both
*money habits* and *Lisa*).

## Context tiers — pinned vs retrieved

- **Profile = pinned tier.** Small, slow-changing, relevant to almost every turn (how to
  talk to this person, how cautious to be). The whole profile rides inline always — and
  because it changes rarely, it lives in the *cacheable* per-user system block without
  busting prompt cache. The tree's single root, `character`, carries the whole-person
  estimate consolidation distills from the axis lines beneath it — the pinned block
  opens with who this person is, then how they tend.
- **Registry + stream = retrieved tier.** Big, fast-growing, situationally relevant.
  Retrieved off-context per turn; injected in the per-turn (cache-safe, messages-array)
  slot.

The tier boundary and the species boundary are the same cut — two independently-made
choices landing on one line, which is the design telling you it's right.

Load-bearing memories (rules: "never auto-approve over $500") must **fire, not surface** —
probabilistic recall of a rule is worse than no rule, because trust is built on it firing.
Such rules live in the pinned profile, never in the similarity-gated tail.

## Retrieval — two motions over one structure

1. **Direct projection** (mechanical, free): every distillant carries a **routing map** —
   keywords, aliases, entity names — compiled at write time. The runtime lexically matches
   the outgoing user message against a single routing table (spanning both trees) and
   pre-opens the matched categories' leaves into the request *before the model runs*. No
   LLM in the pre-turn hot path. Matches are **ranked** mechanically (lexical score, then
   best-leaf salience) under a fixed leaf budget, so over-supply costs ordering pressure,
   never context. **Evaluative-facet vocabulary** (regret, trust, fear…) is *derived*,
   never model-written: whenever a line is set or rewritten, the runtime mirrors
   the facet families it carries into routing — whole families, so a "fear" line
   routes "afraid" — and prunes the families it dropped. The model's judgment lives in
   the line; the runtime compiles it down (the compile-down principle applied to
   facets). Stray model-written facet terms are gated against the line. (Measured
   before adoption, then re-measured: `docs/facet-routing-discovery.md`.)
2. **BFS** (model-driven, fallback): when the address isn't obvious, the model — which has
   already read the inline skeleton — opens a branch, reads children's descriptors,
   descends or backtracks via an `open(path)` tool. Cheap because the tree is shallow and
   branching is bounded: the frontier always fits in a glance.

**The compile-down principle** (recurring trick): write-time intelligence compiles into
dumb, fast, read-time artifacts. Distillation → skeleton; entity linking → alias tables;
routing → keyword maps. The model is never in the hot path, but its judgment is — cached.

**Tree health = retrieval health.** Projection works only while the skeleton fits in
context; BFS stays cheap only while the tree stays shallow. The optimizer isn't tidying for
aesthetics — compression maintains the invariant both retrieval modes depend on.

## Write path — ambient, no remember-tool

There is no explicit `remember` tool taxing the live turn. Writes are ambient:

- **Post-turn harvester**: an off-context model call reads finished turns and extracts
  episodes, state changes, and disposition nudges; classifies species; links to categories
  with the directory in view **leaf to root**: every distillant by id and label, and the
  neighborhood the turn touches expanded — the routing hits with their path to the root
  (line and routing: where the harvester files, and whose routing it extends) and each
  branch tip's children with their lines (the siblings a new fact lands among; a sibling
  named only by its label is a twin waiting to happen). A hit that another hit sits under
  is context on the way up, not a tip. Input scales with the turn's neighborhood, not with
  the memory; the scopes it was measured against are in `docs/harvest-scope-replay.md`.
  New-node creation is **harvester judgment**: durable participant in the person's life →
  node; incidental mention → string tag only.
- **Eviction-coupled distillation**: when the host's context approaches its eviction
  watermark, harvest the chunk about to phase out. Memory is, definitionally, **what
  survives forgetting**. (Hippocampus → cortex consolidation, as systems design.)
- **Forgetting is the user's, and it is final.** "Forget that" / "stop remembering X" is
  not a fact about the user to be classified — it is authority over what is held about
  them. The harvester emits `forgets` naming the ids that hold the content, applied
  after everything else in the batch; the runtime removes the leaf (or a distillant
  with its subtree) and the stream episodes that were evidence for nothing else. An
  episode restating the content, or one recording the request, would keep it
  recallable — the contract forbids both. This is still not a live-turn tool: the ask
  rides the same ambient harvest as every other write.
- **The profile is pinned for the harvester too.** Cross-matching against a routed
  sample lets the harvester mint a near-duplicate of an axis it was never shown, and
  belief strength then never accumulates (every axis at one observation). Every
  disposition rides in every harvest prompt, exactly as the profile rides pinned in
  every recall; the harvester nudges by id.

### Contradiction cross-matching happens at write, via projection pointed backwards

The harvester never asks "does this contradict anything anywhere?" (expensive, vague). The
new fact's categories are projected mechanically; the runtime pre-opens those leaves and
hands them over; the harvester classifies the relation: **novel / duplicate / supporting /
contradicting / superseding**. Same machinery as retrieval, reversed.

Slow drift — a line whose leaves quietly stopped agreeing with it, an axis whose last
N nudges all point one way — is invisible to single writes and is caught by consolidation.

## Belief mechanics — dispositions drift, states switch

People change, often gradually (sliding from one pole toward another). Encode this as
moving belief — but fork by species, or it goes wrong:

- **Dispositions drift.** A scored position on an axis. Each supporting/contradicting
  observation nudges it; **hysteresis** — no flip on one outlier (median-robustness applied
  to beliefs). The score's **trajectory is itself memory** ("has been tightening spending
  for six months").
- **States switch.** "Rent is $2,200" is not a 60%-confidence belief; it is current or
  superseded. Contradiction → supersession with timestamps; the old value becomes an
  episode ("rent was $2,200 until June 2026"). The history of change is some of the most
  informative content in the graph — never delete it.

**The model classifies; the runtime does the arithmetic.** If the LLM hand-assigns
"belief: 0.7", the number is vibes — the RAG-confidence disease in a new costume. Instead
the model emits discrete judgments (supports / contradicts / supersedes) and scores are
*derived* from countable events: reinforcement count, contradiction count, recency
weighting. "Why did this belief move?" then has a real answer: "four contradicting episodes
in sixty days." Auditability is non-negotiable when memory drives decisions (the original
host was a neobank agent).

## Dynamics — consolidation + salience

- **Consolidation** (timing deferred to the model; bottom line: runs when things are about
  to phase out at the watermark): re-distill stale distillants, split/merge/rehome, notice
  slow drift, absorb a node's episodes into its summary (the entity/distillant *is* the
  consolidation product), graph hygiene (merge nodes that turned out to be the same person,
  split conflations). Linking mistakes aren't fatal; they're deferred merges. Two
  hygiene invariants ride on this: a **bare** distillant (a fact named it before any pass
  wrote its line) carries the trigger's worth of pressure by itself, and the pass is shown
  its same-named distillants elsewhere in the tree so it can `merge_into` one instead of
  writing a duplicate a line; and **routing is replaced, not accreted** — harvest aliases
  only ever append, so the pass returns the complete routing set (referring expressions
  only, never episode detail or generic phrases) and the runtime swaps it in, re-deriving
  facet families from the fresh line. **Parent sets are antichains**: a node never lists
  an ancestor beside that ancestor's own descendant (the ancestor is implied and would
  render the node twice); every op that sets parents normalizes.
- **Drift** (the staleness trigger, derived — never stored): a line is a cached
  judgment, and the leaves keep living under it. Drift counts the evidence that the line no
  longer follows from its children: leaves that moved *against their own text* since the
  last pass (contradiction, supersession, a genuine disposition flip — one shared
  `last_against_at` stamp, weight 2) and children whose line text materially changed
  since (`line_changed_at`, weight 1). Drift adds straight into consolidation
  pressure, so semantic rot competes with structural rot in the same one-step-per-turn
  queue — no new scheduler, no extra model calls. Three properties do the real work:
  a pass clears drift *by construction* (timestamps compare against `consolidated_at`);
  a rewrite that returns the identical line stamps nothing, so cascades die at the level
  where the summary absorbed the churn (rollup is eventual, not eager, and terminates at
  the fixed point); and when a due distillant has a due child, the child goes first —
  freshness flows leaves-up, a parent never consumes a knowingly-stale child line. The
  redistill contract grounds the rewrite: the old line is a style reference, not a source —
  a claim with no leaf, event, or child line below it gets dropped, which is what keeps a
  fabricated claim from outliving its next pass. (What this does NOT catch: a fabricated
  line on a distillant where nothing ever changes — no event ever fires against it. The
  guard there would be a slow sweep of longest-untouched distillants; deliberately not
  built.)
- **Salience**: importance assigned at write, strengthened on retrieval, decayed with
  disuse; recall within an opened category ranks by relevance × recency × importance.
  Critically, **salience defends exceptions from median-washing**: a pure line would
  absorb "got scammed by X once" into "occasionally makes payment mistakes" — the useless
  version. High-salience outliers are protected from absorption, can hold their leaf, even
  force their own distillant. The median finds the structure; salience defends the exceptions.

## Entity canonicalization (registry)

"Lisa" = "lisa.eth" = "my sister" must land on one node. Forced architecture, from a
read/write budget asymmetry:

- **Write time** (harvester, has time + judgment): model links mentions against the inline
  directory and **emits aliases** ("my sister" → alias on Lisa's node).
- **Read time** (projection, no LLM available): dumb lexical matching against accumulated
  aliases. The **alias table is the interface** — write-time intelligence compiled down,
  again.
- Tractability gifts: **exact anchors** when the domain provides them (wallet addresses,
  ENS names, handles — primary keys, not fuzzy mentions); **single-user possessives** ("my
  sister", "my landlord") are stable aliases when there's one person per graph — poison in
  multi-user KBs, free vocabulary here.
- Read-time ambiguity is permissive: "Lisa" matching two nodes activates both; over-supply
  costs tokens, not correctness. Repair (merge/split) belongs to consolidation.
- **The user themself is never an entity node.** The whole graph already models them — their
  history under the topical crowns, their tendencies in the profile. A `people/<the-user>`
  node splits the self-model in two and starves the profile (the Franklin run found this
  live; the harvester contract now states it).

## Data model sketch

```
Leaf {
  id, species: episode | state | disposition,
  text,                      // small, atomic
  parents: [distillant ids],   // DAG, multi-parent
  salience: { importance, last_retrieved, retrieval_count },
  belief:   { strength, support_count, contradict_count, last_event,
              last_against },  // derived, not assigned; last_against = when an event
                               // last moved against the text (drift reads this)
  // states:       current value + supersession chain (old values -> episodes)
  // dispositions: axis position + trajectory of nudges
  evidence: [stream ids],
  created_at, updated_at,
}

Distillant {
  id, label,
  line,                // one line; the cached judgment — keep it fresh or rot returns
                       // (BARE until a harvest or pass stamps line_changed_at: an unstamped
                       // line was never a judgment; empty or label-only says nothing either)
  routing_map: [keywords, aliases, anchors],   // compiled for mechanical projection;
                                               // facet terms only while the line carries them
  parents: [distillant ids], children: [ids],   // parents are an antichain (no ancestor beside its descendant)
  residual: misc_count,        // gradient accumulator: facts parked here for lack of better
  line_changed_at,       // stamped on MATERIAL line change; parents read it as drift
  forgotten_at,          // stamped when a forget removed something held here; drift reads it
  // disagreement is DERIVED, not stored: see §Dynamics — Drift. Child timestamps
  // vs consolidated_at; a pass zeroes it by construction.
}

RoutingTable: single, spans registry + profile; entries point into either tree.
```

## What the host app must provide (integration seams)

The original host was a chat agent app (RN/Expo neobank, "Seal") with: per-identity local
storage, a token-watermark context-eviction mechanism (128K high / 70K low), a cacheable
per-user system-prompt block + a cache-safe per-turn messages slot, and off-turn LLM call
machinery (background subagents). Any host needs equivalents:

1. Finished-turn transcripts delivered to the harvester.
2. An eviction signal + the about-to-evict chunk (for distill-before-forget).
3. A pre-send hook on outgoing messages (projection match + leaf injection).
4. An inline slot for the pinned profile (cache-friendly) and a per-turn slot for
   projected leaves (cache-safe).
5. An LLM channel for harvester/consolidation calls, off the interactive turn.
6. An `open(path)` tool exposed to the main model for BFS descent.

## Rejected along the way (and why)

- **md docs / personal-doc files** — staleness by selective reading; whole-doc rewrites.
- **Pure RAG** — confidence scoring degrades with breadth; unauditable.
- **Single topic tree** — heterogeneous children once dispositions and states mix.
- **Embeddings in v1** — entity activation + lexical matching covers a proper-noun-heavy
  domain; no embedding provider needed (the original host's server was Anthropic-only —
  no embeddings API); add vectors later only if paraphrase recall proves weak.
- **Explicit remember tool** — under-remembers, taxes the live turn; ambient harvest wins.
- **Eager entity creation** — sprawl; **promotion-on-2nd-reference** — misses the window;
  harvester judgment chosen instead.
- **LLM-assigned belief numbers** — vibes; derived-from-events arithmetic instead.
- **Inline-everything memory** — fails at depth; off-context retrieval with a pinned core.

## Open questions for the build

1. Consolidation scheduling specifics (model-deferred + watermark bottom line — what's the
   concrete trigger set? residual thresholds? idle time?).
2. Disposition axis discovery — are profile axes emergent like categories, or seeded from a
   small universal set and personalized below? (Built as the latter: three seeded axes
   under the apex, the harvester personalizes below them.)
3. Projection injection format — how do opened leaves render into the turn (block shape,
   token budget per turn, dedup against recent injections)?
4. Multi-resolution answering — when should the agent answer from a line vs opening
   leaves?
5. Cold start — seeding from existing data (contacts, transaction history, chat backlog).
6. Harvester/consolidator model choice and cost envelope; batching policy.
7. Storage engine in Rust + how the host (likely non-Rust) talks to it: embedded lib,
   local service, FFI?
