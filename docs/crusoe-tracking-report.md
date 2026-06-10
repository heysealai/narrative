# Tracking experiment: Robinson Crusoe through the memory system

> **Terminology note (post-dates this report):** on June 11, 2026 the node
> formerly called a *midpoint* was renamed **distillant**, and the node's
> cached one-line summary — which this report calls its *distillant* — is now
> its **line**. This report keeps its original vocabulary.

**Question:** feed a several-hundred-page first-person book through the engine —
how closely does the memory track the person?

**Setup.** Full Gutenberg text of *Robinson Crusoe* (~121k words), split into 72
chunks of ~1,700 words, fed in order. The deterministic engine (routing,
projection, belief arithmetic, persistence) ran as the `narrative` CLI; the
model role — both harvester and, at interrogation time, agent — was played by a
Claude session driving the CLI (`harvest-prompt` → compose ops → `apply`).
A first-person book was chosen deliberately: the system models *one person*,
so the narrator maps cleanly onto "the user". No API keys, no embeddings, no
LLM anywhere in the recall hot path.

**Resulting graph:** 22 midpoints (19 registry / 3 profile), 78 leaves,
172 episodes. The graph is checked in at `examples/crusoe/memory.json`.

## What tracked well

### 1. The profile recovered the character arc as arithmetic

`piety` — the spine of the novel — ended at axis **+0.87 over 25 nudges**, and
the *trajectory* reads as the arc literary criticism describes: three −1s in
the impious youth (father's warning dismissed, storm vows drowned in punch),
wavering false starts (the barley "miracle" abating, the earthquake's rote
prayer), a sustained +1 run at the illness/conversion, a −2 crisis at the
footprint ("my fear banished all my religious hope"), then mature faith.
The polarity-flip detector fired exactly once on this axis — at the
debtor-and-creditor accounting, which is where the genuine turn happens
in the book.

Other axes landed where a close reader would put them: `self-reliance` +0.92,
`prudent-foresight` +0.90, `wanderlust` +0.64 *after* the epilogue (widowed,
off to the East Indies at sixty — correctly never cured), and `contentment`
at a violently-oscillated **+0.03 over 14 nudges** — honest, because Crusoe's
contentment genuinely whipsaws (composure → footprint terror → Friday's
company → escape hunger).

The behavioral-median property did its job: single observations nudged but
never flipped; the EMA needed sustained counter-evidence to cross zero, and
the flip report required prior opposition (a fresh axis committing to its own
pole is not a "flip").

### 2. States switched; history accumulated

Supersession chains captured change-of-state with the old values retained:

- `xury-companion`: sworn companion → *sold to the Portuguese captain,
  freedom in ten years, "a standing regret"*
- `enslaved-sallee`: slave at Sallee → *escaped in the longboat*
- `dwelling-now`: tree → tent ring → fortified cave → double-walled,
  musket-ported, wood-concealed castle (4 supersessions)
- `wreck-accessible`: reachable → gone → heaved up by the earthquake →
  stripped (3 supersessions)
- `sole-survivor`: alone on the island → *departed 19 December 1686, after
  28 years, 2 months, 19 days*
- `brazil-plantation`: thriving → fate unknown → recovered rich (the
  20-year gap bridged correctly)

Recall renders these as "(changed …; was: …)", so the agent sees both the
current value and what it replaced.

### 3. Mechanical recall hit 7/8 interrogation questions

Cold projection (lexical routing only, no model) against ground truth:

| Question | Result |
|---|---|
| when did I wash up on the island? | ✅ landing-date (30 Sept 1659) + departure |
| tell me about Friday | ✅ full Friday cluster incl. father, nation |
| whatever happened to Xury? | ✅ supersession chain with the regret |
| how did the plantation turn out? | ✅ recovered-rich state + will |
| where is my money kept? | ✅ stores + money crown (coin-hoard ranked low) |
| that footprint on the beach? | ✅ explained-supersession + cannibal context |
| how is the flock and dairy? | ✅ split-flock state, Poll, the dead first goat |
| **who do I trust most in the world?** | ❌ nothing recalled |

The miss is the *designed* miss: zero lexical hooks. In a real turn the agent
holds the registry skeleton inline and would descend via `open_memory(people)`
— the BFS tier exists precisely for paraphrase recall. Embeddings remain the
deferred fallback if that proves insufficient in practice.

### 4. Compiled aliases did real work

"my landlord"-style vocabulary recorded at write time routed messages later:
"my sister" → people-midpoints, "the landlord" → money/rent (in pilot),
"lisa.eth"-style exact anchors. In the Crusoe run, routing terms like
`castaway`, `lease`, `orinoco`, `colony` were all write-time compilations
that later resolved queries with no model in the path.

### 5. Consolidation pressure surfaced where it should

`/stats` at the end: `life/island` fat at 26 leaves (flagged as split
candidate — correct: it wants children like `island/threats`,
`island/escape`), residual counters on crown roots (`people: 4`) pointing
at exactly the leaves that deserve midpoints (spaniard-guest,
fridays-father). The gradient accumulators measure real disorder.

## What broke or strained (the valuable part)

1. **Structured-output degeneration on tagged unions.** The original harvest
   schema was one `ops` array with an `anyOf` of six op variants. Under
   constrained decoding the live model collapsed to a single degenerate
   `{"op":"episode","text":""}` — and a later turn produced outright token
   soup inside string fields. Fix: six named homogeneous sections
   (`midpoints/episodes/states/dispositions/aliases/distills`) — no
   discriminator for the grammar to degenerate on, and section order doubles
   as apply order. Plus a one-retry on parse failure. **Lesson: schema shape
   is a reliability surface, not a style choice.**

2. **Wall-clock vs story-time.** Every leaf reads "(noted today)" because the
   engine stamps write-time. A book compresses 35 years into one afternoon;
   the same bug bites real hosts at lower intensity (importing chat backlog).
   Memory needs an *event-time* field distinct from write-time, with the
   harvester allowed to set it ("changed 2mo ago" should mean story time).

3. **No re-parenting / move op.** When `life/island/animals` was created
   (chunk 17), earlier leaves (`island-goats`, `island-pets`) couldn't be
   moved under it — the op vocabulary has no `move`/`reparent`/`merge-leaf`.
   Consolidation needs structural ops, not just `distill`.

4. **Episode tags are dead weight.** Episodes carry midpoint-id tags but
   nothing retrieves through them yet; the stream is only reachable via
   `/stream`. Recall for "what happened around X" wants episode retrieval
   through the same routing table.

5. **`supports` with empty text is awkward.** Reinforcement ops must carry a
   throwaway text field because the schema requires it. Worth a dedicated
   lighter op.

6. **Comparanda cap can hide the cross-match target.** Projection over the
   chunk text caps at 12 leaves; once the graph grew, the right target
   occasionally wasn't in the comparanda window and only harvester memory
   (or luck) prevented duplicate-as-novel. The harvester-facing projection
   should be wider than the agent-facing one, or anchored by id-mention.

## Verdict

The three-store design held up against a 121k-word life. The profile is the
strongest result — disposition drift with belief arithmetic reproduced the
book's character development as an auditable number series, which is exactly
the "behavioral median + trajectory is itself memory" bet. The registry's
supersession chains gave correct then-vs-now answers across 28 story-years.
The known gaps (paraphrase recall, story-time, structural consolidation ops)
are all in the deferred-by-design column, now with concrete evidence for
prioritizing: **event-time first, structural ops second, embeddings last.**
