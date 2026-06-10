# Facet-guidance validation (June 10, 2026)

> **Terminology note (post-dates this report):** on June 11, 2026 the node
> formerly called a *midpoint* was renamed **distillant**, and the node's
> cached one-line summary — which this report calls its *distillant* — is now
> its **line**. This report keeps its original vocabulary.

Round-three priority 1: do the facet prompts (harvester rule 13,
`REDISTILL_SYSTEM`'s scent paragraph, the schema descriptions) actually
produce distillants that route evaluative questions? Run keyless,
Claude-in-the-loop, against /tmp copies of `examples/crusoe` (fixture
untouched). Same method caveat as the BFS test: the driving session is a
frontier model that knows both the novel and the test's purpose — what this
validates is that the prompts as written elicit facet-bearing output from a
model following them literally, and that the mechanical path downstream
works end to end. Small-model discipline remains priority 3 (needs a key).

**Verdict: pass, both directions, with one tuning finding (inflection) —
fixed in the prompts alongside this doc.**

## Test 1 — redistill `people/xury` (the BFS probe-2 partial)

Baseline reproduced: `project "do I have any regrets?"` → `(nothing
recalled)`; distillant "swore faithfulness, proves loyal" — no regret scent
anywhere on the path, exactly as `docs/bfs-interrogation-test.md` recorded.

Hand redistill per `REDISTILL_SYSTEM` (a `midpoints` upsert — it refreshes
the distillant and *adds* routing terms, mirroring the model-side
`redistill_schema`; the lighter `distills` op carries no routing):

- distillant → "…sworn companion, given over to the Portuguese captain; the
  parting is a standing regret."
- routing += `regret, regrets, parting`

After: the skeleton line for `people/xury` carries the regret vocabulary
(descent smells it in round 0), and the probe **routes mechanically** —
`project "do I have any regrets?"` returns the full xury recall block, no
descent needed. The BFS test's instructive partial is closed by the
write-time lever, as designed.

Caveat on this half: `REDISTILL_SYSTEM`'s inline example *is* the Xury
case, so it mostly validates the mechanical path, not prompt
generalization. That's what test 2 is for.

## Test 2 — live harvester turn, novel trust fact

A facet (trust) and a person (the captain's widow) that rule 13's example
does not mention; the widow existed only as a clause inside two
`money/trade` leaves, no midpoint. Turn fed through `harvest-prompt`:
"…she has never wronged me of a shilling… the one soul I would trust with
everything I have."

Following the eleven rules literally, the harvest came out as: a new
`people/widow` midpoint whose distillant advertises the facet ("…never
wronged him of a shilling; the one soul he would trust with everything"),
two atomic states (stewardship, trust), a `reinforces` on `guinea-trader`
(the £200 restated), and a sparing rule-12 move giving `brazil-stock` dual
parents (`money/trade` + `people/widow`). All applied cleanly; the skeleton
now shows the trust line under `people`, so the BFS test's probe 1 ("who do
I trust most?") gets its shortlist one round earlier — the widow joins
friday / portuguese-captain / xury in the round-1 listing with the trust
vocabulary inline.

`project "who do I trust most in the world?"` still misses, as designed —
"trust" is distillant scent for descent, not a routing term (rule 4 keeps
routing to referring expressions). The descent tier is the answer path for
that probe, and its scent is now strictly better.

## The tuning finding: inflection

`routing.rs::matches` is exact-token (single words) / exact-phrase
(multi-word) over normalized text. **No stemming.** Demonstrated on a
control copy: routing term `regret` (singular only) matches "do I regret
anything?" but *misses* "do I have any regrets?". A model writing the
natural singular form silently loses the plural paraphrase.

Fix landed with this doc: one line in `REDISTILL_SYSTEM`'s routing sentence
and harvester rule 4 — when adding a routing term, include the inflected
forms a message would actually contain (regret/regrets). The alternative —
a stemmer in `routing.rs` — is a real design decision (English-only
suffix-stripping, over-match risk) and stays open; the prompt line is the
cheap lever, consistent with the BFS test's conclusion.

## One design observation, not acted on

Routing facet terms (test 1's `regrets`) make `project` itself answer a
zero-hook paraphrase — strictly stronger than descent-only. But pushed as
doctrine ("always put facet words in routing") it would hang generic
emotion vocabulary on many midpoints; with several regrets in a graph,
every regret-shaped message pulls them all. Permissive over-supply is the
stated design posture, so this may be fine — but it changes the BFS test's
"distillants are the lever, not routing" boundary, so it's a conversation,
not a commit. The prompts as landed leave routing guidance at rule 4's
referring-expressions scope.
