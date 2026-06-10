# Facet-routing discovery run (June 10, 2026)

> **Terminology note (post-dates this report):** on June 11, 2026 the node
> formerly called a *midpoint* was renamed **distillant**, and the node's
> cached one-line summary — which this report calls its *distillant* — is now
> its **line**. This report keeps its original vocabulary.

The question from `docs/facet-validation.md`'s last section: should facet
words go into routing as doctrine, or stay distillant-only? The drafted
mechanism (committed alongside this doc, **decision pending**) is the
**distillant gate**: a facet may be mirrored into routing only when the
distillant itself advertises it. This run measures what that prompt change
actually elicits, against the old prompts, before anyone marries it.

## Method

Two arms, identical inputs, one variable — the prompt text:

- **OLD**: the prompts as of master (`15820ef`).
- **NEW**: the gate sentences (rule 4, rule 13, `REDISTILL_SYSTEM`, both
  routing schema descriptions).

Per arm: redistill of the same 10 facet-bearing midpoints (all `people/*`,
`life/origins|voyages|slavery-sallee`, `money/trade`, `work/plantation`),
rendered up front from the pristine fixture so both arms see byte-identical
inputs (OLD renders are NEW renders with the gate sentence stripped —
verified); plus two live harvest turns (a fear-loaded recounting, and an
*incidental* annoyance as the negative control). Driver: claude-sonnet-4-6
via Claude Code subagents, one per call, holding exactly the role contract
(system + output schema source + rendered input), per the handoff technique.
Applied via the new keyless `narrative redistill` / `apply`; probes run
against scratch copies (projection saves). All 24 outputs applied with zero
schema errors. Fabrication check: every suspicious routing token (jamaica,
Ismael, Moely, leopard, sweetmeats, Job, calenture…) traces to its render.

## Numbers

Routing terms: base 235 → OLD 399 / NEW 403. Facet-lexicon terms (regret,
trust, fear, pride, loyal, faithful, devotion, grateful, jealousy, weep,
aversion families): **OLD 20, NEW 29**. Gate violations *at emission*
(facet routed, distillant doesn't carry it): **OLD 2** (`jealousy` on
friday, `faithful` on slavery-sallee), **NEW 0**.

Probes (blocks = midpoint sections in the injection; base = pristine):

| probe | base | OLD | NEW |
|---|---|---|---|
| "do I have any regrets?" | miss | 5 blocks, 5.0KB | 5 blocks, 5.3KB |
| "do I regret leaving home?" | 2 blocks | 3 blocks | 3 blocks |
| "who do I trust most in the world?" | miss | miss | **miss** |
| "am I afraid of anything?" | miss | miss | **miss** |
| "is there anything I am proud of?" | miss | miss | **hit** (english-captain episodes) |
| control: "how is the plantation in Brazil doing?" | 1 block | 4 blocks | 4 blocks |

## Findings

1. **The old prompt already mirrors facets into routing.** Five midpoints
   carry `regret/regrets` routing in the OLD arm; the scent paragraph's
   "(or the routing)" plus the session-3 inflection example ("regret AND
   regrets") were already licensing the behavior. The gate sentence does
   not introduce facet routing — it *bounds* it. The existential-regret
   probe goes from zero recall to a five-midpoint mechanical answer in
   both arms identically.

2. **The gate works as a constraint.** Zero emission violations under NEW
   vs two under OLD. NEW slavery-sallee is the clean fix: its distillant
   says "faithfulness of Xury its standing note", so `faithful/faithfulness`
   routing is licensed; OLD routed `faithful` with no distillant backing.

3. **Under-mirroring is the dominant failure, and the gate doesn't fix
   it.** "Who do I trust most?" still misses in every arm: three NEW
   distillants carry trust vocabulary ("long-trusted steward") yet no
   model mirrored `trust` into routing. `regret` — the facet named in the
   prompt's example — mirrored 5/5; `pride` mirrored once; `trust` never.
   **Doctrine taught by one example becomes doctrine for that example.**
   The one mirrored pride (english-captain) is also the run's only new
   mechanical win (the pride probe).

4. **Synonym brittleness caps the upside.** The fear harvest routed
   `fear, fears` (gated, correct) — and "am I *afraid* of anything?"
   misses anyway: exact-token matching extends the inflection problem to
   synonyms, and no prompt sentence enumerates English. Descent doesn't
   have this problem — a reading model maps afraid→fear in the distillant.
   Empirically re-affirms the BFS test's boundary: **routing facets are
   opportunistic sugar for the literal paraphrase; distillants remain the
   lever.**

5. **Over-supply cost is budget-shaped, not unbounded — the hazard is
   crowding, and it's an ordering problem.** `TOTAL_LEAF_CAP = 12` bounds
   every injection (~5KB worst observed). Pulling all five regret
   midpoints for "do I have *any* regrets?" is the correct recall and
   fits. But "do I regret leaving home?" spends the budget on
   table-order-earlier matches (life 3 + dwelling 5 + voyages 4 = 12) and
   `people/parents` — the original-sin regret, the best answer — never
   renders. `RoutingTable::matches` returns insertion order (≈
   alphabetical), not relevance. Pre-existing trait; facet routing makes
   it visible. If doctrine lands, ranking matched midpoints (match count ×
   salience) is the companion fix.

6. **Maintenance hole: distillant rewrites orphan gated routing.** The
   harvest's `distills` op replaced friday's fresh distillant and left
   `loyal/loyalty/devotion/wept` routing with no distillant backing (same
   on OLD voyages). The gate is an emission-time property; nothing
   re-checks at rewrite. Post-run audit counts 6 orphaned facet terms on
   NEW, 9 on OLD — all from rewrites, not emissions. If the gate becomes
   doctrine, redistill/distills should prune facet routing the new
   distillant no longer carries (mechanically checkable — it's the same
   audit this run scripted).

7. **Side findings, both arms.** Rule-13's example phrasing imprints: "a
   standing X" appeared in 7 distillants across arms, and the OLD harvest
   *fabricated* "parting would be a standing regret" on Friday (he never
   parted — the example's content leaked, not just its shape). The NEW
   harvest leaked meta-language into a distillant ("fear and pride are the
   standing facets here") — prompt jargon in user-facing text; a "no
   meta-language in distillants" line is probably warranted regardless of
   the doctrine call. OLD money/trade emitted a wrong `merge_leaves`
   (brazil-stock into guinea-trader — not duplicates; applied faithfully,
   data folded): the haiku run's write-time-judgment gradient shows at
   sonnet tier too, at a lower rate. Negative control passed both arms:
   the annoyance turn produced zero facet terms.

## What this says about the doctrine question

The gate sentence is cheap, measurably prevents ungated spray (0 vs 2),
and produced the run's only new mechanical recall. Its measured upside
over the status quo is otherwise small — because the status quo already
mirrors the example facet — and the two genuinely open recall gaps (trust,
afraid) are not routing problems: one is under-mirroring the prompts can't
reliably fix, the other is synonymy that exact-token matching can never
fix. Those stay descent-tier, which is the BFS boundary holding, sharpened:
*distillants are the lever; routing compiles them, opportunistically.*

If adopted, the doctrine should ship as three pieces: (a) the gate
sentences as drafted; (b) orphan-pruning on distillant rewrite (finding
6); (c) optionally, relevance-ranked projection order (finding 5) — a
separate, pre-existing decision the budget crowding makes timely.

## Adopted (June 10, 2026)

The user took all three pieces, and the doctrine's invariant became
mechanical — the run itself showed prompts alone both under- and
over-apply it. `routing.rs::FACET_FAMILIES` defines the facet lexicon as
inflection families; the family is also the synonym bridge (a "fear"
distillant licenses routing "afraid"), which softens finding 4 on the
licensing side. `harvest::add_routing` rejects ungated single-word facet
terms (finding 2's two OLD violations would not have applied);
`prune_orphaned_facets` runs on every distillant rewrite — `distills`,
midpoint upsert, redistill — closing finding 6. Projection ranks matches
by lexical score then best-leaf salience under the same budget (finding
5): re-probed on the NEW arm, "do I regret leaving home?" now spends its
12 leaves on three regret-bearing midpoints instead of the dwelling
leaves. Under-mirroring (finding 3) stays prompt-level by design — a gate
can reject, not invent — so trust-class recall remains the descent tier's
job, as concluded.

## Auto-mirror re-measure (June 10, 2026)

Adoption left one over-influence problem standing: the prompt's example
facet decided which facets became mechanically recallable (finding 3 —
regret mirrored 5/5, trust 0/3). Fix: facet routing is now **derived**.
`sync_facet_routing` runs wherever a distillant is set or rewritten and
makes routing's single-word facet vocabulary exactly the union of families
the distillant carries — whole families, so a "fear" distillant routes
"afraid" and "dread". The model stopped writing facet routing; the mirror
instructions came out of the prompts, and the inflection example
de-anchored from "regret" to "payment". Less doctrine in the prompt is
itself the fix for doctrine over-influence.

Method: replay — the same 12 NEW-arm model outputs applied to a fresh
fixture through the new runtime. Zero new model calls; the mechanism is
the only variable.

| probe | NEW arm (model-mirrored) | derived (auto-mirror) |
|---|---|---|
| "do I have any regrets?" | 5 midpoints | 5 midpoints (unchanged) |
| "who do I trust most in the world?" | miss | **hit** — portuguese-captain (will, heir, steward) |
| "am I afraid of anything?" | miss | **hit** — pyrenees, via family breadth |
| "what do I dread?" | miss | **hit** — pyrenees |
| "is there anything I am proud of?" | episodes only | friday, money/trade, pyrenees |
| control (plantation) | 4 midpoints | 4 midpoints (unchanged) |
| junk: "there are tears in the mainsail canvas" | — | 1 spurious block (parents, weep family), ranked behind the topical stores match |

The discovery run's under-mirrors healed mechanically (money/trade pride,
portuguese-captain trust), and friday's orphaned facet terms self-corrected
on the first rewrite. Routing grew 403 → 455 terms (+13%). The honest
residual: polysemous family members ("tears", "cried") buy a small junk
tail — one budget-bounded, ranked-behind block on a fabric-tears message;
that is lexicon tuning (`FACET_FAMILIES` data), not architecture. Example
influence on distillant *wording* remains possible (the hygiene lines
guard it, prompt-level), but wording imprint is now nearly harmless: a
distillant that apes "standing regret" routes regret correctly, and no
example can grant or withhold a facet's recallability anymore.

## Artifacts

`/tmp/facet-discovery/` (renders, agent outputs, per-arm graphs incl.
`arm-mirror`, probe outputs, audit script in the session transcript). Arms
are /tmp-only; the fixture is untouched.
