# Franklin run: a real autobiography through the engine

**June 11, 2026.** Second corpus: *The Autobiography of Benjamin Franklin*
(Gutenberg #148, editor matter stripped, 65,082 words → 36 chunks of ~1,800
words). Same method as the Crusoe rebuild (`crusoe-rebuild-run.md`): Sonnet
subagents as harvester via the keyless CLI, one consolidation step per chunk,
10 settling steps at the end. Chosen because a real life with the famous
13-virtues project is a natural stress test for the profile machinery.

## The self-node incident

The first attempt filed the narrator under `people/benjamin-franklin` —
the harvester modeled **the user as a person in his own registry**, with his
intellect, origins, and career as children of a self-node. The run was
restarted with the rule made explicit, and the rule is now part of the
harvester contract itself (rule 3: *the user themself is never a distillant —
the whole graph already models them*). The Crusoe runs avoided this by
harvester judgment; nothing had ever stated it.

## Resulting graph

**92 distillants (81 registry / 11 profile), 278 leaves, 237 episodes.**
The civic life that dominates the book's second half structured itself:
`work/{printing,assembly-member,postmaster-general,electricity,academy,
defense-association,civic-projects,library,hospital}`, with the virtue
project landing in the profile as `temperament/virtue-project` (the 13
virtues, the ruled book, Order as the incorrigible weak spot, Pride as the
stubbornest passion) next to `temperament/{ethics,self-improvement,
civic-personality,expression}`.

## Interrogation: 10 questions — 8 GROUNDED-FULL, 2 PARTIAL, 0 MISS

| probe | projection | verdict |
|---|---|---|
| Governor Keith? | hit | FULL — promises, no-credit reveal, the fall |
| how did I learn to write? | (nothing) | PARTIAL — boyhood node had it all |
| brother James | hit | FULL — beatings, escape, deathbed reconciliation |
| moral perfection plan? | (nothing)* | FULL — *the pinned profile carried it |
| who is Deborah? | hit | FULL — first sight → erratum → helpmate |
| do I have any regrets? | **hit via facet** | FULL — the errata, by name |
| public projects? | (nothing) | PARTIAL — library/fire/hospital/academy via opens |
| religion? | hit | FULL — Deist at 15, five essentials, 1728 liturgy |
| printing business? | hit | FULL — Meredith, Coleman & Grace, Hall |
| electrical experiments? | hit | FULL — kite, Copley 1753, FRS, Nollet |

Six claims spot-verified verbatim against stored leaves (all present —
`keith-no-letters-confirmed`, `spectator-self-study`,
`deborah-london-erratum-corrected`, `thirteen-virtues`, `copley-medal-1753`).

Readings:

- **Franklin's "errata" routed as regret** — the facet derivation handled a
  period-specific idiom because the lines say what the errata *were*, and
  the regret family rides on the lines.
- **The moral-perfection probe never needed retrieval**: the virtue project
  lives in the pinned profile, which is the design answer for
  load-bearing self-model content — fire, not surface.
- The two projection blanks are the familiar vocabulary class ("write well",
  "public projects" share no exact token with any routing); BFS descent
  recovered both inside two opens.
