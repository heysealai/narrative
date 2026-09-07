# Harvest directory scope replay (September 2026)

The harvester's prompt renders the memory directory — the distillant layer —
on every absorb. Until this replay that was every distillant with its line
and its whole routing vocabulary, so an absorb's input scaled with the size
of the memory rather than with the turn: about 44k input tokens per absorb
on a 112-distillant graph, paid by the user at the rate of the model they
drive. The owner's intent was a leaf-to-root traversal in which only the
part of the memory the turn touches is in view (heysealai/seal#169; the
compact directory of heysealai/seal#168 was the interim candidate).

This replay measured what each scope costs and what it changes in what the
harvester writes: the same fifty turns from the same starting graph, one
run per scope, six runs. The leaf-to-root directory it settled on is now
the contract (`DESIGN.md`, write path; `DirectoryScope::Selective` in
`src/harvest.rs`).

**Verdict: the shipped directory — every distillant by id and label, the
branches the turn is on with line and routing, every hit's children with
their line — reads 40% of the full directory's input and writes the same
graph: the one distillant every scope minted, no twins, no duplicated
leaves. Two narrower neighborhoods each minted one twin of an existing
distillant, both on the same turn and for the same reason: the existing
node was an index line when the turn described it without naming it.
Compact (routing, no lines) matched full at 71% and is superseded.**

## Scopes

Every scope shows every distillant by id, so a fact can always be filed
under an existing node. They differ in how much judgment rides along with
the id:

| Scope | Every distillant carries | The turn's neighborhood carries |
|---|---|---|
| `full` | id, label, line, routing | — |
| `compact` | id, label, routing | — |
| `selective` | id, label | on the branches: line and routing; beside them: line |

The neighborhood (`Neighborhood` in `src/harvest.rs`) is read leaf to root
from the routing match's hits (`RoutingTable::matches` on the turn's text,
both trees). *On the branches*: the hits and every ancestor up to the root
— where the harvester files, and whose routing it extends. *Beside*: every
hit's children — the level the turn is on, the siblings a new fact lands
among, which the harvester recognizes by what they say. The pinned profile
leaves and the comparanda are unchanged across scopes; episodes are never
rendered.

Four selective neighborhoods were run, to find that shape:

| Run | On the branches | Beside |
|---|---|---|
| A | hits and ancestors, line and routing | nothing |
| B | same | every hit's children, line and routing |
| **C (shipped)** | same | every hit's children, line only |
| D | same | only the branch tips' children (hits no other hit sits under), line only |

## Method

- **Graph.** One live user's memory as of a cutoff on September 2, 2026:
  112 distillants (7 profile, 105 registry), 404 leaves, 258 episodes,
  with every node created after the cutoff dropped. The graph and the
  transcript belong to a real person; this report carries counts only.
- **Turns.** The fifty consecutive turns that followed the cutoff (17.5
  hours of one day; 3.3k characters of user text, 17.7k of assistant
  text). Each turn is the host's chunk shape: the user's row and the
  assistant rows that answered it, one speaker-prefixed line each.
- **Request.** The host's: the keyless prompt as one user message, no
  system block, no output schema, a 32,768-token response cap, one retry
  on an unparseable response, the host's brace-to-brace salvage. Model
  `claude-sonnet-5` at medium effort. Event time is the turn's own, for
  the render and the apply.
- **Replay.** `narrative replay turns.json <scope> report.json` from the
  cutoff graph, applying each turn's ops before the next turn renders, so a
  run lives with what it wrote. The per-turn report records prompt size,
  the neighborhood's two sizes, token usage, stop reason, the op mix by
  section and relation, and what was minted. Runs A, B and D are the
  shipped code with the beside set changed (none; line and routing; the
  tips' children only).
- **Reading the result.** Distillant twins: id-plus-label token Jaccard
  against the cutoff graph and between runs, at 0.5 or above, then every
  minted distillant read by hand (the Jaccard flag caught one of the two
  twins; the hand reading caught both). Leaf twins: text token Jaccard the
  same way.

## Cost

| | full | compact | A | B | C (shipped) | D |
|---|---|---|---|---|---|---|
| prompt characters, median | 121,279 | 82,786 | 43,264 | 49,322 | 46,307 | 45,196 |
| input tokens, fifty absorbs | 2,214,858 | 1,564,111 | 797,832 | 983,792 | 876,596 | 816,102 |
| input tokens, median absorb | 44,032 | 31,076 | 15,332 | 17,447 | 16,352 | 15,894 |
| input tokens, largest absorb | 47,099 | 34,030 | 25,786 | 40,918 | 32,079 | 26,168 |
| output tokens, fifty absorbs | 44,212 | 54,247 | 45,670 | 51,476 | 51,700 | 43,742 |
| parse failures / retries / cap hits | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| relative input | 100% | 71% | 36% | 44% | 40% | 37% |

At the Sonnet rate a user pays for input ($2 per million tokens before
margin), the fifty absorbs' input is $4.43 under full and $1.75 under the
shipped directory. Output is flat across scopes and about a tenth of the
input bill.

The neighborhood on this graph, a property of the turn: on the branches, a
median of 6 of the 112 distillants and at most 30; beside, a median of 10
and at most 65 (D: a median of 0, at most 59). Eight turns hit nothing.
The largest neighborhoods are turns that hit two or three of the five
roots at once: the roots carry 17 to 122 routing terms each, a person's
name routes to three of them, and a root's children are a whole level
(39 under one of them). That is where the shipped directory's largest
absorb (32k tokens) comes from.

## What each scope wrote

| | full | compact | A | B | C (shipped) | D |
|---|---|---|---|---|---|---|
| turns that emitted nothing | 19 | 15 | 19 | 19 | 19 | 20 |
| episodes | 20 | 24 | 24 | 18 | 18 | 18 |
| states, novel | 25 | 31 | 31 | 27 | 27 | 24 |
| states recognizing an existing leaf (supports, contradicts, supersedes, duplicate) | 6 | 16 | 8 | 7 | 9 | 7 |
| reinforces | 20 | 18 | 14 | 15 | 15 | 17 |
| disposition nudges | 4 | 3 | 4 | 6 | 4 | 4 |
| distillants minted | 1 | 1 | 2 | 1 | 1 | 2 |
| minted distillants twinning an existing one (hand-read) | 0 | 0 | 1 | 0 | 0 | 1 |
| new leaves | 25 | 32 | 31 | 28 | 28 | 24 |
| new leaves twinning an existing leaf | 0 | 0 | 0 | 0 | 0 | 0 |
| new leaves twinning each other | 1 | 0 | 0 | 0 | 0 | 1 |
| graph after the run (distillants / leaves) | 113 / 429 | 113 / 436 | 114 / 435 | 113 / 432 | 113 / 432 | 114 / 428 |

Every run minted the same one new distillant on the same turn, under the
same parent, with lines that say the same thing: placing a new topic did
not depend on the lines being visible. Leaf-level quality is flat: no run
duplicated an existing leaf, and the comparanda they judge against are
identical by construction.

Compact's larger recognizing count comes mostly from one turn in which the
user redesigned a tool of theirs: compact superseded the four recorded facts
the redesign retired, where full reinforced two leaves and logged the
episode. Dropping the lines did not cost the harvester the recognition the
old design attributed to them.

## The twin

Two turns apart, the user first redesigned one of their tools (naming it)
and then approved the redesign's flow (describing what the tool does
without naming it). On the second turn the routing match hit the tool's
parent branch and one of its other children, but not the tool's own
distillant: the turn carried none of its routing terms.

- Under `full` the harvester recognized the tool from its line and logged
  an episode; under `compact` from its routing vocabulary, filing three
  states under it.
- Under A the tool was an index line — id and label — and the harvester
  minted a sibling for the same tool with four states under it.
- Under B and C the parent was a hit, so its children were beside with
  their lines; the harvester filed the approved flow under the tool's own
  distillant. Recognition needed the line, not the routing: C matched B
  on every count above at 89% of B's input.
- Under D the parent was not a branch tip (a deeper hit sat under it), so
  its other children stayed index lines, and the twin came back.

That is the failure the #169 design note predicted, and it is what fixes
the neighborhood's shape: the level the turn is on is every hit's
children, a hit that another hit sits under included. Narrowing to the
tips saves 7% of input and costs the recognition the design exists for.

## What this says

- **The leaf-to-root directory is the contract.** It reads 40% of the
  full directory's input at the median absorb (16k tokens against 44k on
  this graph), and on this slice wrote the same graph. `DESIGN.md`
  describes it; `render_harvest_prompt` renders it for every host.
- **What its cost now scales with.** The neighborhood, and the
  neighborhood's size is set by the routing vocabularies: a root hit by a
  name or a generic word opens a whole level. The lever left is routing
  hygiene on the roots (`DESIGN.md`: routing is replaced, not accreted —
  the maintenance pass returns the complete set), not the render.
- **The index is the fallback, and it held.** The routing match is
  lexical; a turn whose words hit no routing term sees only the index.
  Eight turns here hit nothing: on six no run wrote anything, on one every
  run nudged a pinned disposition, on one the index alone placed the fact
  under the right existing node. Consolidation's merge of twin distillants
  remains the backstop for the case that did not arise: a node the index
  does not make recognizable and no hit brings into view.
- **Compact is superseded.** It matched full at 71% and never minted a
  twin, but the leaf-to-root directory beats it on cost with the same
  result; it stays in the engine as a replay scope.
- **Sample.** One user, one day, fifty turns of which thirty-one produced
  ops, one model. Enough to see the predicted failure appear, close, and
  reappear when the neighborhood was narrowed; not enough to put a rate on
  it.

## Reproduce

```sh
# turns.json: {"turns": [{"seq", "at", "at_epoch", "user", "assistant"}, …]}
# NARRATIVE_DATA/memory.json: the starting graph (copied fresh per run)
NARRATIVE_MODEL=claude-sonnet-5 NARRATIVE_EFFORT=medium NARRATIVE_DATA=run-selective \
  narrative replay turns.json selective report-selective.json
```

The report is a JSON array with one entry per turn; `--mock` runs the same
loop against the scripted mock (no ops parse, so it measures prompt size
and the neighborhood only).
