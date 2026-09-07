# Harvest directory scope replay (September 2026)

The harvester's prompt renders the memory directory — the distillant layer —
on every absorb. Today that is every distillant with its line and its whole
routing vocabulary, so an absorb's input scales with the size of the memory
rather than with the turn: about 44k input tokens per absorb on a
112-distillant graph, paid by the user at the rate of the model they drive.
The design's own contract asks for less (`DESIGN.md`: the full directory
inline as labels plus aliases), and the owner's intent is more selective
still: a leaf-to-root traversal in which only the part of the memory the
turn touches is in view (heysealai/seal#169, with the compact directory as
heysealai/seal#168).

This replay measures what each scope costs and what it changes in what the
harvester writes. Same fifty turns, same starting graph, one run per scope.

**Verdict: the compact directory is free — it matches the full directory on
every quality count at 71% of the input. The selective directory at 36% of
the input minted one twin of an existing distillant in fifty turns, exactly
where the design note predicted (a branch's child described without its
name, opened by neither the routing match nor the label alone); widening the
neighborhood to the opened distillants' children closed that case in a
second run — see the last section for its numbers.**

## Scopes

Every scope shows every distillant by id, so a fact can always be filed
under an existing node. They differ in how much judgment rides along with
the id (`DirectoryScope` in `src/harvest.rs`):

| Scope | Every distillant carries | Opened neighborhood carries |
|---|---|---|
| `full` (today) | id, label, line, routing vocabulary | — |
| `compact` | id, label, routing vocabulary | — |
| `selective` | id, label | line and routing vocabulary |

The opened neighborhood is the routing match's hits (`RoutingTable::matches`
on the turn's text, both trees), their ancestors up to the root, and — in
the second selective run — the hits' children. The pinned profile leaves
and the comparanda are unchanged across scopes; episodes are never rendered.

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
  scope lives with what it wrote. The per-turn report records prompt size,
  token usage, stop reason, the op mix by section and relation, and what
  was minted.
- **Reading the result.** Distillant twins: id-plus-label token Jaccard
  against the cutoff graph and within the run, at 0.5 or above. Leaf twins:
  text token Jaccard the same way. Every twin flagged this way was read by
  hand.

## Cost

| | full | compact | selective (hits + ancestors) |
|---|---|---|---|
| prompt characters, median | 121,279 | 82,786 | 43,264 |
| input tokens, fifty absorbs | 2,214,858 | 1,564,111 | 797,832 |
| input tokens, median absorb | 44,032 | 31,076 | 15,332 |
| input tokens, largest absorb | 47,099 | 34,030 | 25,786 |
| output tokens, fifty absorbs | 44,212 | 54,247 | 45,670 |
| parse failures / retries / cap hits | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| relative input | 100% | 71% | 36% |

At the Sonnet rate a user pays for input ($2 per million tokens before
margin), the fifty absorbs' input is $4.43, $3.13 and $1.60. Output is
flat across scopes and about a tenth of the input bill.

The opened neighborhood on the cutoff graph (a property of the turn, the
same under every scope): hits plus ancestors open a median of 6 of the 112
distillants (8 turns open none, the largest opens 30); adding the hits'
children opens a median of 14 (the largest, a turn that hits a root, opens
95).

## What each scope wrote

| | full | compact | selective (hits + ancestors) |
|---|---|---|---|
| turns that emitted nothing | 19 | 15 | 19 |
| episodes | 20 | 24 | 24 |
| states, novel | 25 | 31 | 31 |
| states recognizing an existing leaf (supports, contradicts, supersedes) | 6 | 16 | 8 |
| reinforces | 20 | 18 | 14 |
| disposition nudges | 4 | 3 | 4 |
| distillants minted | 1 | 1 | 2 |
| minted distillants twinning an existing one | 0 | 0 | 1 |
| new leaves | 25 | 32 | 31 |
| new leaves twinning an existing leaf | 0 | 0 | 0 |
| new leaves twinning each other | 1 | 0 | 0 |
| graph after the run (distillants / leaves) | 113 / 429 | 113 / 436 | 114 / 435 |

All three scopes minted the same one new distillant on the same turn, under
the same parent, with lines that say the same thing: the placement of a
new topic did not depend on the lines being visible. Leaf-level quality is
flat: no scope duplicated an existing leaf, and the comparanda they judge
against are identical by construction.

Compact's larger recognizing count comes mostly from one turn in which the
user redesigned a tool of theirs: compact superseded the four recorded facts
the redesign retired, where full reinforced two leaves and logged the
episode. Dropping the lines did not cost the harvester the recognition the
design attributes to them.

## The divergence

Two turns apart, the user first redesigned one of their tools (naming it)
and then approved the redesign's flow (describing what the tool does
without naming it). On the second turn the routing match opened the parent
branch but not the tool's own distillant: the turn carried none of that
distillant's routing terms. Under `full` the harvester recognized the tool
from its line and logged an episode; under `compact` it recognized the tool
from its routing vocabulary and filed three states under it; under
`selective` the distillant was an index line — id and label — and the
harvester minted a sibling for the same tool with four states under it.

That is the failure the #169 design note predicted: the label alone does
not carry what a node is about, so a hit's child described in other words
gets a twin. It is a sibling-level miss (the parent was open), which is why
the second run widens the neighborhood to the hits' children.

## Selective with the hits' children

The shipped `selective` scope is the second run's: the neighborhood is the
hits, their ancestors, and the hits' children. (The first run's narrower
neighborhood is the same code without the children step.)

| | selective (hits + ancestors + children) |
|---|---|
| prompt characters, median | 49,322 |
| input tokens, fifty absorbs | 983,792 (44% of full) |
| input tokens, median absorb | 17,447 |
| input tokens, largest absorb | 40,918 |
| output tokens, fifty absorbs | 51,476 |
| parse failures / retries / cap hits | 0 / 0 / 0 |
| turns that emitted nothing | 19 |
| episodes | 18 |
| states, novel | 27 |
| states recognizing an existing leaf | 7 |
| reinforces | 15 |
| disposition nudges | 6 |
| distillants minted | 1 (the one every scope minted) |
| minted distillants twinning an existing one | 0 |
| new leaves | 28 |
| new leaves twinning an existing leaf / each other | 0 / 0 |
| graph after the run (distillants / leaves) | 113 / 432 |

On the divergence turn the parent's children were in view with their lines
and routing, and the harvester filed the approved flow as a state under the
tool's own distillant, as full and compact had. The children's price is
paid on the turns that hit a root: a root's children are most of a level,
so the largest absorb read 95 of the 112 distillants for 41k tokens, close
to full, while the median absorb stayed at 40% of full.

## What this says

- **Compact is free.** No measured loss against full on this slice, 29%
  less input, and it is the directory `DESIGN.md` already specifies. A host
  can switch to it now (heysealai/seal#168).
- **The selective shape is right, one level down included.** Leaf to root
  gives the context a fact is read in; the opened nodes' children are the
  siblings a new fact is placed among, and without them a child described
  in other words gets a twin. With them the harvester read 44% of full's
  input and wrote a graph of the same shape: the one distillant every
  scope minted, no twins, no duplicated leaves.
- **Two things this replay did not settle.** A root hit opens most of a
  level; rendering a root's children in compact form (routing, no line)
  would cap that, and deserves its own run. And the routing match is
  lexical: a turn whose words hit no routing term sees only the index.
  Eight turns here opened nothing. On six of them no scope wrote
  anything; on one every scope nudged a pinned disposition; on one the
  index alone was enough — selective filed the fact under the right
  existing distillant by id and label, as compact did, where full wrote
  nothing. Consolidation's merge of twin distillants remains the backstop
  for the case that did not arise: a node the index does not make
  recognizable.
- **Sample.** One user, one day, fifty turns of which thirty-one produced
  ops, one model. Enough to see the predicted failure appear and close;
  not enough to put a rate on it.


## Reproduce

```sh
# turns.json: {"turns": [{"seq", "at", "at_epoch", "user", "assistant"}, …]}
# NARRATIVE_DATA/memory.json: the starting graph (copied fresh per scope)
NARRATIVE_MODEL=claude-sonnet-5 NARRATIVE_EFFORT=medium NARRATIVE_DATA=run-full \
  narrative replay turns.json full report-full.json
```

The report is a JSON array with one entry per turn; `--mock` runs the same
loop against the scripted mock (no ops parse, so it measures prompt size
and the opened neighborhood only).
