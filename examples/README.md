# Examples

Three full-book memory graphs, each built by feeding a first-person text
through the harvest pipeline chunk by chunk (Sonnet subagents playing
harvester via the keyless CLI, one consolidation step per chunk, settling
passes at the end). Each has a run report under `../docs/`.

| graph | corpus | distillants | leaves | episodes | report |
|---|---|---|---|---|---|
| `crusoe/` | *Robinson Crusoe* (Defoe, ~121k words) | 50 (44R/6P) | 294 | 339 | [crusoe-rebuild-run.md](../docs/crusoe-rebuild-run.md) |
| `franklin/` | *Autobiography of Benjamin Franklin* (~65k) | 92 (81R/11P) | 278 | 237 | [franklin-run.md](../docs/franklin-run.md) |
| `antonia/` | *My Ántonia* (Cather, ~81k) | 54 (45R/9P) | 311 | 205 | [antonia-run.md](../docs/antonia-run.md) |

Each stresses a different part of the design: Crusoe the self-model and
states/supersession (the island arc), Franklin the profile (the 13-virtues
project becomes `temperament/virtue-project`) and a sprawling civic registry,
Ántonia the people registry (a four-level `people/antonia` subtree — the
user's memory is mostly about someone else).

Play with one:

```sh
NARRATIVE_DATA=examples/crusoe cargo run -- profile        # the character arc as axes
NARRATIVE_DATA=examples/franklin cargo run -- map          # the registry skeleton
NARRATIVE_DATA=examples/antonia cargo run -- open people/antonia
NARRATIVE_DATA=examples/crusoe cargo run -- project "whatever happened to Xury?"
NARRATIVE_DATA=examples/franklin cargo run -- project "do I have any regrets?"
NARRATIVE_DATA=examples/antonia cargo run -- stream 20
NARRATIVE_DATA=examples/crusoe cargo run -- stats
```

(The CLI expects `$NARRATIVE_DATA/memory.json`. `project` touches salience
and saves — run recall experiments against a copy, not the fixtures.)
