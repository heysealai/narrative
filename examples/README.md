# Examples

## crusoe/memory.json

The memory graph produced by feeding the full text of *Robinson Crusoe*
(Defoe, ~121k words, 66 chunks) through the harvest pipeline on the
drift-aware engine — the June 2026 rebuild described in
[`../docs/crusoe-rebuild-run.md`](../docs/crusoe-rebuild-run.md)
(50 distillants, 294 leaves, 339 episodes; the original run is
[`../docs/crusoe-tracking-report.md`](../docs/crusoe-tracking-report.md)).

Play with it:

```sh
NARRATIVE_DATA=examples/crusoe cargo run -- profile        # the character arc as axes
NARRATIVE_DATA=examples/crusoe cargo run -- map            # the registry skeleton
NARRATIVE_DATA=examples/crusoe cargo run -- project "whatever happened to Xury?"
NARRATIVE_DATA=examples/crusoe cargo run -- open people/friday
NARRATIVE_DATA=examples/crusoe cargo run -- stream 20
NARRATIVE_DATA=examples/crusoe cargo run -- stats
```

(The CLI expects `$NARRATIVE_DATA/memory.json`, so the file lives at
`examples/crusoe/memory.json`. `project` touches salience and saves — run
recall experiments against a copy, not the fixture.)
