//! Consolidation: the descent step of "gradient descent with a model as the
//! optimizer". A pass re-distills a distillant (summary + routing from its
//! leaves) and may restructure: split by creating child distillants and moving
//! leaves under them, and merge leaves that turned out to be duplicates.
//!
//! Timing in the host couples to the context-eviction watermark; the sim
//! approximates that with an after-harvest pressure trigger (`auto_step`) —
//! one descent step on the worst offender per turn — plus manual /distill.

use std::fmt::Write as _;

use anyhow::Result;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::harvest::{self, Op};
use crate::llm::{ChatRequest, Llm};
use crate::model::{
    age_str, Graph, Id, Tree, COMPRESS_BATCH, DIGEST_MIN_CUT, DIGEST_WINDOW, MECH_DIGEST_PREFIX, PROFILE_APEX,
    STREAM_CAP,
};

const REDISTILL_SYSTEM: &str = "You maintain one distillant of a personal memory tree. \
A distillant's line is its behavioral median: one line that summarizes what the \
leaves below collectively say about the user — the central tendency, not a list. \
High-importance exceptions are not averaged away; if one leaf is a critical outlier, \
the line may flag it. Routing terms are the recall vocabulary: words or short \
phrases a future message about this topic would plausibly contain. Return the \
COMPLETE routing set you want this distillant to carry: it replaces the current \
one, so keep what is right, drop what is not, add what is missing. Routing is a \
referring-expression index, not a word list — names, aliases, anchors, topics, \
with the inflected forms a message would actually contain (payment AND payments; \
matching is exact, no stemming). Never keep episode detail (numbers, one-off \
phrasings, item labels) or generic phrases a message about anything could \
contain: every stray term routes unrelated messages here. Evaluative vocabulary \
(trust, regret, fear...) is derived from the line automatically and needs no \
routing entries.\n\
\n\
The line is also the scent retrieval follows: a question can only descend \
here if the relevant vocabulary appears in this line (or the routing). When the \
leaves carry evaluative weight — trust, regret, fear, pride, conflict — advertise \
it alongside the topic: 'sworn companion, sold to the captain — parting is a \
standing regret' routes a regret question; 'proves loyal' does not. The line \
itself is plain prose about the user — never machinery words (facet, routing, \
distillant) — and the example above shows shape, not wording to reuse.\n\
\n\
Ground every claim: the line you return must be supported by what is \
shown below — the leaves, the recent events, the child lines. The current \
line is a style reference, not a source: a claim that appears only \
there has no evidence and must be dropped, not carried forward. When the \
evidence has moved against the old line (a value changed, a belief \
contradicted, a tendency reversed), the new line follows the evidence.\n\
\n\
You may also restructure when the leaves have outgrown one node:\n\
- SPLIT: when distinct clusters live here (rule of thumb: >8 leaves, or the \
line needs an 'and'), declare child distillants in `distillants` and move each \
clustered leaf under its child with `moves` (a move replaces the leaf's whole \
parent set). Leaves you do not move stay where they are.\n\
- MERGE: when two leaves state the same fact, absorb the lesser into the \
canonical one with `merge_leaves`; evidence and belief counts combine.\n\
- MERGE AWAY: when this distillant turns out to be the same thing as another \
one listed under 'Same-named distillants elsewhere', name it in `merge_into`; \
this distillant is absorbed into it (leaves, children, routing) and deleted, and \
the line and routing you return are ignored. A distillant shown with no line \
was created bare (a fact named it before any pass wrote it): give it its first \
line, or merge it away.\n\
Restructure only on real pressure — a tidy distillant needs nothing but a fresh \
line. The line you return describes this distillant AFTER your changes.";

fn redistill_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "line": {"type": "string", "description": "one line, the behavioral median; advertise evaluative facets the leaves carry (trust, regret, fear, pride)"},
            "routing": {"type": "array", "items": {"type": "string"}, "description": "the COMPLETE recall vocabulary this distillant should carry (replaces the current set): referring expressions, aliases, anchors, topics; evaluative vocabulary is derived from the line automatically"},
            "merge_into": {"type": "string", "description": "id of an existing distillant this one duplicates (from 'Same-named distillants elsewhere'); this distillant is absorbed into it and deleted. Empty string when it stands on its own."},
            "distillants": {
                "description": "New child distillants to split this one into; empty when no split is needed.",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "path-like kebab id, e.g. life/island/animals"},
                        "label": {"type": "string"},
                        "line": {"type": "string"},
                        "routing": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["id", "label", "line", "routing"],
                    "additionalProperties": false
                }
            },
            "moves": {
                "description": "Re-home leaves (typically under the new children). A move replaces the leaf's full parent set.",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "leaf": {"type": "string"},
                        "parents": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["leaf", "parents"],
                    "additionalProperties": false
                }
            },
            "merge_leaves": {
                "description": "Absorb duplicate leaves: 'from' is deleted, its evidence and counts fold into 'into'.",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "from": {"type": "string"},
                        "into": {"type": "string"}
                    },
                    "required": ["from", "into"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["line", "routing", "merge_into", "distillants", "moves", "merge_leaves"],
        "additionalProperties": false
    })
}

#[derive(Deserialize)]
struct Redistilled {
    line: String,
    routing: Vec<String>,
    #[serde(default)]
    merge_into: String,
    #[serde(default)]
    distillants: Vec<NewChild>,
    #[serde(default)]
    moves: Vec<MoveSpec>,
    #[serde(default)]
    merge_leaves: Vec<MergeSpec>,
}

#[derive(Deserialize)]
struct NewChild {
    id: String,
    label: String,
    line: String,
    routing: Vec<String>,
}

#[derive(Deserialize)]
struct MoveSpec {
    leaf: String,
    parents: Vec<String>,
}

#[derive(Deserialize)]
struct MergeSpec {
    from: String,
    into: String,
}

/// The redistill input: the distillant's current line and routing, its leaves,
/// recent episodes tagged here, and its children. None when the distillant
/// does not exist.
fn redistill_body(graph: &Graph, distillant_id: &str) -> Option<String> {
    let m = graph.distillants.get(distillant_id)?;
    let current_line = if m.is_bare() {
        "(none — created bare; write its first line, or merge it away)".to_string()
    } else {
        m.line.clone()
    };
    let mut body = format!(
        "Distillant: {} (label: {})\nCurrent line: {}\nCurrent routing: {}\n",
        m.id,
        m.label,
        current_line,
        m.routing.join(", ")
    );
    if m.id == PROFILE_APEX {
        body.push_str(
            "This is the profile apex: its line is the whole-person estimate — who this person is, \
             drawn only from the axis lines below (how they tend, how they talk, how they handle money). \
             One dense line of character, not a list of the axes. Where the axes pull against each \
             other (tight with themselves, open-handed with others), the tension is the character: \
             state it, never average it away.\n",
        );
    }
    body.push_str("\nLeaves:\n");
    let leaves = graph.leaves_under(distillant_id);
    if leaves.is_empty() {
        body.push_str("(none)\n");
    }
    for l in &leaves {
        match l.kind.salience() {
            Some(s) => {
                let _ = writeln!(body, "- [{}] {} (importance {:.1})", l.id, l.text, s.importance);
            }
            None => {
                let _ = writeln!(body, "- [{}] {}", l.id, l.text);
            }
        }
    }
    let recent: Vec<String> = graph
        .episodes
        .iter()
        .rev()
        .filter(|e| e.tags.iter().any(|t| t == distillant_id))
        .take(8)
        .map(|e| format!("- {}", e.text))
        .collect();
    if !recent.is_empty() {
        body.push_str("Recent events here (absorb what they say into the line):\n");
        for line in recent.iter().rev() {
            let _ = writeln!(body, "{line}");
        }
    }
    let children = graph.child_distillants(distillant_id);
    if !children.is_empty() {
        body.push_str("Child distillants:\n");
        for c in children {
            let _ = writeln!(body, "- {} — {}", c.id, c.headline());
        }
    }
    let namesakes = same_named_elsewhere(graph, distillant_id);
    if !namesakes.is_empty() {
        body.push_str("Same-named distillants elsewhere (merge_into one of these if this is the same thing):\n");
        for n in namesakes {
            let _ = writeln!(body, "- {} — {}", n.id, n.headline());
        }
    }
    Some(body)
}

/// Distillants in the same tree whose terminal path segment equals this
/// one's — the shape a duplicate takes when a harvest names a topic under
/// a second parent (`work/x-report` beside `life/hacks/x-report`). Shown
/// to the pass so a merge is a choice it can make, never something it
/// would have to already know about.
fn same_named_elsewhere<'g>(graph: &'g Graph, distillant_id: &str) -> Vec<&'g crate::model::Distillant> {
    let Some(m) = graph.distillants.get(distillant_id) else { return Vec::new() };
    let segment = |id: &str| id.rsplit('/').next().unwrap_or(id).to_string();
    let mine = segment(distillant_id);
    let mut out: Vec<&crate::model::Distillant> = graph
        .distillants
        .values()
        .filter(|o| o.id != distillant_id && o.tree == m.tree && segment(&o.id) == mine)
        .collect();
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

/// The keyless prompt: system contract, the exact output schema, and one
/// distillant's input — self-contained; a driver holding only this render
/// can play the consolidation role.
pub fn render_redistill_prompt(graph: &Graph, distillant_id: &str) -> Option<String> {
    let body = redistill_body(graph, distillant_id)?;
    Some(format!(
        "# System\n{REDISTILL_SYSTEM}\n\n# Output schema (reply with one JSON object matching it)\n{}\n\n# Input\n{body}",
        serde_json::to_string_pretty(&redistill_schema()).expect("static schema serializes")
    ))
}

/// Apply one redistill response (the `redistill_schema` shape) to a
/// distillant. Shared by the model-driven pass and the keyless
/// `narrative redistill` subcommand.
pub fn apply_redistilled(
    graph: &mut Graph,
    distillant_id: &str,
    raw: &str,
    now: u64,
) -> Result<Vec<String>> {
    let Some(m) = graph.distillants.get(distillant_id) else {
        anyhow::bail!("no distillant {distillant_id}");
    };
    let tree = m.tree;
    let out: Redistilled = serde_json::from_str(raw)?;

    // A merge-away ends the pass: the distillant is gone, its line and
    // routing moot. An invalid target (missing, self, other tree) is traced
    // by the merge and the pass continues as a plain redistill.
    let merges_away = !out.merge_into.trim().is_empty();
    if merges_away {
        let into = out.merge_into.trim().to_string();
        let mut traces = harvest::apply_ops(
            graph,
            vec![Op::MergeDistillant { from: distillant_id.to_string(), into }],
            now,
        )
        .traces;
        let merged = !graph.distillants.contains_key(distillant_id);
        if merged {
            return Ok(traces);
        }
        traces.push(format!("merge_into rejected — {distillant_id} redistilled in place"));
        return apply_redistilled_in_place(graph, distillant_id, tree, out, now, traces);
    }
    apply_redistilled_in_place(graph, distillant_id, tree, out, now, Vec::new())
}

fn apply_redistilled_in_place(
    graph: &mut Graph,
    distillant_id: &str,
    tree: Tree,
    out: Redistilled,
    now: u64,
    mut traces: Vec<String>,
) -> Result<Vec<String>> {
    let new_children: Vec<String> = out.distillants.iter().map(|c| c.id.clone()).collect();

    // Structural ops first; the returned line describes the result.
    let mut ops: Vec<Op> = Vec::new();
    for c in out.distillants {
        ops.push(Op::Distillant {
            id: c.id,
            tree,
            label: c.label,
            line: c.line,
            routing: c.routing,
            parents: vec![distillant_id.to_string()],
        });
    }
    for mv in out.moves {
        ops.push(Op::Move { leaf: mv.leaf, parents: mv.parents });
    }
    for mg in out.merge_leaves {
        ops.push(Op::MergeLeaf { from: mg.from, into: mg.into });
    }
    if !ops.is_empty() {
        traces.extend(harvest::apply_ops(graph, ops, now).traces);
    }

    if let Some(m) = graph.distillants.get_mut(distillant_id)
        && m.line != out.line
    {
        traces.push(format!("✎ distill {} — {}", distillant_id, out.line));
        m.line = out.line;
        // Material change: parents now hold drift. A rewrite that comes back
        // identical leaves the stamp alone — the cascade stops here.
        m.line_changed_at = now;
    }
    // Routing is REPLACED, never accreted: the response carries the complete
    // topical set, and routing only ever grew before this pass (harvest
    // aliases append). The fresh line stays the source of facet truth —
    // the derived families are mirrored from it, then the response's terms
    // enter through the same gate as always.
    if let Some(m) = graph.distillants.get_mut(distillant_id) {
        let dropped: Vec<String> = m
            .routing
            .iter()
            .filter(|t| !out.routing.iter().any(|r| r.trim().eq_ignore_ascii_case(t)))
            .filter(|t| crate::routing::facet_family(t).is_none())
            .cloned()
            .collect();
        m.routing.clear();
        if !dropped.is_empty() {
            traces.push(format!("✂ routing {} − {} (not in the pass's set)", distillant_id, dropped.join(", ")));
        }
    }
    harvest::sync_facet_routing(graph, distillant_id, &mut traces);
    harvest::add_routing(graph, distillant_id, &out.routing, &mut traces);
    if let Some(m) = graph.distillants.get_mut(distillant_id) {
        m.misc_count = 0;
        m.consolidated_at = now;
    }
    // Children born of this pass carry a fresh line by construction;
    // stamping them keeps the trigger from re-firing on a just-made split.
    for c in &new_children {
        if let Some(m) = graph.distillants.get_mut(c) {
            m.consolidated_at = now;
        }
    }
    if traces.is_empty() {
        traces.push(format!("line of {distillant_id} already fresh"));
    }
    Ok(traces)
}

/// One model-driven descent step on a distillant: render its input, ask for
/// the redistill, apply it.
pub fn redistill(llm: &dyn Llm, graph: &mut Graph, distillant_id: &str, now: u64) -> Result<Vec<String>> {
    let Some(body) = redistill_body(graph, distillant_id) else {
        return Ok(vec![format!("no distillant {distillant_id}")]);
    };
    let req = ChatRequest {
        system: REDISTILL_SYSTEM.to_string(),
        messages: vec![json!({"role": "user", "content": body})],
        tools: vec![],
        output_schema: Some(redistill_schema()),
        max_tokens: 2048,
        thinking: false,
    };
    let resp = llm.chat(&req)?;
    apply_redistilled(graph, distillant_id, &resp.text(), now)
}

/// The split rule of thumb: more leaves than this directly under one distillant
/// is fat (the same number REDISTILL_SYSTEM quotes to the model).
pub const FAT_LEAF_THRESHOLD: usize = 8;
/// Pressure at which the automatic after-harvest pass fires.
pub const PRESSURE_TRIGGER: u32 = 3;
/// Pressure a bare distillant carries by itself: no line means no cached
/// judgment at all, which is the trigger's worth of staleness on its own.
pub const BARE_PRESSURE: u32 = PRESSURE_TRIGGER;

/// Drift weight of a leaf that moved against its own text since the pass
/// (contradiction, supersession, disposition flip) — the strongest staleness
/// evidence, two such leaves alone reach the trigger.
pub const DRIFT_AGAINST: u32 = 2;
/// Drift weight of a child whose line text changed since the pass —
/// the parent summarized a line that no longer exists. One child among
/// several leaves and children is a fraction of the evidence.
pub const DRIFT_CHILD_LINE: u32 = 1;
/// Drift weight of that same changed child line when the distillant holds
/// no leaves of its own. Such a line summarizes its children and nothing
/// else, so one child moving is not a fraction of the evidence — it is the
/// evidence base moving, and that is a pass due on its own. The profile
/// apex is the standing case: it never holds leaves (a disposition belongs
/// to an axis), so its axis lines are all it has to be right about.
pub const DRIFT_CHILD_LINE_SOLE_EVIDENCE: u32 = PRESSURE_TRIGGER;

/// Semantic drift on one distillant: evidence that the cached line no
/// longer follows from what hangs below it. Derived, never stored — the
/// timestamps already in the graph are compared against `consolidated_at`,
/// so a redistill pass zeroes it by construction. Counts leaves that moved
/// against their own text, a forget that removed something held here (the
/// line summarized what no longer exists), and children whose lines
/// materially changed; mere reinforcement and new leaves are not drift
/// (new leaves are fat). A changed child weighs by what else the
/// distillant has: a fraction of the evidence beside its own leaves, the
/// whole of it when there are none.
pub fn drift(graph: &Graph, distillant_id: &str) -> u32 {
    let Some(m) = graph.distillants.get(distillant_id) else { return 0 };
    let t = m.consolidated_at;
    let leaves = graph.leaves_under(distillant_id);
    let against = leaves.iter().filter(|l| l.kind.belief().is_some_and(|b| b.last_against_at > t)).count() as u32;
    let child_lines = graph
        .child_distillants(distillant_id)
        .iter()
        .filter(|c| c.line_changed_at > t)
        .count() as u32;
    let forgotten_under = u32::from(m.forgotten_at > t);
    let children_are_the_whole_evidence = leaves.is_empty();
    let child_weight = if children_are_the_whole_evidence {
        DRIFT_CHILD_LINE_SOLE_EVIDENCE
    } else {
        DRIFT_CHILD_LINE
    };
    (against + forgotten_under) * DRIFT_AGAINST + child_lines * child_weight
}

/// Consolidation pressure on one distillant: how far past fat it has grown,
/// plus its residual counter (facts parked here for lack of anywhere better),
/// plus semantic drift (the cached line's evidence moved on), plus the
/// trigger's worth when it is bare (no line was ever written). One queue:
/// structural rot and semantic rot compete for the same per-turn step.
pub fn pressure(graph: &Graph, distillant_id: &str) -> u32 {
    let Some(m) = graph.distillants.get(distillant_id) else { return 0 };
    let fat = graph.leaves_under(distillant_id).len().saturating_sub(FAT_LEAF_THRESHOLD);
    let bare = if m.is_bare() { BARE_PRESSURE } else { 0 };
    fat as u32 + m.misc_count + drift(graph, distillant_id) + bare
}

/// Due = at/over the trigger AND something actually happened since the last
/// pass — new material under it, or drift (which is since-the-pass by
/// construction). A fat distillant the model already declined to restructure
/// stays quiet until something new lands — that, not a clock, is the
/// thrash guard.
fn is_due(graph: &Graph, m: &crate::model::Distillant) -> bool {
    pressure(graph, &m.id) >= PRESSURE_TRIGGER
        && (drift(graph, &m.id) > 0
            || graph.leaves_under(&m.id).iter().any(|l| l.updated_at > m.consolidated_at))
}

/// The distillant the next automatic pass should hit: the worst offender,
/// then descend — when a due distillant has a due child, the child goes
/// first, because the parent's rewrite consumes child lines and would
/// otherwise inherit a stale one. Freshness flows leaves-up.
pub fn due(graph: &Graph) -> Option<String> {
    let mut cur = graph
        .distillants
        .values()
        .filter(|m| is_due(graph, m))
        .max_by_key(|m| pressure(graph, &m.id))
        .map(|m| m.id.clone())?;
    // Bounded walk: the graph is a DAG, but don't bet a loop on it.
    for _ in 0..32 {
        let Some(child) = graph
            .child_distillants(&cur)
            .into_iter()
            .filter(|c| is_due(graph, c))
            .max_by_key(|c| pressure(graph, &c.id))
        else {
            break;
        };
        cur = child.id.clone();
    }
    Some(cur)
}

/// One automatic consolidation step, run after each harvest. Distillant
/// pressure first (it shapes recall); otherwise one stream step — the
/// stream's hard cap backstops a starved queue. One step per turn bounds
/// the cost — the sim-side stand-in for the host's eviction-watermark
/// coupling.
pub fn auto_step(llm: &dyn Llm, graph: &mut Graph, now: u64) -> Result<Vec<String>> {
    if let Some(id) = due(graph) {
        let mut traces = vec![format!("⚙ consolidating {} (pressure {})", id, pressure(graph, &id))];
        traces.extend(redistill(llm, graph, &id, now)?);
        return Ok(traces);
    }
    distill_stream(llm, graph, now)
}

// ---------------------------------------------------------------------------
// The stream species: digest distillation.
// ---------------------------------------------------------------------------

const DIGEST_SYSTEM: &str = "You compress the oldest period of a personal memory \
stream into one digest episode. Write a short chronological summary — a few lines, \
1500 characters at most — preserving what would matter later: names, numbers, \
dates, places, decisions, and evaluative weight (trust, regret, fear, pride, \
conflict). Drop pleasantries and play-by-play; keep consequences. If the input is \
itself an earlier mechanical digest (fragments marked '[digest of ...]'), rewrite \
the fragments into the same kind of readable summary — never keep the fragment \
markers.\n\
\n\
When the input is a numbered episode listing, you also choose where the period \
ends: return take, how many of the listed episodes (oldest-first) your digest \
covers. Cut at a natural boundary — a voyage, a year, a move, a project closing — \
within the stated bounds; when no boundary stands out, take 100. Digest only \
what you take.";

fn digest_schema(cut: bool) -> Value {
    let mut schema = json!({
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "the digest episode text: short, chronological; names, numbers, decisions and evaluative weight preserved"}
        },
        "required": ["text"],
        "additionalProperties": false
    });
    if cut {
        schema["properties"]["take"] = json!({
            "type": "integer",
            "description": "how many of the numbered episodes, oldest-first, the digest covers — cut at a natural period boundary within the stated bounds; 100 when none stands out"
        });
        schema["required"] = json!(["text", "take"]);
    }
    schema
}

#[derive(Deserialize)]
struct Digested {
    text: String,
    #[serde(default)]
    take: Option<usize>,
}

/// What the next stream consolidation step should do.
pub enum StreamStep {
    /// Stream past the soft cap: distill the oldest period into a digest
    /// while the full episode texts are still alive. Carries the cut window —
    /// how many of the oldest episodes the model sees; the model picks how
    /// many of them the digest actually covers.
    Fold(usize),
    /// A mechanical (fragments-only) digest sits in the stream — a hard-cap
    /// emergency fold, or one predating this species. Polish what survived.
    Polish(Id),
}

impl StreamStep {
    pub fn describe(&self, graph: &Graph) -> String {
        match self {
            StreamStep::Fold(window) => format!(
                "stream digest due: cut a period within the oldest {window} of {} episodes (soft cap {STREAM_CAP})",
                graph.episodes.len()
            ),
            StreamStep::Polish(id) => format!("stream digest due: polish mechanical [{id}]"),
        }
    }
}

/// The stream trigger. Folds before polishes: an over-cap stream is losing
/// the race to the hard cap; a mechanical digest is already as lossy as it
/// will ever get.
pub fn stream_due(graph: &Graph) -> Option<StreamStep> {
    if graph.episodes.len() > STREAM_CAP {
        return Some(StreamStep::Fold(DIGEST_WINDOW.min(graph.episodes.len())));
    }
    graph
        .episodes
        .iter()
        .find(|e| e.text.starts_with(MECH_DIGEST_PREFIX))
        .map(|e| StreamStep::Polish(e.id.clone()))
}

/// The digest input: full texts for a fold, the surviving fragments for a
/// polish. Also the keyless `digest-prompt` body.
pub fn digest_body(graph: &Graph, step: &StreamStep, now: u64) -> String {
    let mut body = String::new();
    match step {
        StreamStep::Fold(window) => {
            let lo = DIGEST_MIN_CUT.min(*window);
            let _ = writeln!(
                body,
                "# Stream period to digest ({window} oldest episodes, numbered oldest-first)"
            );
            for (i, e) in graph.episodes.iter().take(*window).enumerate() {
                let _ = writeln!(body, "{}. ({}) {}", i + 1, age_str(now, e.event_at()), e.text);
            }
            let _ = writeln!(body, "\nCut bounds: take between {lo} and {window}.");
        }
        StreamStep::Polish(id) => {
            let _ = writeln!(body, "# Mechanical digest to polish");
            if let Some(e) = graph.episodes.iter().find(|e| e.id == *id) {
                let _ = writeln!(body, "({}) {}", age_str(now, e.event_at()), e.text);
            }
        }
    }
    body
}

/// Apply distilled digest text to whatever stream step is due. Shared by the
/// model-driven pass and the keyless `narrative digest` subcommand. For a
/// fold, `take` is the model's cut — how many of the windowed episodes the
/// digest covers — clamped to the window's bounds; absent, the default batch.
pub fn apply_digest(
    graph: &mut Graph,
    text: String,
    take: Option<usize>,
    _now: u64,
) -> Result<Vec<String>> {
    let Some(step) = stream_due(graph) else {
        anyhow::bail!("no stream digest due");
    };
    let mut text = text.trim().to_string();
    if text.is_empty() {
        anyhow::bail!("empty digest text");
    }
    // A digest that still reads as mechanical would re-trigger the polish
    // pass forever; the marker is load-bearing, so disarm it.
    if text.starts_with(MECH_DIGEST_PREFIX) {
        text = format!("(distilled) {text}");
    }
    match step {
        StreamStep::Fold(window) => {
            let lo = DIGEST_MIN_CUT.min(window);
            let take = take.unwrap_or(COMPRESS_BATCH).clamp(lo, window);
            let id = graph.fold_oldest(take, Some(text));
            Ok(vec![format!("⚙ stream digest: folded {take} episodes into [{id}]")])
        }
        StreamStep::Polish(id) => {
            // Digests are period summaries, not events — the one episode
            // species whose text may be rewritten.
            if let Some(e) = graph.episodes.iter_mut().find(|e| e.id == id) {
                e.text = text;
            }
            Ok(vec![format!("⚙ stream digest: polished [{id}]")])
        }
    }
}

/// One model-driven stream step: render the due input, ask for the digest,
/// apply it.
pub fn distill_stream(llm: &dyn Llm, graph: &mut Graph, now: u64) -> Result<Vec<String>> {
    let Some(step) = stream_due(graph) else { return Ok(Vec::new()) };
    let body = digest_body(graph, &step, now);
    let req = ChatRequest {
        system: DIGEST_SYSTEM.to_string(),
        messages: vec![json!({"role": "user", "content": body})],
        tools: vec![],
        output_schema: Some(digest_schema(matches!(step, StreamStep::Fold(_)))),
        max_tokens: 2048,
        thinking: false,
    };
    let resp = llm.chat(&req)?;
    let out: Digested = serde_json::from_str(&resp.text())?;
    apply_digest(graph, out.text, out.take, now)
}

/// The keyless prompt: system contract plus the due input, self-contained.
pub fn render_digest_prompt(graph: &Graph, step: &StreamStep, now: u64) -> String {
    format!("# System\n{DIGEST_SYSTEM}\n\n# Input\n{}", digest_body(graph, step, now))
}

/// Residual report: where is consolidation pressure building?
pub fn stats(graph: &Graph) -> String {
    let mut out = String::new();
    let n_reg = graph.distillants.values().filter(|m| m.tree == Tree::Registry).count();
    let n_prof = graph.distillants.values().filter(|m| m.tree == Tree::Profile).count();
    let n_rules = graph.rules().len();
    let _ = writeln!(
        out,
        "distillants: {} registry / {} profile · leaves: {} ({} rules) · episodes: {}",
        n_reg,
        n_prof,
        graph.leaves.len() - n_rules,
        n_rules,
        graph.episodes.len()
    );

    let mut fat: Vec<(usize, &str)> = graph
        .distillants
        .keys()
        .map(|id| (graph.leaves_under(id).len(), id.as_str()))
        .filter(|(n, _)| *n > 0)
        .collect();
    fat.sort_by_key(|(n, _)| std::cmp::Reverse(*n));
    if !fat.is_empty() {
        let _ = writeln!(out, "fattest distillants (split candidates when >{FAT_LEAF_THRESHOLD}):");
        for (n, id) in fat.iter().take(5) {
            let _ = writeln!(out, "  {id}: {n} leaves");
        }
    }

    let mut residual: Vec<(&str, u32)> = graph
        .distillants
        .values()
        .filter(|m| m.misc_count > 0)
        .map(|m| (m.id.as_str(), m.misc_count))
        .collect();
    residual.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    if !residual.is_empty() {
        out.push_str("residual pressure (facts parked on crown roots — needs distillants):\n");
        for (id, n) in residual {
            let _ = writeln!(out, "  {id}: {n}");
        }
    }

    let mut bare: Vec<&str> = graph
        .distillants
        .values()
        .filter(|m| m.is_bare())
        .map(|m| m.id.as_str())
        .collect();
    bare.sort();
    if !bare.is_empty() {
        let _ = writeln!(out, "bare distillants (no line yet — a pass writes one or merges them away): {}", bare.join(", "));
    }

    let mut drifted: Vec<(&str, u32)> = graph
        .distillants
        .values()
        .map(|m| (m.id.as_str(), drift(graph, &m.id)))
        .filter(|(_, d)| *d > 0)
        .collect();
    drifted.sort_by_key(|(_, d)| std::cmp::Reverse(*d));
    if !drifted.is_empty() {
        out.push_str("semantic drift (the cached line's evidence moved on):\n");
        for (id, d) in drifted {
            let _ = writeln!(out, "  {id}: drift {d}");
        }
    }

    let orphans: Vec<&str> = graph
        .leaves
        .values()
        .filter(|l| !l.is_rule())
        .filter(|l| l.parents.iter().all(|p| !graph.distillants.contains_key(p)))
        .map(|l| l.id.as_str())
        .collect();
    if !orphans.is_empty() {
        let _ = writeln!(out, "orphan leaves (no surviving parent): {}", orphans.join(", "));
    }
    match due(graph) {
        Some(id) => {
            let _ = writeln!(
                out,
                "next auto pass: {id} (pressure {}, trigger {PRESSURE_TRIGGER})",
                pressure(graph, &id)
            );
        }
        None => match stream_due(graph) {
            Some(step) => {
                let _ = writeln!(out, "next auto pass: {}", step.describe(graph));
            }
            None => {
                let _ = writeln!(
                    out,
                    "next auto pass: none due (fires at pressure ≥ {PRESSURE_TRIGGER} with new material)"
                );
            }
        },
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::MockLlm;
    use crate::model::{Graph, Leaf};

    #[test]
    fn redistill_applies_model_output() {
        let mut g = Graph::seed();
        let l = Leaf::state("rent-amount".into(), "Rent is $2,200.".into(), vec!["money".into()], 0.5, 1_000);
        g.leaves.insert(l.id.clone(), l);
        g.distillants.get_mut("money").unwrap().misc_count = 3;

        let llm = MockLlm::scripted(vec![json!([{
            "type": "text",
            "text": "{\"line\": \"Rent dominates; $2,200 monthly.\", \"routing\": [\"lease\"]}"
        }])]);
        let traces = redistill(&llm, &mut g, "money", 2_000).unwrap();
        assert!(traces.iter().any(|t| t.contains("Rent dominates")));
        let m = &g.distillants["money"];
        assert_eq!(m.line, "Rent dominates; $2,200 monthly.");
        assert!(m.routing.iter().any(|r| r == "lease"));
        assert_eq!(m.misc_count, 0, "residual cleared after a descent step");
        assert_eq!(m.consolidated_at, 2_000, "pass stamped");
    }

    #[test]
    fn redistill_prompt_and_apply_are_the_keyless_pair() {
        let mut g = Graph::seed();
        let l = Leaf::state("rent-amount".into(), "Rent is $2,200.".into(), vec!["money".into()], 0.5, 1_000);
        g.leaves.insert(l.id.clone(), l);

        let p = render_redistill_prompt(&g, "money").unwrap();
        assert!(p.starts_with("# System\n"), "self-contained contract: {p}");
        assert!(p.contains("# Output schema"), "schema inline: {p}");
        assert!(p.contains("\"merge_leaves\""), "schema fields present");
        assert!(p.contains("Rent is $2,200."), "leaves rendered: {p}");
        assert!(render_redistill_prompt(&g, "ghost").is_none());

        let raw = r#"{"line": "Rent dominates.", "routing": ["lease"]}"#;
        let traces = apply_redistilled(&mut g, "money", raw, 2_000).unwrap();
        assert!(traces.iter().any(|t| t.contains("Rent dominates")), "{traces:?}");
        assert!(g.distillants["money"].routing.iter().any(|r| r == "lease"));
        assert_eq!(g.distillants["money"].consolidated_at, 2_000);
        assert!(apply_redistilled(&mut g, "ghost", raw, 2_000).is_err());
    }

    #[test]
    fn redistill_gates_and_prunes_facet_routing() {
        let mut g = Graph::seed();
        {
            let m = g.distillants.get_mut("money").unwrap();
            m.line = "A standing regret about the lease.".into();
            m.routing = vec!["regret".into(), "lease".into()];
        }
        let raw = r#"{"line": "Rent dominates.", "routing": ["regrets", "trust", "landlord"]}"#;
        let traces = apply_redistilled(&mut g, "money", raw, 2_000).unwrap();
        let r = &g.distillants["money"].routing;
        assert!(!r.contains(&"regret".to_string()), "orphan of the old line pruned: {traces:?}");
        assert!(!r.contains(&"regrets".to_string()), "new facet gated by the new line");
        assert!(!r.contains(&"trust".to_string()), "gated");
        assert!(!r.contains(&"lease".to_string()), "a term the pass did not return is dropped: routing is replaced, not accreted: {traces:?}");
        assert!(r.contains(&"landlord".to_string()));
    }

    #[test]
    fn redistill_derives_facet_routing_from_the_new_line() {
        let mut g = Graph::seed();
        let raw = r#"{"line": "The landlord is the one soul he trusts.", "routing": []}"#;
        apply_redistilled(&mut g, "money", raw, 2_000).unwrap();
        let r = &g.distillants["money"].routing;
        assert!(
            r.contains(&"trust".to_string()) && r.contains(&"trusted".to_string()),
            "trust family derived with no model-written routing at all: {r:?}"
        );
    }

    #[test]
    fn redistill_can_split_and_merge() {
        let mut g = Graph::seed();
        for (id, text) in [
            ("island-goats", "Keeps a herd of tame goats."),
            ("island-pets", "Has a parrot named Poll."),
            ("rent-a", "Rent is $2,200/mo."),
            ("rent-b", "Rent: $2,200 monthly."),
        ] {
            let l = Leaf::state(id.into(), text.into(), vec!["life".into()], 0.5, 1_000);
            g.leaves.insert(l.id.clone(), l);
        }
        let out = json!({
            "line": "Island life: the animals now live under their own distillant.",
            "routing": ["island"],
            "distillants": [{
                "id": "life/island/animals",
                "label": "Island animals",
                "line": "The island menagerie.",
                "routing": ["goats", "parrot"]
            }],
            "moves": [
                {"leaf": "island-goats", "parents": ["life/island/animals"]},
                {"leaf": "island-pets", "parents": ["life/island/animals"]}
            ],
            "merge_leaves": [{"from": "rent-b", "into": "rent-a"}]
        });
        let llm = MockLlm::scripted(vec![json!([{"type": "text", "text": out.to_string()}])]);
        let traces = redistill(&llm, &mut g, "life", 2_000).unwrap();

        let child = &g.distillants["life/island/animals"];
        assert_eq!(child.parents, vec!["life"], "new child hangs under the distilled distillant");
        assert_eq!(child.tree, Tree::Registry, "child inherits the tree");
        assert_eq!(g.leaves["island-goats"].parents, vec!["life/island/animals"]);
        assert_eq!(g.leaves["island-pets"].parents, vec!["life/island/animals"]);
        assert!(!g.leaves.contains_key("rent-b"), "duplicate absorbed");
        assert_eq!(g.leaves["rent-a"].kind.belief().unwrap().support, 2);
        assert!(g.distillants["life"].line.contains("their own distillant"));
        assert!(traces.iter().any(|t| t.contains("✚ distillant life/island/animals")));
        assert!(traces.iter().any(|t| t.contains("moved [island-goats]")));
        assert_eq!(g.distillants["life"].consolidated_at, 2_000);
        assert_eq!(
            g.distillants["life/island/animals"].consolidated_at, 2_000,
            "fresh split children are stamped too, so the trigger does not re-fire on them"
        );
    }

    #[test]
    fn redistill_replaces_routing_instead_of_accreting() {
        let mut g = Graph::seed();
        {
            let m = g.distillants.get_mut("money").unwrap();
            m.routing = vec!["no. 001".into(), "no. 002".into(), "metadata".into(), "rent".into()];
        }
        let raw = r#"{"line": "Rent dominates.", "routing": ["rent", "landlord"], "merge_into": ""}"#;
        let traces = apply_redistilled(&mut g, "money", raw, 2_000).unwrap();
        assert_eq!(g.distillants["money"].routing, vec!["rent", "landlord"], "{traces:?}");
        assert!(traces.iter().any(|t| t.contains("✂ routing money − no. 001, no. 002, metadata")), "{traces:?}");
        assert!(render_redistill_prompt(&g, "money").unwrap().contains("\"merge_into\""));
    }

    #[test]
    fn bare_stub_is_due_by_itself_and_the_body_says_so() {
        let mut g = written_seed();
        harvest::apply_ops(
            &mut g,
            vec![Op::State {
                id: "x-report-run".into(),
                distillants: vec!["work/x-engagement-report".into()],
                text: "Ran the X report for Mason.".into(),
                relation: crate::harvest::Relation::Novel,
                target: "".into(),
                importance: 0.5,
                aliases: vec![],
                occurred_at: None,
            }],
            1_000,
        );
        assert_eq!(pressure(&g, "work/x-engagement-report"), BARE_PRESSURE);
        assert_eq!(due(&g).as_deref(), Some("work/x-engagement-report"), "a stub with no line is due on its own");
        let body = redistill_body(&g, "work/x-engagement-report").unwrap();
        assert!(body.contains("Current line: (none — created bare"), "{body}");
        assert!(stats(&g).contains("bare distillants (no line yet"), "{}", stats(&g));

        let raw = r#"{"line": "Runs the X engagement report for friends.", "routing": ["x report"], "merge_into": ""}"#;
        apply_redistilled(&mut g, "work/x-engagement-report", raw, 2_000).unwrap();
        assert_eq!(pressure(&g, "work/x-engagement-report"), 0, "a written line ends bare pressure");
        // The stub is settled; its parent holds nothing but this child, so
        // the line just written is the whole of what `work` summarizes.
        assert_eq!(due(&g).as_deref(), Some("work"), "the freshly written child is drift on the parent above it");
    }

    #[test]
    fn redistill_lists_namesakes_and_merge_into_absorbs_the_stub() {
        let mut g = Graph::seed();
        harvest::apply_ops(
            &mut g,
            vec![
                Op::Distillant {
                    id: "life/hacks".into(),
                    tree: Tree::Registry,
                    label: "Hacks".into(),
                    line: "The hack drawer.".into(),
                    routing: vec![],
                    parents: vec!["life".into()],
                },
                Op::Distillant {
                    id: "life/hacks/x-engagement-report".into(),
                    tree: Tree::Registry,
                    label: "X Engagement Report".into(),
                    line: "A hack that scrapes X and emails a dashboard.".into(),
                    routing: vec!["engagement report".into()],
                    parents: vec!["life/hacks".into()],
                },
                Op::State {
                    id: "x-report-run".into(),
                    distillants: vec!["work/x-engagement-report".into()],
                    text: "Ran the X report for Mason.".into(),
                    relation: crate::harvest::Relation::Novel,
                    target: "".into(),
                    importance: 0.5,
                    aliases: vec![],
                    occurred_at: None,
                },
            ],
            1_000,
        );
        let body = redistill_body(&g, "work/x-engagement-report").unwrap();
        assert!(
            body.contains("Same-named distillants elsewhere (merge_into one of these if this is the same thing):\n- life/hacks/x-engagement-report — A hack that scrapes X"),
            "{body}"
        );
        assert!(!redistill_body(&g, "life/hacks").unwrap().contains("Same-named"), "no namesake, no section");

        let raw = r#"{"line": "ignored", "routing": ["ignored"], "merge_into": "life/hacks/x-engagement-report"}"#;
        let traces = apply_redistilled(&mut g, "work/x-engagement-report", raw, 2_000).unwrap();
        assert!(!g.distillants.contains_key("work/x-engagement-report"), "{traces:?}");
        assert_eq!(g.leaves["x-report-run"].parents, vec!["life/hacks/x-engagement-report"]);
        let kept = &g.distillants["life/hacks/x-engagement-report"];
        assert_eq!(kept.line, "A hack that scrapes X and emails a dashboard.", "the absorbed stub's ignored line never lands");
        assert!(!kept.routing.contains(&"ignored".to_string()));
        assert!(traces.iter().any(|t| t.contains("⊕ merged distillant work/x-engagement-report into life/hacks/x-engagement-report")), "{traces:?}");

        // An invalid target (here: the other tree) falls back to a plain in-place redistill.
        let raw = r#"{"line": "Hacks, distilled.", "routing": ["hacks"], "merge_into": "temperament"}"#;
        let traces = apply_redistilled(&mut g, "life/hacks", raw, 3_000).unwrap();
        assert!(g.distillants.contains_key("life/hacks"));
        assert_eq!(g.distillants["life/hacks"].line, "Hacks, distilled.");
        assert!(traces.iter().any(|t| t.contains("merge_into rejected")), "{traces:?}");
        assert!(traces.iter().any(|t| t.contains("invalid")), "the merge names why: {traces:?}");
    }

    /// A seed whose crowns already carry a judged line — the shape every
    /// graph takes once its first passes have run. Pressure and drift are
    /// measured on top of a written crown here; a fresh seed is bare, and
    /// bare is its own pressure.
    fn written_seed() -> Graph {
        let mut g = Graph::seed();
        for m in g.distillants.values_mut() {
            m.line = format!("{} so far.", m.label);
            m.line_changed_at = 1;
            m.consolidated_at = 1;
        }
        g
    }

    fn bare_child(g: &mut Graph, id: &str, parent: &str) {
        let label = id.rsplit('/').next().unwrap_or(id).replace('-', " ");
        let m = crate::model::Distillant::bare(id, Tree::Registry, &label, vec![parent.to_string()], Vec::new());
        g.distillants.insert(id.to_string(), m);
    }

    fn leaf_under(g: &mut Graph, id: &str, parent: &str, at: u64) {
        let l = Leaf::state(id.into(), format!("fact {id}"), vec![parent.into()], 0.5, at);
        g.leaves.insert(l.id.clone(), l);
    }

    fn belief_of<'g>(g: &'g mut Graph, id: &str) -> &'g mut crate::model::Belief {
        g.leaves.get_mut(id).unwrap().kind.belief_mut().expect("a weighed leaf")
    }

    #[test]
    fn a_seeded_axis_comes_due_on_its_first_material() {
        let mut g = Graph::seed();
        assert_eq!(due(&g), None, "a fresh seed has nothing to judge");
        let l = Leaf::disposition("checks-balance".into(), "checks the balance before any spend".into(), vec!["money-style".into()], 0.5, 1_000);
        g.leaves.insert(l.id.clone(), l);
        assert_eq!(pressure(&g, "money-style"), BARE_PRESSURE, "an unjudged axis is bare pressure");
        assert_eq!(due(&g).as_deref(), Some("money-style"), "the axis is due the moment a leaf lands");
        let raw = r#"{"line": "Checks the balance before every spend.", "routing": [], "merge_into": ""}"#;
        apply_redistilled(&mut g, "money-style", raw, 2_000).unwrap();
        assert!(!g.distillants["money-style"].is_bare());
        assert_eq!(pressure(&g, "money-style"), 0);
    }

    #[test]
    fn one_moved_axis_is_enough_to_refresh_a_written_portrait() {
        let mut g = Graph::seed();
        for id in ["communication", "money-style", "temperament", PROFILE_APEX] {
            let m = g.distillants.get_mut(id).unwrap();
            m.line = format!("{id} line.");
            m.line_changed_at = 1_000;
            m.consolidated_at = 1_000;
        }
        assert_eq!(drift(&g, PROFILE_APEX), 0, "a fresh portrait holds");
        g.distillants.get_mut("money-style").unwrap().line_changed_at = 2_000;
        assert_eq!(drift(&g, PROFILE_APEX), DRIFT_CHILD_LINE_SOLE_EVIDENCE);
        assert_eq!(due(&g).as_deref(), Some(PROFILE_APEX), "the portrait is stale the moment an axis moves");

        // A distillant that carries leaves of its own weighs one changed
        // child as the fraction of the evidence it is.
        let l = Leaf::state("own".into(), "a fact of its own".into(), vec!["money".into()], 0.5, 1_000);
        g.leaves.insert(l.id.clone(), l);
        bare_child(&mut g, "money/rent", "money");
        g.distillants.get_mut("money").unwrap().consolidated_at = 1_000;
        g.distillants.get_mut("money/rent").unwrap().line_changed_at = 2_000;
        assert_eq!(drift(&g, "money"), DRIFT_CHILD_LINE);
    }

    #[test]
    fn the_apex_is_due_once_an_axis_line_exists_and_is_distilled_from_the_axes() {
        let mut g = Graph::seed();
        let l = Leaf::disposition("checks-balance".into(), "checks the balance before any spend".into(), vec!["money-style".into()], 0.5, 1_000);
        g.leaves.insert(l.id.clone(), l);
        let raw = r#"{"line": "Checks the balance before every spend.", "routing": [], "merge_into": ""}"#;
        apply_redistilled(&mut g, "money-style", raw, 2_000).unwrap();
        assert_eq!(due(&g).as_deref(), Some(PROFILE_APEX), "a written axis is drift on the bare apex");
        let body = redistill_body(&g, PROFILE_APEX).unwrap();
        assert!(body.contains("This is the profile apex"), "{body}");
        assert!(body.contains("- money-style — Checks the balance before every spend."), "the axis lines are the evidence: {body}");
        let raw = r#"{"line": "Careful with every euro of their own.", "routing": [], "merge_into": ""}"#;
        apply_redistilled(&mut g, PROFILE_APEX, raw, 3_000).unwrap();
        assert_eq!(due(&g), None, "the portrait absorbed the axes");
        assert!(crate::projection::render_profile(&g).starts_with("- character — Careful with every euro of their own.\n  - communication"), "{}", crate::projection::render_profile(&g));
    }

    #[test]
    fn pressure_counts_fat_and_residual() {
        let mut g = written_seed();
        assert_eq!(pressure(&g, "money"), 0);
        assert_eq!(pressure(&g, "ghost"), 0);
        for i in 0..10 {
            leaf_under(&mut g, &format!("f{i}"), "money", 1_000);
        }
        assert_eq!(pressure(&g, "money"), 2, "two past the fat threshold");
        g.distillants.get_mut("money").unwrap().misc_count = 4;
        assert_eq!(pressure(&g, "money"), 6, "residual adds to fat");
    }

    #[test]
    fn due_picks_worst_offender_and_quiets_after_a_pass() {
        let mut g = written_seed();
        for i in 0..10 {
            leaf_under(&mut g, &format!("m{i}"), "money", 1_000);
        }
        assert_eq!(due(&g), None, "pressure 2 is under the trigger");
        for i in 0..12 {
            leaf_under(&mut g, &format!("l{i}"), "life", 1_000);
        }
        for i in 0..3 {
            leaf_under(&mut g, &format!("p{i}"), "people", 1_000);
        }
        g.distillants.get_mut("people").unwrap().misc_count = 3;
        assert_eq!(due(&g), Some("life".into()), "worst offender first (4 beats 3)");
        g.distillants.get_mut("life").unwrap().consolidated_at = 2_000;
        assert_eq!(due(&g), Some("people".into()), "a consolidated distillant goes quiet");
        g.distillants.get_mut("people").unwrap().consolidated_at = 2_000;
        assert_eq!(due(&g), None, "nothing over the trigger has new material");
        leaf_under(&mut g, "l-new", "life", 3_000);
        assert_eq!(due(&g), Some("life".into()), "new material re-arms the trigger");
    }

    /// A child distillant under `parent` with an empty line, fresh at `at`.
    fn mid_under(g: &mut Graph, id: &str, parent: &str, at: u64) {
        g.distillants.insert(
            id.into(),
            crate::model::Distillant {
                id: id.into(),
                tree: crate::model::Tree::Registry,
                label: id.into(),
                line: String::new(),
                routing: vec![],
                parents: vec![parent.into()],
                misc_count: 0,
                consolidated_at: at,
                line_changed_at: 0,
                forgotten_at: 0,
            },
        );
    }

    #[test]
    fn drift_counts_leaves_that_moved_against_their_text() {
        let mut g = written_seed();
        g.distillants.get_mut("money").unwrap().consolidated_at = 1_500;
        leaf_under(&mut g, "rent", "money", 1_000);
        leaf_under(&mut g, "lease", "money", 1_000);
        assert_eq!(drift(&g, "money"), 0, "quiet leaves are not drift");

        crate::belief::contradict(belief_of(&mut g, "rent"), 2_000);
        assert_eq!(drift(&g, "money"), DRIFT_AGAINST, "a contradiction since the pass counts");
        {
            let lease = g.leaves.get_mut("lease").unwrap();
            let crate::model::LeafKind::State(state) = &mut lease.kind else { panic!() };
            let old = std::mem::replace(&mut lease.text, "lease ended".into());
            crate::belief::supersede(state, old, 2_100, 2_100);
        }
        assert_eq!(drift(&g, "money"), 2 * DRIFT_AGAINST, "a supersession counts too");
        assert_eq!(
            pressure(&g, "money"),
            2 * DRIFT_AGAINST,
            "drift is pressure: semantic rot competes in the same queue"
        );

        g.distillants.get_mut("money").unwrap().consolidated_at = 3_000;
        assert_eq!(drift(&g, "money"), 0, "a pass clears drift by construction");
    }

    #[test]
    fn a_forget_under_a_distillant_is_drift_until_the_next_pass() {
        let mut g = written_seed();
        leaf_under(&mut g, "katsu", "life", 1_000);
        leaf_under(&mut g, "walks", "life", 1_000);
        g.distillants.get_mut("life").unwrap().consolidated_at = 1_500;
        assert_eq!(drift(&g, "life"), 0);
        g.forget("katsu", 2_000);
        assert_eq!(drift(&g, "life"), DRIFT_AGAINST, "the line was written over a leaf that is gone");
        assert_eq!(due(&g), None, "one forget is under the trigger on its own");
        let raw = r#"{"line": "Walks, mostly.", "routing": [], "merge_into": ""}"#;
        apply_redistilled(&mut g, "life", raw, 3_000).unwrap();
        assert_eq!(drift(&g, "life"), 0, "a pass clears it by construction");
    }

    #[test]
    fn events_before_the_pass_are_not_drift() {
        let mut g = Graph::seed();
        leaf_under(&mut g, "rent", "money", 1_000);
        crate::belief::contradict(belief_of(&mut g, "rent"), 1_200);
        g.distillants.get_mut("money").unwrap().consolidated_at = 1_500;
        assert_eq!(drift(&g, "money"), 0, "the line was written knowing this");
    }

    #[test]
    fn child_line_change_drifts_the_parent_only_when_text_differs() {
        let mut g = Graph::seed();
        g.distillants.get_mut("life").unwrap().consolidated_at = 1_500;
        mid_under(&mut g, "life/island", "life", 1_000);

        let same = r#"{"line": "", "routing": []}"#;
        apply_redistilled(&mut g, "life/island", same, 2_000).unwrap();
        assert_eq!(drift(&g, "life"), 0, "an identical rewrite stops the cascade");

        let changed = r#"{"line": "Marooned; the island is home now.", "routing": []}"#;
        apply_redistilled(&mut g, "life/island", changed, 2_500).unwrap();
        assert_eq!(g.distillants["life/island"].line_changed_at, 2_500);
        // `life` holds no leaves here, so its one child is its whole evidence.
        assert_eq!(drift(&g, "life"), DRIFT_CHILD_LINE_SOLE_EVIDENCE, "a changed child line drifts the parent");
        assert_eq!(drift(&g, "life/island"), 0, "the freshly passed child itself is clean");
    }

    #[test]
    fn due_fires_on_pure_drift_and_descends_to_a_due_child() {
        let mut g = Graph::seed();
        // Leaves predate the pass; only their beliefs moved after it. The
        // old new-material filter (updated_at) would have kept this quiet.
        g.distillants.get_mut("life").unwrap().consolidated_at = 1_500;
        for id in ["a", "b"] {
            leaf_under(&mut g, id, "life", 1_000);
            belief_of(&mut g, id).last_against_at = 2_000;
        }
        assert_eq!(due(&g), Some("life".into()), "pure drift re-arms the trigger");

        // A due child outranks its due parent: the parent's rewrite would
        // consume the child's stale line. Freshness flows leaves-up.
        mid_under(&mut g, "life/island", "life", 1_500);
        for id in ["c", "d"] {
            leaf_under(&mut g, id, "life/island", 1_000);
            belief_of(&mut g, id).last_against_at = 2_000;
        }
        assert_eq!(due(&g), Some("life/island".into()), "children before parents");
    }

    #[test]
    fn redistill_prompt_demands_grounding() {
        let g = Graph::seed();
        let prompt = render_redistill_prompt(&g, "money").unwrap();
        assert!(
            prompt.contains("a style reference, not a source"),
            "the anchoring guard must stay in the contract"
        );
    }

    #[test]
    fn auto_step_consolidates_the_worst_offender_once() {
        let mut g = written_seed();
        for i in 0..12 {
            leaf_under(&mut g, &format!("l{i}"), "life", 1_000);
        }
        let llm = MockLlm::scripted(vec![json!([{
            "type": "text",
            "text": "{\"line\": \"Island life, mostly.\", \"routing\": []}"
        }])]);
        let traces = auto_step(&llm, &mut g, 2_000).unwrap();
        assert!(
            traces.iter().any(|t| t.contains("⚙ consolidating life (pressure 4)")),
            "{traces:?}"
        );
        assert_eq!(g.distillants["life"].consolidated_at, 2_000);
        // Still fat, but the model declined to split and nothing new landed:
        // the next step is a no-op instead of thrash.
        let traces = auto_step(&llm, &mut g, 3_000).unwrap();
        assert!(traces.is_empty(), "{traces:?}");
    }

    #[test]
    fn stream_fold_distills_oldest_batch_with_model_text() {
        let mut g = written_seed();
        let first = g.push_episode("sold Xury".into(), vec![], 1, None);
        let mut l = Leaf::state("xury".into(), "Xury sold.".into(), vec!["people".into()], 0.5, 1);
        l.evidence.push(first.clone());
        g.leaves.insert(l.id.clone(), l);
        for i in 0..STREAM_CAP {
            g.push_episode(format!("e{i}"), vec![], i as u64 + 2, None);
        }
        // Over the soft cap, under the hard one: nothing folded mechanically.
        assert_eq!(g.episodes.len(), STREAM_CAP + 1);
        assert!(matches!(stream_due(&g), Some(StreamStep::Fold(n)) if n == DIGEST_WINDOW));

        let llm = MockLlm::scripted(vec![json!([{
            "type": "text",
            "text": "{\"text\": \"Early days: sold Xury to the captain — a standing regret.\"}"
        }])]);
        let traces = auto_step(&llm, &mut g, 9_000).unwrap();
        assert!(traces.iter().any(|t| t.contains("folded 100 episodes")), "{traces:?}");
        assert_eq!(g.episodes.len(), STREAM_CAP + 1 - COMPRESS_BATCH + 1);
        assert!(g.episodes[0].text.starts_with("Early days"), "digest carries the model text");
        assert_eq!(g.leaves["xury"].evidence, vec!["ep-digest-1"], "evidence remapped");
        assert!(stream_due(&g).is_none(), "under cap and no mechanical marker: quiet");
    }

    #[test]
    fn fold_honors_the_model_cut_and_renders_a_numbered_window() {
        let mut g = Graph::seed();
        for i in 0..=STREAM_CAP {
            g.push_episode(format!("e{i}"), vec![], i as u64 + 1, None);
        }
        let step = stream_due(&g).unwrap();
        let body = digest_body(&g, &step, 9_000);
        assert!(body.contains("1. ("), "episodes are numbered: {body}");
        assert!(
            body.contains(&format!("Cut bounds: take between {DIGEST_MIN_CUT} and {DIGEST_WINDOW}.")),
            "{body}"
        );

        let llm = MockLlm::scripted(vec![json!([{
            "type": "text",
            "text": "{\"text\": \"The first voyage, whole.\", \"take\": 130}"
        }])]);
        let traces = auto_step(&llm, &mut g, 9_000).unwrap();
        assert!(traces.iter().any(|t| t.contains("folded 130 episodes")), "{traces:?}");
        assert_eq!(g.episodes.len(), STREAM_CAP + 1 - 130 + 1);
        assert!(g.episodes[0].text.starts_with("The first voyage"));
    }

    #[test]
    fn fold_clamps_the_cut_to_the_window_bounds() {
        let mut g = Graph::seed();
        for i in 0..=STREAM_CAP {
            g.push_episode(format!("e{i}"), vec![], i as u64 + 1, None);
        }
        let traces = apply_digest(&mut g, "Tiny cut asked.".into(), Some(3), 9_000).unwrap();
        assert!(
            traces.iter().any(|t| t.contains(&format!("folded {DIGEST_MIN_CUT} episodes"))),
            "an under-minimum cut clamps up: {traces:?}"
        );

        let mut g = Graph::seed();
        for i in 0..=STREAM_CAP {
            g.push_episode(format!("e{i}"), vec![], i as u64 + 1, None);
        }
        let traces = apply_digest(&mut g, "Huge cut asked.".into(), Some(9_999), 9_000).unwrap();
        assert!(
            traces.iter().any(|t| t.contains(&format!("folded {DIGEST_WINDOW} episodes"))),
            "an over-window cut clamps down: {traces:?}"
        );
    }

    #[test]
    fn mechanical_digest_polish_rewrites_in_place_and_disarms() {
        let mut g = Graph::seed();
        g.push_episode(
            format!("{MECH_DIGEST_PREFIX}3 earlier episodes] a; b; c"),
            vec!["life".into()],
            5,
            None,
        );
        g.push_episode("later event".into(), vec![], 6, None);
        let Some(StreamStep::Polish(id)) = stream_due(&g) else { panic!("polish should be due") };
        let llm = MockLlm::scripted(vec![json!([{
            "type": "text",
            "text": "{\"text\": \"A readable period summary.\"}"
        }])]);
        let traces = auto_step(&llm, &mut g, 9_000).unwrap();
        assert!(traces.iter().any(|t| t.contains("polished")), "{traces:?}");
        let e = g.episodes.iter().find(|e| e.id == id).unwrap();
        assert_eq!(e.text, "A readable period summary.");
        assert_eq!(e.tags, vec!["life"], "polish keeps id and tags");
        assert_eq!(g.episodes.len(), 2, "polish replaces text, not episodes");
        assert!(stream_due(&g).is_none(), "marker gone, trigger disarmed");
    }

    #[test]
    fn distillant_pressure_takes_precedence_over_stream() {
        let mut g = Graph::seed();
        for i in 0..12 {
            leaf_under(&mut g, &format!("l{i}"), "life", 1_000);
        }
        g.push_episode(format!("{MECH_DIGEST_PREFIX}2 earlier episodes] x; y"), vec![], 5, None);
        let llm = MockLlm::scripted(vec![json!([{
            "type": "text",
            "text": "{\"line\": \"Island life.\", \"routing\": []}"
        }])]);
        let traces = auto_step(&llm, &mut g, 2_000).unwrap();
        assert!(traces.iter().any(|t| t.contains("consolidating life")), "{traces:?}");
        assert!(
            g.episodes[0].text.starts_with(MECH_DIGEST_PREFIX),
            "one step per turn: stream untouched while a distillant was due"
        );
        // Next turn nothing new for distillants; the stream step runs, here on
        // the unscripted mock's digest auto-behavior.
        let traces = auto_step(&llm, &mut g, 3_000).unwrap();
        assert!(traces.iter().any(|t| t.contains("polished")), "{traces:?}");
        assert!(g.episodes[0].text.starts_with("[mock digest]"), "{}", g.episodes[0].text);
    }

    #[test]
    fn echoed_mechanical_marker_is_disarmed() {
        let mut g = Graph::seed();
        g.push_episode(format!("{MECH_DIGEST_PREFIX}2 earlier episodes] x; y"), vec![], 5, None);
        let traces =
            apply_digest(&mut g, format!("{MECH_DIGEST_PREFIX}still mechanical"), None, 9_000)
                .unwrap();
        assert!(traces.iter().any(|t| t.contains("polished")), "{traces:?}");
        assert!(g.episodes[0].text.starts_with("(distilled) "));
        assert!(stream_due(&g).is_none(), "echoed marker disarmed, no thrash");
    }
}
