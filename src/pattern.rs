//! The pattern pass: the one consolidation step that reads across the whole
//! stream — counts the action ledger, clusters what repeats, and has the model
//! rule on one cluster at a time. Run from maintenance beside redistill and digest.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use anyhow::Result;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::harvest::{self, Op};
use crate::llm::{ChatRequest, Llm};
use crate::model::{age_str, slugify, Distillant, Graph, Id, Ruling, Tally, Tree, Verdict, HABITS};

const DAY: u64 = 86_400;

pub const MIN_MARKS: u32 = 5;
pub const MIN_DISTILLANTS: u32 = 3;
pub const MIN_SPAN_SECS: u64 = 14 * DAY;
pub const REASK_MARKS: u32 = 5;
pub const REFRESH_MARKS: u32 = 3;
pub const REFRESH_STALE_SECS: u64 = 7 * DAY;
pub const FADE_GAPS: u64 = 3;
pub const FADE_FLOOR_SECS: u64 = 14 * DAY;
pub const FOLD_GAPS: u64 = 6;
pub const FOLD_FLOOR_SECS: u64 = 60 * DAY;
pub const TAG_BATCH: usize = 100;
pub const TAG_VOCABULARY_LINES: usize = 80;
pub const PROMPT_EVENTS: usize = 12;
pub const PROMPT_LEAVES: usize = 20;
pub const PROMPT_EPISODE_CHARS: usize = 300;
pub const ROUTING_MAX: usize = 12;
const ROUTING_STOPWORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "by", "for", "with", "from", "it", "its", "is", "be",
    "do", "does", "did", "me", "my", "i", "you", "your", "we", "us", "he", "she", "they", "them", "this", "that", "these",
    "those", "up", "out", "again", "also", "just", "now", "then", "some", "any", "all", "one", "run", "make", "get",
    "go", "use", "try", "see", "want", "need", "like", "new", "more", "thing", "things", "something", "stuff",
];

pub fn routable(term: &str) -> bool {
    let term = term.trim().to_lowercase();
    if term.len() < 2 {
        return false;
    }
    let single = !term.contains(' ');
    !(single && ROUTING_STOPWORDS.contains(&term.as_str()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Mint,
    Refresh,
    Fade,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PatternStep {
    Tag(Vec<Id>),
    Rule { action: String, phase: Phase },
    Fold(Id),
}

impl PatternStep {
    pub fn describe(&self) -> String {
        match self {
            PatternStep::Tag(ids) => format!("pattern pass due: tag {} untagged episodes with their actions", ids.len()),
            PatternStep::Rule { action, phase } => {
                let verb = match phase {
                    Phase::Mint => "rule on",
                    Phase::Refresh => "refresh the habit for",
                    Phase::Fade => "fade the habit for",
                };
                format!("pattern pass due: {verb} the repeated action \"{action}\"")
            }
            PatternStep::Fold(id) => format!("pattern pass due: fold the faded habit {id} into the stream"),
        }
    }

    pub fn needs_model(&self) -> bool {
        !matches!(self, PatternStep::Fold(_))
    }
}

pub fn is_habit_id(id: &str) -> bool {
    id == HABITS || id.starts_with("habits/")
}

pub fn marks_of(ledger: &std::collections::BTreeMap<String, Vec<u64>>, actions: &[String]) -> Vec<u64> {
    let mut marks: Vec<u64> = actions.iter().filter_map(|a| ledger.get(a)).flatten().copied().collect();
    marks.sort_unstable();
    marks
}

fn carries(e: &crate::model::Episode, actions: &[String]) -> bool {
    e.actions.iter().flatten().any(|a| actions.contains(a))
}

pub fn tally_of(graph: &Graph, actions: &[String], marks: &[u64], now: u64) -> Tally {
    let since = |days: u64| marks.iter().filter(|&&t| t.saturating_add(days * DAY) >= now).count() as u32;
    let mut gaps: Vec<u64> = marks.windows(2).map(|w| w[1].saturating_sub(w[0])).collect();
    gaps.sort_unstable();
    let median_gap_secs = gaps.get(gaps.len() / 2).copied().unwrap_or(0);
    let distillants: BTreeSet<&str> = graph
        .episodes
        .iter()
        .filter(|e| carries(e, actions))
        .flat_map(|e| e.tags.iter())
        .filter(|t| graph.distillants.contains_key(t.as_str()) && !is_habit_id(t))
        .map(String::as_str)
        .collect();
    Tally {
        actions: actions.to_vec(),
        n_all: marks.len() as u32,
        n_30d: since(30),
        n_7d: since(7),
        first_seen: marks.first().copied().unwrap_or(0),
        last_seen: marks.last().copied().unwrap_or(0),
        distillants: distillants.len() as u32,
        median_gap_secs,
        computed_at: now,
        faded: false,
    }
}

pub fn fade_after(t: &Tally) -> u64 {
    (FADE_GAPS * t.median_gap_secs).max(FADE_FLOOR_SECS)
}

pub fn fold_after(t: &Tally) -> u64 {
    (FOLD_GAPS * t.median_gap_secs).max(FOLD_FLOOR_SECS)
}

fn idle(t: &Tally, now: u64) -> u64 {
    now.saturating_sub(t.last_seen)
}

pub fn mint_worthy(t: &Tally) -> bool {
    t.n_all >= MIN_MARKS
        && t.distillants >= MIN_DISTILLANTS
        && t.last_seen.saturating_sub(t.first_seen) >= MIN_SPAN_SECS
}

fn verdict_allows(graph: &Graph, action: &str, t: &Tally) -> bool {
    match graph.patterns.verdicts.get(action) {
        None => true,
        Some(v) => t.n_all >= v.marks + REASK_MARKS,
    }
}

fn refresh_due(stored: &Tally, fresh: &Tally, now: u64) -> Option<Phase> {
    let grew = fresh.n_all > stored.n_all;
    let stale = now.saturating_sub(stored.computed_at) > REFRESH_STALE_SECS;
    if !stored.faded && idle(fresh, now) > fade_after(fresh) {
        return Some(Phase::Fade);
    }
    if fresh.n_all >= stored.n_all + REFRESH_MARKS || (grew && (stale || stored.faded)) {
        return Some(Phase::Refresh);
    }
    None
}

pub fn untagged(graph: &Graph) -> Vec<Id> {
    graph.episodes.iter().filter(|e| e.awaits_actions()).map(|e| e.id.clone()).take(TAG_BATCH).collect()
}

pub fn due(graph: &Graph, now: u64) -> Option<PatternStep> {
    let untagged = untagged(graph);
    if !untagged.is_empty() {
        return Some(PatternStep::Tag(untagged));
    }
    let ledger = graph.ledger();
    let mut habits: Vec<&Distillant> = graph.distillants.values().filter(|m| m.tally.is_some()).collect();
    habits.sort_by(|a, b| a.id.cmp(&b.id));
    let mut candidates: Vec<(u32, String, Phase)> = Vec::new();
    for habit in habits {
        let stored = habit.tally.as_ref().expect("filtered on tally");
        let marks = marks_of(&ledger, &stored.actions);
        if marks.is_empty() {
            return Some(PatternStep::Fold(habit.id.clone()));
        }
        let fresh = tally_of(graph, &stored.actions, &marks, now);
        if stored.faded && idle(&fresh, now) > fold_after(&fresh) {
            return Some(PatternStep::Fold(habit.id.clone()));
        }
        if let Some(phase) = refresh_due(stored, &fresh, now) {
            let action = stored.actions.first().cloned().unwrap_or_default();
            candidates.push((fresh.n_all, action, phase));
        }
    }
    for (action, marks) in &ledger {
        if graph.habit_of(action).is_some() {
            continue;
        }
        let fresh = tally_of(graph, std::slice::from_ref(action), marks, now);
        if mint_worthy(&fresh) && verdict_allows(graph, action, &fresh) {
            candidates.push((fresh.n_all, action.clone(), Phase::Mint));
        }
    }
    candidates.sort_by(|(n, a, _), (m, b, _)| m.cmp(n).then_with(|| a.cmp(b)));
    candidates.into_iter().next().map(|(_, action, phase)| PatternStep::Rule { action, phase })
}

const RULE_SYSTEM: &str = "You judge one repeated action in a personal memory graph. The runtime \
counted it mechanically across the whole stream: how often, over what span, under how many \
distinct projects, when first and last seen. You decide what the count means and, when it is a \
habit, write the habits/* distillant that stands for it.\n\
\n\
A HABIT is a way of acting that repeats across unrelated undertakings: the person reaches for \
it whatever the project, and a future request could ask for it. A PROJECT is one undertaking \
that happens to have many events: the evidence all belongs to one thing. A TRAIT is the person \
acting, but as a reaction or a manner (rating, refusing, swearing, thinking aloud), not something \
a request would ask for: the profile holds it, no habit is minted. NOISE is a tag that is not \
the person acting at all. An action so broad it underlies nearly every event (asking, deciding, \
paying in general) is a trait or noise, never a habit: a habit is specific enough that its \
routing catches requests for it and little else.\n\
\n\
The line is ONE short sentence about the person (under 160 characters): what they do and what \
for, in the tense the tally supports. Never restate the numbers — the count renders beside the \
line automatically — and never list instances. A count still growing (marks in the last 7 or 30 \
days) is present tense: 'publishes a hosted page for almost anything he makes'. A count that \
stopped (no marks for several times its usual gap) is past tense: 'ran pranks on friends for a \
month; none since'. The tally is the evidence: never claim a rhythm the numbers do not show. \
Plain prose about the person, never machinery words (tally, distillant, routing).\n\
\n\
Routing is the recall vocabulary: 5 to 12 terms, the nouns and verbs a request for THIS \
behaviour would carry and a request for anything else would not ('website', 'site', 'page', \
'publish', 'put it online'). Never a URL, a project name, the tag's own spelling, or a word a \
message about anything could contain (for, on, run, again, try, see if, make): every stray term \
routes unrelated messages here. Matching is exact-token, no stemming: include the forms a \
message would actually contain (page AND pages).\n\
\n\
adopt_leaves names the evidence leaves listed below that ARE this habit in action (a page \
published, a run scheduled); the habit becomes their second parent and the project keeps \
them. Leave out leaves that are about the project rather than the way of acting.\n\
\n\
When an existing habit distillant is shown, keep its id: you are refreshing its line and \
routing from the new count, not minting a twin. When the existing habits listed already \
cover this action under another spelling, answer habit with THAT id.";

fn rule_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["habit", "project", "trait", "noise"]},
            "id": {"type": "string", "description": "habits/<kebab> — the existing id when one is shown; empty when the verdict is not habit"},
            "label": {"type": "string", "description": "short label; empty when not habit"},
            "line": {"type": "string", "description": "one sentence about the person in the tense the tally supports; empty when not habit"},
            "routing": {"type": "array", "items": {"type": "string"}, "description": "the recall vocabulary for this habit, 5 to 12 terms (replaces the current set): the nouns and verbs a request for this behaviour would carry and a request for anything else would not"},
            "adopt_leaves": {"type": "array", "items": {"type": "string"}, "description": "ids of the listed evidence leaves that are this habit in action"},
            "reason": {"type": "string", "description": "one line: why this verdict"}
        },
        "required": ["verdict", "id", "label", "line", "routing", "adopt_leaves", "reason"],
        "additionalProperties": false
    })
}

#[derive(Deserialize)]
struct Ruled {
    verdict: String,
    #[serde(default)]
    id: String,
    #[serde(default)]
    label: String,
    #[serde(default)]
    line: String,
    #[serde(default)]
    routing: Vec<String>,
    #[serde(default)]
    adopt_leaves: Vec<String>,
    #[serde(default)]
    reason: String,
}

const TAG_SYSTEM: &str = "You tag episodes of a personal memory stream with the ACTION they \
record: what the user did, as 0-2 short verb-shaped kebab-case tags ('published-page', \
'emailed-result', 'scheduled-run', 'paid-stranger', 'pranked-friend'). Reuse the vocabulary \
shown whenever a spelling fits, so one kind of act stays one count; coin a new tag only for a \
kind of act the vocabulary lacks. An act the assistant takes on the user's instruction or \
budget counts as the user acting; an act the assistant took unprompted does not. An episode \
that is not the user acting (news, someone else's doing, a state of the world, a period \
summary) gets an empty list. Answer every episode id listed, once.";

fn tag_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "episode": {"type": "string"},
                        "actions": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["episode", "actions"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["tags"],
        "additionalProperties": false
    })
}

#[derive(Deserialize)]
struct Tagged {
    tags: Vec<TagEntry>,
}

#[derive(Deserialize)]
struct TagEntry {
    episode: String,
    actions: Vec<String>,
}

pub fn tally_words(t: &Tally, now: u64) -> String {
    let mut words = format!(
        "{} times; first {}, last {}; {} in the last 30 days, {} in the last 7; across {} distillants",
        t.n_all,
        age_str(now, t.first_seen),
        age_str(now, t.last_seen),
        t.n_30d,
        t.n_7d,
        t.distillants
    );
    if t.median_gap_secs > 0 {
        let days = t.median_gap_secs / DAY;
        let _ = match days {
            0 => write!(words, "; typical gap under a day"),
            1 => write!(words, "; typical gap about a day"),
            d => write!(words, "; typical gap about {d} days"),
        };
    }
    words
}

fn tag_body(graph: &Graph, ids: &[Id], now: u64) -> String {
    let ledger = graph.ledger();
    let mut vocabulary: Vec<(&String, usize)> = ledger.iter().map(|(a, ats)| (a, ats.len())).collect();
    vocabulary.sort_by(|(a, x), (b, y)| y.cmp(x).then_with(|| a.cmp(b)));
    let mut body = String::from("# Action vocabulary so far (reuse these spellings; the count follows each)\n");
    if vocabulary.is_empty() {
        body.push_str("(none yet)\n");
    }
    for (action, n) in vocabulary.into_iter().take(TAG_VOCABULARY_LINES) {
        let _ = writeln!(body, "- {action} ×{n}");
    }
    body.push_str("\n# Episodes to tag\n");
    for id in ids {
        if let Some(e) = graph.episodes.iter().find(|e| &e.id == id) {
            let _ = writeln!(body, "- [{}] ({}) {}", e.id, age_str(now, e.event_at()), clipped(&e.text));
        }
    }
    body
}

fn clipped(text: &str) -> String {
    let mut out: String = text.chars().take(PROMPT_EPISODE_CHARS).collect();
    if out.len() < text.len() {
        out.push('…');
    }
    out
}

fn evidence_of(leaf: &crate::model::Leaf, events: &[&crate::model::Episode], habit_id: Option<&str>) -> bool {
    if leaf.is_rule() {
        return false;
    }
    let adopted_elsewhere = leaf.parents.iter().any(|p| is_habit_id(p) && Some(p.as_str()) != habit_id);
    if adopted_elsewhere {
        return false;
    }
    events.iter().any(|e| leaf.evidence.contains(&e.id) && e.tags.iter().any(|t| leaf.parents.contains(t)))
}

fn actions_of(graph: &Graph, action: &str) -> Vec<String> {
    match graph.habit_of(action) {
        Some(h) => h.tally.as_ref().map(|t| t.actions.clone()).unwrap_or_default(),
        None => vec![action.to_string()],
    }
}

fn rule_body(graph: &Graph, action: &str, phase: Phase, now: u64) -> String {
    let ledger = graph.ledger();
    let actions = actions_of(graph, action);
    let marks = marks_of(&ledger, &actions);
    let tally = tally_of(graph, &actions, &marks, now);
    let mut body = format!("# Repeated action: {}\nTally: {}\n", actions.join(" + "), tally_words(&tally, now));
    match phase {
        Phase::Mint => body.push_str("Status: no habit distillant stands for this yet — rule on it.\n"),
        Phase::Refresh => body.push_str("Status: the count grew since the line was written — refresh the line and routing from the tally.\n"),
        Phase::Fade => {
            let _ = writeln!(
                body,
                "Status: last seen {} — idle beyond several times its usual gap. If it is still a habit, write the line in PAST tense.",
                age_str(now, tally.last_seen)
            );
        }
    }
    match graph.habit_of(action) {
        Some(habit) => {
            let _ = writeln!(
                body,
                "Existing habit distillant (keep this id): {} (label: {})\nCurrent line: {}\nCurrent routing: {}",
                habit.id,
                habit.label,
                habit.headline(),
                habit.routing.join(", ")
            );
        }
        None => body.push_str("Existing habit distillant: (none)\n"),
    }
    let mut events: Vec<&crate::model::Episode> =
        graph.episodes.iter().rev().filter(|e| carries(e, &actions)).take(PROMPT_EVENTS).collect();
    events.reverse();
    let _ = writeln!(body, "\nEvents carrying this action ({} most recent, oldest first):", events.len());
    if events.is_empty() {
        body.push_str("(only folded counts remain; no live events)\n");
    }
    for e in &events {
        let _ = writeln!(body, "- ({}) [{}] {} [tags: {}]", age_str(now, e.event_at()), e.id, clipped(&e.text), e.tags.join(", "));
    }
    let habit_id = graph.habit_of(action).map(|h| h.id.clone());
    let mut leaves: Vec<&crate::model::Leaf> =
        graph.leaves.values().filter(|l| evidence_of(l, &events, habit_id.as_deref())).collect();
    leaves.sort_by(|a, b| b.updated_at.cmp(&a.updated_at).then_with(|| a.id.cmp(&b.id)));
    leaves.truncate(PROMPT_LEAVES);
    body.push_str("\nEvidence leaves those events back (adopt the ones that are this habit in action):\n");
    if leaves.is_empty() {
        body.push_str("(none)\n");
    }
    for l in leaves {
        let adopted = habit_id.as_ref().is_some_and(|h| l.parents.contains(h));
        let _ = writeln!(
            body,
            "- [{}] (under {}){} {}",
            l.id,
            l.parents.join("+"),
            if adopted { " [already adopted]" } else { "" },
            l.text
        );
    }
    let mut habits: Vec<&Distillant> =
        graph.distillants.values().filter(|m| m.tally.is_some() && Some(&m.id) != habit_id.as_ref()).collect();
    habits.sort_by(|a, b| a.id.cmp(&b.id));
    if !habits.is_empty() {
        body.push_str("\nOther habits already standing (answer with one of these ids if this action is the same habit; its line and routing are then yours to refresh):\n");
        for h in habits {
            let actions = h.tally.as_ref().map(|t| t.actions.join(" + ")).unwrap_or_default();
            let _ = writeln!(body, "- {} (actions: {}) — {} | routing: {}", h.id, actions, h.headline(), h.routing.join(", "));
        }
    }
    body
}

pub fn render_prompt(graph: &Graph, step: &PatternStep, now: u64) -> Option<String> {
    let (system, schema, body) = match step {
        PatternStep::Tag(ids) => (TAG_SYSTEM, tag_schema(), tag_body(graph, ids, now)),
        PatternStep::Rule { action, phase } => (RULE_SYSTEM, rule_schema(), rule_body(graph, action, *phase, now)),
        PatternStep::Fold(_) => return None,
    };
    Some(format!(
        "# System\n{system}\n\n# Output schema (reply with one JSON object matching it)\n{}\n\n# Input\n{body}",
        serde_json::to_string_pretty(&schema).expect("static schema serializes")
    ))
}

pub fn apply(graph: &mut Graph, step: &PatternStep, raw: &str, now: u64) -> Result<Vec<String>> {
    match step {
        PatternStep::Tag(ids) => apply_tags(graph, ids, raw),
        PatternStep::Rule { action, phase } => apply_ruling(graph, action, *phase, raw, now),
        PatternStep::Fold(id) => Ok(fold_habit(graph, id, now)),
    }
}

fn apply_tags(graph: &mut Graph, ids: &[Id], raw: &str) -> Result<Vec<String>> {
    let out: Tagged = serde_json::from_str(raw)?;
    let mut acted = 0usize;
    let mut tagged = 0usize;
    for id in ids {
        let Some(e) = graph.episodes.iter_mut().find(|e| &e.id == id) else { continue };
        let actions: Vec<String> = out
            .tags
            .iter()
            .filter(|t| &t.episode == id)
            .flat_map(|t| t.actions.iter())
            .map(|a| slugify(a))
            .filter(|a| !a.is_empty())
            .collect();
        acted += usize::from(!actions.is_empty());
        tagged += 1;
        e.actions = Some(actions);
    }
    Ok(vec![format!("◦ tagged {tagged} episodes with actions ({acted} record the user acting)")])
}

fn habit_id_of(returned: &str, existing: Option<&str>) -> Id {
    if let Some(id) = existing {
        return id.to_string();
    }
    let tail = returned.strip_prefix("habits/").unwrap_or(returned);
    let tail = slugify(tail).replace('/', "-");
    if tail.is_empty() {
        format!("{HABITS}/habit")
    } else {
        format!("{HABITS}/{tail}")
    }
}

fn apply_ruling(graph: &mut Graph, action: &str, phase: Phase, raw: &str, now: u64) -> Result<Vec<String>> {
    let out: Ruled = serde_json::from_str(raw)?;
    let ledger = graph.ledger();
    let mut actions = actions_of(graph, action);
    let mut tally = tally_of(graph, &actions, &marks_of(&ledger, &actions), now);
    let existing = graph.habit_of(action).map(|h| h.id.clone());
    let mut traces = Vec::new();
    match out.verdict.as_str() {
        "habit" => {}
        "project" | "trait" | "noise" => {
            let ruling = match out.verdict.as_str() {
                "project" => Ruling::Project,
                "trait" => Ruling::Trait,
                _ => Ruling::Noise,
            };
            traces.push(format!("○ \"{action}\" ruled {} ({} marks): {}", out.verdict, tally.n_all, out.reason.trim()));
            graph.patterns.verdicts.insert(action.to_string(), Verdict { ruling, at: now, marks: tally.n_all });
            if let Some(id) = existing {
                traces.extend(fold_habit(graph, &id, now));
            }
            return Ok(traces);
        }
        other => anyhow::bail!("pattern ruling with unknown verdict {other:?}"),
    }
    let twin = graph.distillants.get(out.id.trim()).filter(|m| m.tally.is_some() && existing.is_none()).map(|m| m.id.clone());
    if let Some(twin) = &twin {
        let standing = graph.distillants[twin].tally.as_ref().map(|t| t.actions.clone()).unwrap_or_default();
        actions = standing;
        actions.push(action.to_string());
        tally = tally_of(graph, &actions, &marks_of(&ledger, &actions), now);
    }
    let id = habit_id_of(out.id.trim(), existing.as_deref().or(twin.as_deref()));
    if out.line.trim().is_empty() {
        anyhow::bail!("pattern ruling: habit verdict with no line");
    }
    let label = if out.label.trim().is_empty() { id.rsplit('/').next().unwrap_or(&id).replace('-', " ") } else { out.label.trim().to_string() };
    let line_before = graph.distillants.get(&id).map(|m| m.line.clone());
    let mut ops = vec![Op::Distillant {
        id: id.clone(),
        tree: Tree::Registry,
        label,
        line: out.line.trim().to_string(),
        routing: Vec::new(),
        parents: vec![HABITS.to_string()],
    }];
    for leaf in &out.adopt_leaves {
        ops.push(Op::Adopt { leaf: leaf.clone(), parent: id.clone() });
    }
    traces.extend(harvest::apply_ops(graph, ops, now).traces);
    if let (Some(before), Some(m)) = (line_before, graph.distillants.get(&id))
        && before != m.line
    {
        traces.push(format!("✎ distill {id} — {}", m.line));
    }
    if let Some(m) = graph.distillants.get_mut(&id) {
        let dropped: Vec<String> = m
            .routing
            .iter()
            .filter(|t| !out.routing.iter().any(|r| r.trim().eq_ignore_ascii_case(t)))
            .filter(|t| crate::routing::facet_family(t).is_none())
            .cloned()
            .collect();
        m.routing.clear();
        if !dropped.is_empty() {
            traces.push(format!("✂ routing {id} − {} (not in the pass's set)", dropped.join(", ")));
        }
    }
    let (routing, unroutable): (Vec<String>, Vec<String>) =
        out.routing.iter().map(|r| r.trim().to_string()).partition(|r| routable(r));
    let routing: Vec<String> = routing.into_iter().take(ROUTING_MAX).collect();
    if !unroutable.is_empty() {
        traces.push(format!("✋ routing {id} ✗ {} (a word a message about anything could contain)", unroutable.join(", ")));
    }
    harvest::sync_facet_routing(graph, &id, &mut traces);
    harvest::add_routing(graph, &id, &routing, &mut traces);
    if routing.is_empty() {
        traces.push(format!("⚠ habit {id} carries no routing — nothing will recall it"));
    }
    tally.faded = phase == Phase::Fade;
    if let Some(m) = graph.distillants.get_mut(&id) {
        m.tally = Some(tally.clone());
        m.consolidated_at = now;
        m.misc_count = 0;
    }
    for a in &actions {
        graph.patterns.verdicts.remove(a);
    }
    let verb = match (twin.is_some(), phase) {
        (true, _) => "took in",
        (false, Phase::Mint) => "minted for",
        (false, Phase::Refresh) => "refreshed for",
        (false, Phase::Fade) => "faded for",
    };
    traces.push(format!("★ habit {id} {verb} \"{action}\" (×{}, {} adopted): {}", tally.n_all, out.adopt_leaves.len(), out.reason.trim()));
    Ok(traces)
}

pub fn fold_habit(graph: &mut Graph, id: &str, now: u64) -> Vec<String> {
    let Some(m) = graph.distillants.get(id) else {
        return vec![format!("⚠ fold of unknown habit {id}; skipped")];
    };
    let Some(tally) = m.tally.clone() else {
        return vec![format!("⚠ fold of {id}: not a habit; skipped")];
    };
    let actions = tally.actions.join(" + ");
    let line = m.headline();
    let mut homes: Vec<Id> = m.parents.clone();
    if homes.is_empty() {
        homes.push(HABITS.to_string());
    }
    graph.distillants.remove(id);
    let mut kept_leaves = 0usize;
    for leaf in graph.leaves.values_mut() {
        if !leaf.parents.iter().any(|p| p == id) {
            continue;
        }
        leaf.parents.retain(|p| p != id);
        if leaf.parents.is_empty() {
            leaf.parents = homes.clone();
        }
        kept_leaves += 1;
    }
    let child_ids: Vec<Id> = graph.distillants.values().filter(|c| c.parents.iter().any(|p| p == id)).map(|c| c.id.clone()).collect();
    for cid in child_ids {
        let Some(child) = graph.distillants.get(&cid) else { continue };
        let mut parents: Vec<Id> = child.parents.iter().filter(|p| *p != id).cloned().collect();
        parents.extend(homes.iter().cloned());
        let parents = graph.antichain(parents);
        if let Some(child) = graph.distillants.get_mut(&cid) {
            child.parents = parents;
        }
    }
    for e in &mut graph.episodes {
        if e.tags.iter().any(|t| t == id) {
            e.tags.retain(|t| t != id);
            for h in &homes {
                if !e.tags.contains(h) {
                    e.tags.push(h.clone());
                }
            }
        }
    }
    for h in &homes {
        if let Some(p) = graph.distillants.get_mut(h) {
            p.forgotten_at = now;
        }
    }
    let text = format!("A habit that faded — {line}");
    let ep = graph.push_episode_acting(text, homes.clone(), Some(Vec::new()), now, Some(tally.last_seen));
    for a in &tally.actions {
        graph.patterns.verdicts.insert(a.clone(), Verdict { ruling: Ruling::Faded, at: now, marks: tally.n_all });
    }
    vec![format!(
        "⌛ folded habit {id} (\"{actions}\", ×{}, last {}) into [{ep}]; {kept_leaves} leaves keep their other homes",
        tally.n_all,
        age_str(now, tally.last_seen)
    )]
}

pub fn step(llm: &dyn Llm, graph: &mut Graph, now: u64) -> Result<Vec<String>> {
    let Some(step) = due(graph, now) else { return Ok(Vec::new()) };
    let mut traces = vec![format!("⚙ {}", step.describe())];
    let (system, schema, body) = match &step {
        PatternStep::Tag(ids) => (TAG_SYSTEM, tag_schema(), tag_body(graph, ids, now)),
        PatternStep::Rule { action, phase } => (RULE_SYSTEM, rule_schema(), rule_body(graph, action, *phase, now)),
        PatternStep::Fold(id) => {
            traces.extend(fold_habit(graph, id, now));
            return Ok(traces);
        }
    };
    let req = ChatRequest {
        system: system.to_string(),
        messages: vec![json!({"role": "user", "content": body})],
        tools: vec![],
        output_schema: Some(schema),
        max_tokens: 2048,
        thinking: false,
    };
    let resp = llm.chat(&req)?;
    traces.extend(apply(graph, &step, &resp.text(), now)?);
    Ok(traces)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Leaf;

    const NOW: u64 = 100 * DAY;

    fn graph_with_marks(action: &str, days: &[u64], projects: usize) -> Graph {
        let mut g = Graph::seed();
        for (i, d) in days.iter().enumerate() {
            let project = format!("work/p{}", i % projects.max(1));
            g.distillants.entry(project.clone()).or_insert_with(|| Distillant::bare(&project, Tree::Registry, "p", vec!["work".into()], vec![]));
            let at = NOW - d * DAY;
            let ep = g.push_episode_acting(format!("did {action} {i}"), vec![project.clone()], Some(vec![action.to_string()]), at, None);
            let leaf = Leaf::state(format!("fact-{i}"), format!("page {i} is live"), vec![project], 0.5, at);
            let mut leaf = leaf;
            leaf.evidence.push(ep);
            g.leaves.insert(leaf.id.clone(), leaf);
        }
        g
    }

    fn habit_ruling(id: &str, line: &str, adopt: &[&str]) -> String {
        json!({
            "verdict": "habit", "id": id, "label": "Publishes pages", "line": line,
            "routing": ["website", "site", "page", "publish"],
            "adopt_leaves": adopt, "reason": "spans projects"
        })
        .to_string()
    }

    #[test]
    fn untagged_episodes_are_tagged_before_anything_is_ruled() {
        let mut g = graph_with_marks("published-page", &[1, 5, 10, 15, 20], 3);
        g.push_episode("an old episode".into(), vec![], NOW - 30 * DAY, None);
        let Some(PatternStep::Tag(ids)) = due(&g, NOW) else { panic!("tag step first") };
        assert_eq!(ids, vec!["ep-6"]);
        let traces = apply(&mut g, &PatternStep::Tag(ids), r#"{"tags":[{"episode":"ep-6","actions":["Emailed Result"]}]}"#, NOW).unwrap();
        assert!(traces[0].contains("tagged 1"), "{traces:?}");
        assert_eq!(g.episodes[5].actions, Some(vec!["emailed-result".to_string()]));
        assert!(!matches!(due(&g, NOW), Some(PatternStep::Tag(_))));
    }

    #[test]
    fn a_tag_the_model_omits_is_tagged_empty_so_the_backfill_converges() {
        let mut g = Graph::seed();
        g.push_episode("a".into(), vec![], NOW, None);
        g.push_episode("b".into(), vec![], NOW, None);
        let ids = untagged(&g);
        apply(&mut g, &PatternStep::Tag(ids), r#"{"tags":[{"episode":"ep-1","actions":["x"]}]}"#, NOW).unwrap();
        assert_eq!(g.episodes[1].actions, Some(Vec::new()));
        assert!(untagged(&g).is_empty());
    }

    #[test]
    fn candidates_need_marks_spread_and_span() {
        let few = graph_with_marks("x", &[1, 5, 10, 15], 3);
        assert_eq!(due(&few, NOW), None, "four marks is under the floor");
        let one_project = graph_with_marks("x", &[1, 5, 10, 15, 20], 1);
        assert_eq!(due(&one_project, NOW), None, "one project is a project");
        let burst = graph_with_marks("x", &[1, 1, 2, 2, 3], 3);
        assert_eq!(due(&burst, NOW), None, "a three-day burst is not a habit");
        let habit = graph_with_marks("x", &[1, 5, 10, 15, 20], 3);
        assert_eq!(due(&habit, NOW), Some(PatternStep::Rule { action: "x".into(), phase: Phase::Mint }));
    }

    #[test]
    fn tally_counts_windows_gap_and_spread() {
        let g = graph_with_marks("x", &[1, 3, 9, 20, 40], 3);
        let ledger = g.ledger();
        let t = tally_of(&g, &["x".to_string()], &ledger["x"], NOW);
        assert_eq!((t.n_all, t.n_7d, t.n_30d), (5, 2, 4));
        assert_eq!(t.distillants, 3);
        assert_eq!(t.first_seen, NOW - 40 * DAY);
        assert_eq!(t.last_seen, NOW - DAY);
        assert_eq!(t.median_gap_secs, 11 * DAY, "gaps 2, 6, 11, 20 → median 11");
    }

    #[test]
    fn the_ledger_survives_the_fold() {
        let mut g = Graph::seed();
        for i in 0..200u64 {
            g.push_episode_acting(format!("e{i}"), vec![], Some(vec!["x".into()]), i + 1, None);
        }
        g.fold_oldest(150, Some("digest".into()));
        assert_eq!(g.episodes.len(), 51);
        assert_eq!(g.ledger()["x"].len(), 200);
        assert_eq!(g.episodes[0].folded_actions["x"].len(), 150);
        g.fold_oldest(50, None);
        assert_eq!(g.ledger()["x"].len(), 200, "a digest of a digest keeps the marks");
        assert_eq!(g.ledger()["x"][0], 1);
    }

    #[test]
    fn a_habit_ruling_mints_the_node_adopts_evidence_and_routes_plain_words() {
        let mut g = graph_with_marks("published-page", &[1, 5, 10, 15, 20], 3);
        let step = due(&g, NOW).unwrap();
        let p = render_prompt(&g, &step, NOW).unwrap();
        assert!(p.starts_with("# System\n"), "{p}");
        assert!(p.contains("Tally: 5 times"), "{p}");
        assert!(p.contains("[fact-0] (under work/p0)"), "{p}");
        let traces = apply(&mut g, &step, &habit_ruling("habits/publishes-pages", "Publishes a page for almost anything.", &["fact-0", "fact-1", "fact-2"]), NOW).unwrap();
        assert!(traces.iter().any(|t| t.contains("★ habit habits/publishes-pages minted")), "{traces:?}");
        let h = &g.distillants["habits/publishes-pages"];
        assert_eq!(h.parents, vec![HABITS.to_string()]);
        assert_eq!(h.tally.as_ref().map(|t| t.n_all), Some(5));
        assert!(h.routing.iter().any(|r| r == "website"));
        assert_eq!(g.leaves["fact-0"].parents, vec!["work/p0".to_string(), "habits/publishes-pages".to_string()], "adopted beside the project");
        let table = crate::routing::RoutingTable::build(&g);
        assert_eq!(table.matches("make me a website"), vec!["habits/publishes-pages".to_string()]);
        assert_eq!(due(&g, NOW), None, "a fresh habit is quiet");
    }

    #[test]
    fn evidence_shown_is_under_a_distillant_the_event_names_and_not_adopted_elsewhere() {
        let mut g = graph_with_marks("x", &[1, 5, 10, 15, 20], 3);
        let ep0 = g.episodes[0].id.clone();
        let stray = {
            let mut l = Leaf::state("stray".into(), "unrelated fact".into(), vec!["money".into()], 0.5, NOW);
            l.evidence.push(ep0.clone());
            l
        };
        g.leaves.insert(stray.id.clone(), stray);
        g.distillants.insert("habits/other".into(), Distillant::bare("habits/other", Tree::Registry, "Other", vec![HABITS.into()], vec![]));
        g.leaves.get_mut("fact-1").unwrap().parents.push("habits/other".into());
        let p = render_prompt(&g, &PatternStep::Rule { action: "x".into(), phase: Phase::Mint }, NOW).unwrap();
        assert!(p.contains("[fact-0]"), "{p}");
        assert!(!p.contains("[stray]"), "a leaf cited by the batch but filed elsewhere is not this action's evidence: {p}");
        assert!(!p.contains("[fact-1]"), "already adopted by another habit: {p}");
    }

    #[test]
    fn routing_from_a_ruling_drops_words_a_message_about_anything_could_contain() {
        let mut g = graph_with_marks("x", &[1, 5, 10, 15, 20], 3);
        let step = due(&g, NOW).unwrap();
        let raw = json!({
            "verdict": "habit", "id": "habits/x", "label": "X", "line": "Does x.",
            "routing": ["for", "on", "run", "again", "website", "put it online", "x", "a"],
            "adopt_leaves": [], "reason": "r"
        })
        .to_string();
        let traces = apply(&mut g, &step, &raw, NOW).unwrap();
        assert!(traces.iter().any(|t| t.starts_with("✋ routing habits/x ✗ for, on, run, again, x, a")), "{traces:?}");
        let mut routing = g.distillants["habits/x"].routing.clone();
        routing.sort();
        assert_eq!(routing, vec!["put it online", "website"]);
    }

    #[test]
    fn a_trait_verdict_declines_like_noise() {
        let mut g = graph_with_marks("swore", &[1, 5, 10, 15, 20], 3);
        let step = due(&g, NOW).unwrap();
        let raw = r#"{"verdict":"trait","id":"","label":"","line":"","routing":[],"adopt_leaves":[],"reason":"a manner, the profile holds it"}"#;
        let traces = apply(&mut g, &step, raw, NOW).unwrap();
        assert!(traces[0].contains("ruled trait"), "{traces:?}");
        assert_eq!(g.patterns.verdicts["swore"].ruling, Ruling::Trait);
        assert!(g.habit_of("swore").is_none());
    }

    #[test]
    fn a_declined_cluster_is_not_reasked_until_it_grows() {
        let mut g = graph_with_marks("x", &[1, 5, 10, 15, 20], 3);
        let step = due(&g, NOW).unwrap();
        let traces = apply(&mut g, &step, r#"{"verdict":"project","id":"","label":"","line":"","routing":[],"adopt_leaves":[],"reason":"one launch"}"#, NOW).unwrap();
        assert!(traces[0].contains("ruled project"), "{traces:?}");
        assert_eq!(g.patterns.verdicts["x"].marks, 5);
        assert_eq!(due(&g, NOW), None);
        for i in 0..4u64 {
            g.push_episode_acting(format!("more {i}"), vec!["work/p1".into()], Some(vec!["x".into()]), NOW - i * 3600, None);
        }
        assert_eq!(due(&g, NOW), None, "four more marks is under the re-ask stride");
        g.push_episode_acting("fifth".into(), vec!["work/p2".into()], Some(vec!["x".into()]), NOW, None);
        assert_eq!(due(&g, NOW), Some(PatternStep::Rule { action: "x".into(), phase: Phase::Mint }));
    }

    #[test]
    fn a_habit_refreshes_on_growth_fades_when_idle_and_folds_when_gone() {
        let mut g = graph_with_marks("x", &[1, 4, 8, 12, 16], 3);
        let step = due(&g, NOW).unwrap();
        apply(&mut g, &step, &habit_ruling("habits/x", "Does x every few days.", &[]), NOW).unwrap();
        assert_eq!(due(&g, NOW), None);
        for i in 0..3u64 {
            g.push_episode_acting(format!("again {i}"), vec!["work/p0".into()], Some(vec!["x".into()]), NOW + i, None);
        }
        assert_eq!(due(&g, NOW + 3), Some(PatternStep::Rule { action: "x".into(), phase: Phase::Refresh }), "three more marks refresh");
        let traces = apply(&mut g, &PatternStep::Rule { action: "x".into(), phase: Phase::Refresh }, &habit_ruling("habits/x", "Does x most days.", &[]), NOW + 3).unwrap();
        assert!(traces.iter().any(|t| t == "✎ distill habits/x — Does x most days."), "a refreshed line is traced: {traces:?}");
        assert_eq!(g.distillants["habits/x"].tally.as_ref().map(|t| t.n_all), Some(8));

        let later = NOW + 3 + FADE_FLOOR_SECS + DAY;
        assert_eq!(due(&g, later), Some(PatternStep::Rule { action: "x".into(), phase: Phase::Fade }));
        let p = render_prompt(&g, &PatternStep::Rule { action: "x".into(), phase: Phase::Fade }, later).unwrap();
        assert!(p.contains("PAST tense"), "{p}");
        apply(&mut g, &PatternStep::Rule { action: "x".into(), phase: Phase::Fade }, &habit_ruling("habits/x", "Did x most days for a while; none since.", &[]), later).unwrap();
        assert!(g.distillants["habits/x"].tally.as_ref().is_some_and(|t| t.faded));
        assert_eq!(due(&g, later), None, "a faded habit is quiet");

        let gone = NOW + 3 + FOLD_FLOOR_SECS + DAY;
        assert_eq!(due(&g, gone), Some(PatternStep::Fold("habits/x".into())));
        let traces = apply(&mut g, &PatternStep::Fold("habits/x".into()), "", gone).unwrap();
        assert!(traces[0].starts_with("⌛ folded habit habits/x"), "{traces:?}");
        assert!(!g.distillants.contains_key("habits/x"));
        assert!(g.episodes.last().unwrap().text.contains("Did x most days for a while"));
        assert_eq!(g.patterns.verdicts["x"].ruling, Ruling::Faded);
        assert_eq!(due(&g, gone), None, "folded: the count must grow before it is re-asked");
    }

    #[test]
    fn a_habit_that_never_faded_is_faded_before_it_folds() {
        let mut g = graph_with_marks("x", &[1, 4, 8, 12, 16], 3);
        let step = due(&g, NOW).unwrap();
        apply(&mut g, &step, &habit_ruling("habits/x", "Does x every few days.", &[]), NOW).unwrap();
        let long_gone = NOW + FOLD_FLOOR_SECS + DAY;
        assert_eq!(
            due(&g, long_gone),
            Some(PatternStep::Rule { action: "x".into(), phase: Phase::Fade }),
            "the line is still present tense: the model rewrites it before the fold takes it"
        );
        apply(&mut g, &PatternStep::Rule { action: "x".into(), phase: Phase::Fade }, &habit_ruling("habits/x", "Did x for a while; none since.", &[]), long_gone).unwrap();
        assert_eq!(due(&g, long_gone), Some(PatternStep::Fold("habits/x".into())));
    }

    #[test]
    fn folding_keeps_adopted_leaves_at_their_other_homes_and_rehomes_the_rest() {
        let mut g = graph_with_marks("x", &[1, 5, 10, 15, 20], 3);
        let step = due(&g, NOW).unwrap();
        apply(&mut g, &step, &habit_ruling("habits/x", "Does x.", &["fact-0"]), NOW).unwrap();
        let only_here = Leaf::state("only-here".into(), "x only".into(), vec!["habits/x".into()], 0.5, NOW);
        g.leaves.insert(only_here.id.clone(), only_here);
        fold_habit(&mut g, "habits/x", NOW);
        assert_eq!(g.leaves["fact-0"].parents, vec!["work/p0".to_string()]);
        assert_eq!(g.leaves["only-here"].parents, vec![HABITS.to_string()]);
        assert_eq!(g.distillants[HABITS].forgotten_at, NOW, "the crown's line is due");
    }

    #[test]
    fn a_ruling_naming_a_standing_habit_folds_the_action_into_it() {
        let mut g = graph_with_marks("published-page", &[1, 5, 10, 15, 20], 3);
        let step = due(&g, NOW).unwrap();
        apply(&mut g, &step, &habit_ruling("habits/publishes-pages", "Publishes pages.", &[]), NOW).unwrap();
        for i in 0..5u64 {
            g.push_episode_acting(format!("put up {i}"), vec![format!("work/p{}", i % 3)], Some(vec!["put-page-online".into()]), NOW - i * 5 * DAY, None);
        }
        assert_eq!(due(&g, NOW), Some(PatternStep::Rule { action: "put-page-online".into(), phase: Phase::Mint }));
        apply(&mut g, &PatternStep::Rule { action: "put-page-online".into(), phase: Phase::Mint }, &habit_ruling("habits/publishes-pages", "Publishes pages.", &[]), NOW).unwrap();
        assert_eq!(g.distillants.values().filter(|m| m.tally.is_some()).count(), 1, "no twin");
        let t = g.distillants["habits/publishes-pages"].tally.as_ref().unwrap();
        assert_eq!(t.n_all, 10, "both spellings count");
        assert_eq!(t.actions, vec!["published-page".to_string(), "put-page-online".to_string()], "the habit stands for both spellings");
        assert_eq!(due(&g, NOW), None, "the second spelling is covered; nothing comes due again");
        let p = render_prompt(&g, &PatternStep::Rule { action: "published-page".into(), phase: Phase::Refresh }, NOW).unwrap();
        assert!(p.contains("# Repeated action: published-page + put-page-online\nTally: 10 times"), "{p}");
    }

    #[test]
    fn the_mock_driven_step_runs_end_to_end() {
        use crate::llm::MockLlm;
        let mut g = graph_with_marks("x", &[1, 5, 10, 15, 20], 3);
        let llm = MockLlm::scripted(vec![json!([{"type": "text", "text": habit_ruling("habits/x", "Does x.", &[])}])]);
        let traces = step(&llm, &mut g, NOW).unwrap();
        assert!(traces[0].starts_with("⚙ pattern pass due: rule on"), "{traces:?}");
        assert!(g.distillants.contains_key("habits/x"));
    }
}
