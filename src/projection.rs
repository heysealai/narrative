//! Projection (retrieval) and prompt rendering.
//!
//! Two context tiers:
//! - Pinned: the rules and the whole profile tree ride inline always
//!   (small, slow-changing, rendered without ages so the block is
//!   byte-stable between commits).
//! - Retrieved: registry leaves projected per-turn via the routing table.
//!
//! Plus the registry *skeleton* (labels + lines, no leaf bodies), which
//! is the in-context map the model uses for `open_memory` BFS descent.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use crate::belief;
use crate::model::{age_str, Graph, Id, Leaf, LeafKind, Tree};
use crate::routing::{normalize, RoutingTable};

/// The reserved open id that reads the rules with their history — what
/// `render_open` answers when asked for it, beside the distillant ids.
pub const RULES: &str = "rules";

/// The line above the pinned rules: what they are and what they outrank.
pub const RULES_FRAMING: &str = "Standing instructions — every line was explicitly asked for; it outranks every tendency and fact below and the default voice, never the boundaries; only the user changes one.";

pub const PER_MIDPOINT_CAP: usize = 5;
pub const TOTAL_LEAF_CAP: usize = 12;
/// Recency window: how many matching episodes ride along with a projection.
pub const EPISODE_RECALL_CAP: usize = 5;

#[derive(Debug, Clone)]
pub struct OpenedDistillant {
    pub distillant_id: Id,
    pub leaf_ids: Vec<Id>,
}

#[derive(Debug, Clone, Default)]
pub struct Projection {
    pub opened: Vec<OpenedDistillant>,
    /// Recent episodes whose tags hit an activated distillant, oldest first.
    pub episodes: Vec<Id>,
}

impl Projection {
    pub fn is_empty(&self) -> bool {
        self.opened.is_empty() && self.episodes.is_empty()
    }

    pub fn summary(&self, graph: &Graph) -> String {
        let mut parts: Vec<String> = self
            .opened
            .iter()
            .map(|o| {
                let label = graph
                    .distillants
                    .get(&o.distillant_id)
                    .map(|m| m.id.as_str())
                    .unwrap_or(o.distillant_id.as_str());
                format!("{} ({} leaves)", label, o.leaf_ids.len())
            })
            .collect();
        if !self.episodes.is_empty() {
            parts.push(format!("{} episodes", self.episodes.len()));
        }
        parts.join(", ")
    }
}

/// The distinct words of the message, normalized the way the routing table
/// normalizes its terms, so a leaf is compared on the same footing as the
/// distillant that routed it.
struct MessageWords(BTreeSet<String>);

impl MessageWords {
    fn of(text: &str) -> Self {
        MessageWords(normalize(text).split_whitespace().map(str::to_string).collect())
    }

    fn shared_with(&self, leaf: &Leaf) -> usize {
        normalize(&leaf.text)
            .split_whitespace()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .filter(|word| self.0.contains(*word))
            .count()
    }
}

/// How well one leaf answers the message: the message words its text
/// carries, then its salience. Every leaf under a distillant is about the
/// distillant, but only some are about what the message asks; ordered by
/// salience alone, the leaf the message names can fall outside the cap.
#[derive(Clone, Copy)]
struct LeafRank {
    shared_words: usize,
    salience: f32,
}

impl LeafRank {
    const NONE: LeafRank = LeafRank { shared_words: 0, salience: 0.0 };

    fn of(leaf: &Leaf, words: &MessageWords, now: u64) -> Self {
        LeafRank { shared_words: words.shared_with(leaf), salience: belief::score(leaf, now) }
    }

    /// Descending: more shared words first, then higher salience.
    fn order(a: &Self, b: &Self) -> std::cmp::Ordering {
        b.shared_words
            .cmp(&a.shared_words)
            .then_with(|| b.salience.partial_cmp(&a.salience).unwrap_or(std::cmp::Ordering::Equal))
    }
}

/// A distillant the message routed to, with its leaves already in the
/// order the message wants them (best answer first).
struct Activated<'g> {
    distillant_id: Id,
    lexical_score: u32,
    leaves: Vec<(&'g Leaf, LeafRank)>,
}

impl<'g> Activated<'g> {
    fn of(graph: &'g Graph, distillant_id: Id, lexical_score: u32, words: &MessageWords, now: u64) -> Self {
        let mut leaves: Vec<(&Leaf, LeafRank)> = graph
            .leaves_under(&distillant_id)
            .into_iter()
            .map(|leaf| (leaf, LeafRank::of(leaf, words, now)))
            .collect();
        leaves.sort_by(|(_, a), (_, b)| LeafRank::order(a, b));
        Activated { distillant_id, lexical_score, leaves }
    }

    fn best_leaf(&self) -> LeafRank {
        self.leaves.first().map(|(_, rank)| *rank).unwrap_or(LeafRank::NONE)
    }
}

/// Match the outgoing message against the routing table and pre-open the
/// activated distillants' best leaves. Registry only — the profile is pinned.
/// Episodes route too, via their tags: the most recent few tagged with any
/// activated distillant (either tree — episodes are never pinned) ride along.
pub fn project(graph: &Graph, text: &str, now: u64) -> Projection {
    project_with_caps(graph, text, now, PER_MIDPOINT_CAP, TOTAL_LEAF_CAP)
}

/// The harvester cross-matches against a wider window than the agent reads —
/// callers pick their caps; `project` applies the agent-facing defaults.
pub fn project_with_caps(
    graph: &Graph,
    text: &str,
    now: u64,
    per_distillant: usize,
    total_cap: usize,
) -> Projection {
    let table = RoutingTable::build(graph);
    project_with_table(graph, &table, text, now, per_distillant, total_cap)
}

/// The projection over a routing table the caller already built — the
/// harvester matches the same turn against the same table twice (directory
/// and comparanda) and builds it once.
pub fn project_with_table(
    graph: &Graph,
    table: &RoutingTable,
    text: &str,
    now: u64,
    per_distillant: usize,
    total_cap: usize,
) -> Projection {
    // Rank matches: lexical score first, then the best leaf underneath
    // (message words shared, then salience). The leaf budget below goes to
    // the best-ranked matches — table order was alphabetical, and an early
    // weak match could starve the actual answer (the budget-crowding
    // finding of docs/facet-routing-discovery.md).
    let words = MessageWords::of(text);
    let mut activated: Vec<Activated> = table
        .matches_scored(text)
        .into_iter()
        .map(|(mid, score)| Activated::of(graph, mid, score, &words, now))
        .collect();
    activated.sort_by(|a, b| {
        b.lexical_score
            .cmp(&a.lexical_score)
            .then_with(|| LeafRank::order(&a.best_leaf(), &b.best_leaf()))
    });
    let mut total = 0usize;
    let mut opened = Vec::new();
    for a in &activated {
        let Some(m) = graph.distillants.get(&a.distillant_id) else { continue };
        if m.tree == Tree::Profile {
            continue; // pinned tier, already inline
        }
        let take = a
            .leaves
            .iter()
            .take(per_distillant.min(total_cap.saturating_sub(total)))
            .map(|(leaf, _)| leaf.id.clone())
            .collect::<Vec<_>>();
        if take.is_empty() {
            continue;
        }
        total += take.len();
        opened.push(OpenedDistillant { distillant_id: a.distillant_id.clone(), leaf_ids: take });
        if total >= total_cap {
            break;
        }
    }
    let tag_is_activated = |tag: &Id| activated.iter().any(|a| &a.distillant_id == tag);
    let mut episodes: Vec<Id> = graph
        .episodes
        .iter()
        .rev()
        .filter(|e| e.tags.iter().any(tag_is_activated))
        .take(EPISODE_RECALL_CAP)
        .map(|e| e.id.clone())
        .collect();
    episodes.reverse(); // chronological
    Projection { opened, episodes }
}

/// Reinforce what was recalled — retrieval strengthens salience.
pub fn touch(graph: &mut Graph, p: &Projection, now: u64) {
    for o in &p.opened {
        for id in &o.leaf_ids {
            if let Some(l) = graph.leaves.get_mut(id) {
                belief::mark_retrieved(l, now);
            }
        }
    }
}

fn render_leaf_line(out: &mut String, graph: &Graph, leaf_id: &str, now: u64) {
    let Some(l) = graph.leaves.get(leaf_id) else { return };
    match &l.kind {
        LeafKind::State(state) => {
            let _ = write!(out, "- [{}] {}", l.id, l.text);
            if let Some(last) = state.history.last() {
                let _ = write!(out, " (changed {}; was: {})", age_str(now, last.superseded_at), last.value);
            } else {
                let _ = write!(out, " (noted {})", age_str(now, l.occurred_at.unwrap_or(l.created_at)));
            }
            let s = belief::strength(&state.belief);
            if s < 0.45 {
                let _ = write!(
                    out,
                    " [shaky: {} for / {} against]",
                    state.belief.support, state.belief.contradict
                );
            }
            out.push('\n');
        }
        LeafKind::Disposition(d) => {
            let _ = writeln!(
                out,
                "- [{}] {} (axis {:+.2}, {} observations)",
                l.id,
                l.text,
                d.axis,
                d.nudges.len()
            );
        }
        LeafKind::Rule(rule) => {
            let _ = write!(out, "- [{}] {}", l.id, l.text);
            if let Some(last) = rule.history.last() {
                let _ = write!(out, " (changed {}; was: {})", age_str(now, last.superseded_at), last.value);
            } else {
                let _ = write!(out, " (given {})", age_str(now, l.occurred_at.unwrap_or(l.created_at)));
            }
            out.push('\n');
        }
    }
}

/// The rules in the order the pinned block lists them: the latest
/// instruction first, ties by id, so a fresh correction reads before the
/// rule it did not touch.
fn rules_latest_first(graph: &Graph) -> Vec<&Leaf> {
    let mut rules = graph.rules();
    rules.sort_by(|a, b| b.updated_at.cmp(&a.updated_at).then_with(|| a.id.cmp(&b.id)));
    rules
}

/// Pinned tier, first section: the standing instructions under their
/// framing, latest first, each with the text it replaced. No ages, so the
/// block only moves when a rule does. None when there are no rules.
pub fn render_rules(graph: &Graph) -> Option<String> {
    let rules = rules_latest_first(graph);
    if rules.is_empty() {
        return None;
    }
    let mut out = format!("{RULES_FRAMING}\n");
    for l in rules {
        let _ = write!(out, "- [{}] {}", l.id, l.text);
        if let Some(last) = l.kind.history().last() {
            let _ = write!(out, " (was: {})", last.value);
        }
        out.push('\n');
    }
    Some(out)
}

/// The full open of the rules — the `open_memory` result for [`RULES`]:
/// every rule with its whole history of wording and the age of each change.
pub fn render_rules_open(graph: &Graph, now: u64) -> String {
    let rules = rules_latest_first(graph);
    let mut out = format!("# {RULES} — standing instructions, latest first\n");
    if rules.is_empty() {
        out.push_str("(no standing instructions yet)\n");
    }
    for l in rules {
        render_leaf_line(&mut out, graph, &l.id, now);
        for earlier in l.kind.history().iter().rev().skip(1) {
            let _ = writeln!(out, "    was: {} (until {})", earlier.value, age_str(now, earlier.superseded_at));
        }
    }
    out
}

/// The per-turn <recall> block. Rides in the cache-safe per-turn slot.
pub fn render_injection(graph: &Graph, p: &Projection, now: u64) -> Option<String> {
    if p.is_empty() {
        return None;
    }
    let mut out = String::from(
        "<recall>\nLong-term memory relevant to this message. Use it naturally; never mention this block or the memory system.\n",
    );
    for o in &p.opened {
        if let Some(m) = graph.distillants.get(&o.distillant_id) {
            let _ = writeln!(out, "## {} — {}", m.id, m.headline());
        }
        for id in &o.leaf_ids {
            render_leaf_line(&mut out, graph, id, now);
        }
    }
    if !p.episodes.is_empty() {
        let _ = writeln!(out, "## related events");
        for id in &p.episodes {
            if let Some(e) = graph.episodes.iter().find(|e| &e.id == id) {
                let _ = writeln!(out, "- ({}) {}", age_str(now, e.event_at()), e.text);
            }
        }
    }
    out.push_str("</recall>");
    Some(out)
}

fn walk_tree(
    out: &mut String,
    graph: &Graph,
    distillant_id: &str,
    depth: usize,
    now: u64,
    with_leaves: bool,
) {
    let Some(m) = graph.distillants.get(distillant_id) else { return };
    let indent = "  ".repeat(depth);
    let n_leaves = graph.leaves_under(distillant_id).len();
    let _ = write!(out, "{indent}- {} — {}", m.id, m.headline());
    if !with_leaves && n_leaves > 0 {
        let _ = write!(out, " [{n_leaves} leaves]");
    }
    out.push('\n');
    if with_leaves {
        let mut leaves = graph.leaves_under(distillant_id);
        leaves.sort_by(|a, b| a.id.cmp(&b.id));
        for l in leaves {
            let mut line = String::new();
            render_leaf_line(&mut line, graph, &l.id, now);
            for ln in line.lines() {
                let _ = writeln!(out, "{indent}  {ln}");
            }
        }
    }
    let mut children = graph.child_distillants(distillant_id);
    children.sort_by(|a, b| a.id.cmp(&b.id));
    for c in children {
        walk_tree(out, graph, &c.id, depth + 1, now, with_leaves);
    }
}

/// Pinned tier: the whole profile, lines and leaves. Small by
/// construction; lives in the cacheable per-user system block.
pub fn render_profile(graph: &Graph, now: u64) -> String {
    let mut out = String::new();
    let mut roots = graph.roots(Tree::Profile);
    roots.sort_by(|a, b| a.id.cmp(&b.id));
    for r in roots {
        walk_tree(&mut out, graph, &r.id, 0, now, true);
    }
    out
}

/// Registry skeleton: the map the model reads to decide where to descend.
/// Labels + lines + leaf counts — never leaf bodies.
pub fn render_registry_skeleton(graph: &Graph, now: u64) -> String {
    let mut out = String::new();
    let mut roots = graph.roots(Tree::Registry);
    roots.sort_by(|a, b| a.id.cmp(&b.id));
    for r in roots {
        walk_tree(&mut out, graph, &r.id, 0, now, false);
    }
    out
}

/// Full open of one distillant — the `open_memory` tool result, and /open in
/// the sim. The reserved id [`RULES`] opens the standing instructions.
pub fn render_open(graph: &Graph, distillant_id: &str, now: u64) -> String {
    if distillant_id == RULES {
        return render_rules_open(graph, now);
    }
    let Some(m) = graph.distillants.get(distillant_id) else {
        return format!(
            "No distillant with id \"{distillant_id}\". Use an id exactly as it appears in the memory map, or \"{RULES}\" for the standing instructions."
        );
    };
    let mut out = format!("# {} — {}\n", m.id, m.headline());
    let mut leaves = graph.leaves_under(distillant_id);
    leaves.sort_by(|a, b| {
        belief::score(b, now)
            .partial_cmp(&belief::score(a, now))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if leaves.is_empty() {
        out.push_str("(no facts stored directly here)\n");
    }
    for l in leaves {
        render_leaf_line(&mut out, graph, &l.id, now);
    }
    let recent: Vec<&crate::model::Episode> = graph
        .episodes
        .iter()
        .rev()
        .filter(|e| e.tags.iter().any(|t| t == distillant_id))
        .take(8)
        .collect();
    if !recent.is_empty() {
        out.push_str("Recent events here:\n");
        for e in recent.into_iter().rev() {
            let _ = writeln!(out, "- ({}) {}", age_str(now, e.event_at()), e.text);
        }
    }
    let mut children = graph.child_distillants(distillant_id);
    children.sort_by(|a, b| a.id.cmp(&b.id));
    if !children.is_empty() {
        out.push_str("Child distillants:\n");
        for c in children {
            let _ = writeln!(
                out,
                "- {} — {} [{} leaves]",
                c.id,
                c.headline(),
                graph.leaves_under(&c.id).len()
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Graph, Leaf, Distillant, Tree};

    fn fixture() -> Graph {
        let mut g = Graph::seed();
        g.distillants.insert(
            "money/rent".into(),
            Distillant {
                id: "money/rent".into(),
                tree: Tree::Registry,
                label: "Rent".into(),
                line: "Rent and landlord dealings.".into(),
                routing: vec!["rent".into(), "landlord".into()],
                parents: vec!["money".into()],
                misc_count: 0,
                consolidated_at: 0,
                line_changed_at: 0,
                forgotten_at: 0,
            },
        );
        for i in 0..8 {
            let l = Leaf::state(
                format!("rent-fact-{i}"),
                format!("rent fact number {i}"),
                vec!["money/rent".into()],
                0.1 + 0.1 * i as f32,
                1_000,
            );
            g.leaves.insert(l.id.clone(), l);
        }
        g
    }

    #[test]
    fn projection_caps_and_ranks() {
        let g = fixture();
        let p = project(&g, "is my rent due?", 2_000);
        // 'money' (crown) activates too but has no direct leaves, so only
        // 'money/rent' actually opens.
        assert_eq!(p.opened.len(), 1);
        let rent = p
            .opened
            .iter()
            .find(|o| o.distillant_id == "money/rent")
            .expect("rent distillant opened");
        assert_eq!(rent.leaf_ids.len(), PER_MIDPOINT_CAP);
        assert_eq!(rent.leaf_ids[0], "rent-fact-7", "highest importance first");
    }

    #[test]
    fn the_leaf_the_message_names_survives_the_cap() {
        let mut g = fixture();
        // The least salient leaf under rent.
        let l = Leaf::state(
            "rent-deposit".into(),
            "The deposit came back in full.".into(),
            vec!["money/rent".into()],
            0.05,
            1_000,
        );
        g.leaves.insert(l.id.clone(), l);
        let p = project(&g, "did my rent deposit come back?", 2_000);
        let rent = p.opened.iter().find(|o| o.distillant_id == "money/rent").unwrap();
        assert_eq!(rent.leaf_ids[0], "rent-deposit", "the message's own words outrank salience");
        assert_eq!(rent.leaf_ids.len(), PER_MIDPOINT_CAP, "the cap still holds");
        assert_eq!(rent.leaf_ids[1], "rent-fact-7", "salience orders the leaves the message names equally");
    }

    #[test]
    fn a_leaf_naming_the_message_lifts_its_distillant_on_equal_score() {
        let mut g = Graph::seed();
        for (mid, text) in [("life/aaa", "fact"), ("life/zzz", "the roof leaks")] {
            g.distillants.insert(
                mid.into(),
                Distillant {
                    id: mid.into(),
                    tree: Tree::Registry,
                    label: mid.into(),
                    line: "x".into(),
                    routing: vec!["home".into()],
                    parents: vec!["life".into()],
                    misc_count: 0,
                    consolidated_at: 0,
                    line_changed_at: 0,
                    forgotten_at: 0,
                },
            );
            let l = Leaf::state(
                format!("{}-leaf", mid.replace('/', "-")),
                text.into(),
                vec![mid.into()],
                0.5,
                1_000,
            );
            g.leaves.insert(l.id.clone(), l);
        }
        let p = project(&g, "home roof again", 2_000);
        assert_eq!(p.opened[0].distillant_id, "life/zzz", "equal lexical score, equal salience: the shared word decides");
    }

    #[test]
    fn no_match_no_injection() {
        let g = fixture();
        let p = project(&g, "completely unrelated chatter", 2_000);
        assert!(render_injection(&g, &p, 2_000).is_none());
    }

    #[test]
    fn injection_contains_leaf_text_and_ids() {
        let g = fixture();
        let p = project(&g, "rent question", 2_000);
        let inj = render_injection(&g, &p, 2_000).unwrap();
        assert!(inj.contains("money/rent"));
        assert!(inj.contains("[rent-fact-7]"));
        assert!(inj.starts_with("<recall>"));
    }

    #[test]
    fn skeleton_hides_leaf_bodies() {
        let g = fixture();
        let s = render_registry_skeleton(&g, 2_000);
        assert!(s.contains("money/rent"));
        assert!(s.contains("[8 leaves]"));
        assert!(!s.contains("rent fact number"), "skeleton must not leak leaf bodies");
    }

    #[test]
    fn touch_reinforces() {
        let mut g = fixture();
        let p = project(&g, "rent", 2_000);
        touch(&mut g, &p, 2_000);
        assert_eq!(g.leaves["rent-fact-7"].kind.salience().unwrap().retrieval_count, 1);
    }

    #[test]
    fn budget_goes_to_best_ranked_matches_not_table_order() {
        let mut g = Graph::seed();
        // Three distillants all routed by "home"; the alphabetically-last one
        // holds the important leaf. Each has 5 leaves so the 12-leaf budget
        // cannot fit all three — ranking decides who is starved.
        for (mid, importance) in [("life/aaa", 0.2f32), ("life/mmm", 0.2), ("life/zzz", 0.9)] {
            g.distillants.insert(
                mid.into(),
                Distillant {
                    id: mid.into(),
                    tree: Tree::Registry,
                    label: mid.into(),
                    line: "x".into(),
                    routing: vec!["home".into()],
                    parents: vec!["life".into()],
                    misc_count: 0,
                    consolidated_at: 0,
                    line_changed_at: 0,
                    forgotten_at: 0,
                },
            );
            for i in 0..5 {
                let l = Leaf::state(
                    format!("{}-{i}", mid.replace('/', "-")),
                    "fact".into(),
                    vec![mid.into()],
                    importance,
                    1_000,
                );
                g.leaves.insert(l.id.clone(), l);
            }
        }
        let p = project(&g, "thinking about home", 2_000);
        assert_eq!(p.opened[0].distillant_id, "life/zzz", "best leaf outranks table order");
        assert_eq!(p.opened[0].leaf_ids.len(), 5);
        let total: usize = p.opened.iter().map(|o| o.leaf_ids.len()).sum();
        assert_eq!(total, TOTAL_LEAF_CAP, "budget still fills");
        let starved = &p.opened.last().unwrap();
        assert_eq!(starved.leaf_ids.len(), 2, "the worst-ranked match absorbs the shortfall");

        // Lexical score dominates leaf salience: a two-term match on the
        // weak distillant puts it first.
        if let Some(m) = g.distillants.get_mut("life/aaa") {
            m.routing.push("moving house".into());
        }
        let p = project(&g, "thinking about moving house back home", 2_000);
        assert_eq!(p.opened[0].distillant_id, "life/aaa", "phrase + token beats one token");
    }

    #[test]
    fn wider_caps_widen_the_window() {
        let g = fixture();
        let p = project_with_caps(&g, "is my rent due?", 2_000, 12, 48);
        let rent = p.opened.iter().find(|o| o.distillant_id == "money/rent").unwrap();
        assert_eq!(rent.leaf_ids.len(), 8, "all leaves visible to the harvester");
    }

    #[test]
    fn episodes_route_through_tags() {
        let mut g = fixture();
        for i in 0..8 {
            g.push_episode(format!("rent event {i}"), vec!["money/rent".into()], 1_000 + i, None);
        }
        g.push_episode("unrelated event".into(), vec!["work".into()], 2_000, None);
        let p = project(&g, "rent question", 3_000);
        assert_eq!(p.episodes.len(), EPISODE_RECALL_CAP, "recency window caps the ride-along");
        let inj = render_injection(&g, &p, 3_000).unwrap();
        assert!(inj.contains("## related events"));
        assert!(inj.contains("rent event 7"), "most recent episodes included");
        assert!(!inj.contains("rent event 0"), "oldest fall outside the window");
        assert!(!inj.contains("unrelated event"), "unmatched tags stay out");
    }

    #[test]
    fn episode_only_match_still_injects() {
        let mut g = Graph::seed();
        g.push_episode("Paid rent late.".into(), vec!["money".into()], 1_000, None);
        let p = project(&g, "money stuff", 2_000);
        assert!(p.opened.is_empty(), "no leaves under the crown root");
        assert!(!p.is_empty(), "the episode alone keeps the projection alive");
        let inj = render_injection(&g, &p, 2_000).unwrap();
        assert!(inj.contains("Paid rent late."));
    }

    #[test]
    fn event_time_governs_rendered_age() {
        let mut g = fixture();
        let day = 86_400;
        let now = 100 * day;
        let mut l = Leaf::state(
            "old-fact".into(),
            "Sold the plantation.".into(),
            vec!["money/rent".into()],
            0.5,
            now, // written today...
        );
        l.occurred_at = Some(now - 60 * day); // ...about something two months back
        g.leaves.insert(l.id.clone(), l);
        let mut out = String::new();
        render_leaf_line(&mut out, &g, "old-fact", now);
        assert!(out.contains("(noted 2mo ago)"), "story time, not write time: {out}");

        g.push_episode("Washed ashore.".into(), vec!["money/rent".into()], now, Some(now - 30 * day));
        let p = project(&g, "rent", now);
        let inj = render_injection(&g, &p, now).unwrap();
        assert!(inj.contains("(1mo ago) Washed ashore."), "episode age from event time: {inj}");
    }

    #[test]
    fn rules_render_pinned_without_ages_and_open_with_them() {
        let mut g = fixture();
        assert!(render_rules(&g).is_none(), "no rules, no block");
        let day = 86_400;
        let mut five = Leaf::rule("five-lines".into(), "keep replies to five lines".into(), Some("i-1".into()), 10 * day);
        five.kind = LeafKind::Rule(crate::model::RuleKind {
            history: vec![crate::model::Supersession { value: "keep replies to three lines".into(), superseded_at: 8 * day }],
            instruction: Some("i-2".into()),
        });
        g.leaves.insert(five.id.clone(), five);
        let ask = Leaf::rule("ask-first".into(), "ask before any spend over $20".into(), None, 12 * day);
        g.leaves.insert(ask.id.clone(), ask);

        let pinned = render_rules(&g).unwrap();
        assert!(pinned.starts_with(RULES_FRAMING), "{pinned}");
        let lines: Vec<&str> = pinned.lines().skip(1).collect();
        assert_eq!(lines[0], "- [ask-first] ask before any spend over $20", "latest instruction first");
        assert_eq!(lines[1], "- [five-lines] keep replies to five lines (was: keep replies to three lines)");
        assert!(!pinned.contains("ago"), "the pinned block carries no ages: {pinned}");
        assert_eq!(render_rules(&g), render_rules(&g), "byte-stable");

        let opened = render_open(&g, RULES, 20 * day);
        assert!(opened.contains("- [ask-first] ask before any spend over $20 (given 8d ago)"), "{opened}");
        assert!(opened.contains("- [five-lines] keep replies to five lines (changed 12d ago; was: keep replies to three lines)"), "{opened}");
        assert!(!render_profile(&g, 20 * day).contains("five-lines"), "a rule is not part of the profile");
        assert!(project(&g, "five lines please", 20 * day).opened.is_empty(), "no routing reaches a rule");
        assert!(render_open(&g, "nope", 1).contains(RULES), "the miss names the reserved id");
    }

    #[test]
    fn open_lists_recent_events() {
        let mut g = fixture();
        g.push_episode("Negotiated with landlord.".into(), vec!["money/rent".into()], 1_500, None);
        let out = render_open(&g, "money/rent", 2_000);
        assert!(out.contains("Recent events here:"));
        assert!(out.contains("Negotiated with landlord."));
    }
}
