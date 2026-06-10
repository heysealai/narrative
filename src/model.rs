//! Core data model: the three stores (stream / registry / profile).
//!
//! - Stream: time-ordered episodes, the shared evidence pool.
//! - Registry: noun-shaped tree of state facts (current value + supersession history).
//! - Profile: trait-shaped tree of dispositions (scored axes with trajectories).
//!
//! Both trees are DAGs of `Distillant`s; `Leaf`s hang off distillants (multi-parent).

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

pub type Id = String;

pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "snake_case")]
pub enum Tree {
    Registry,
    Profile,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "snake_case")]
pub enum Species {
    State,
    Disposition,
}

/// One event in the stream. Immutable once written.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Episode {
    pub id: Id,
    /// Write time: when the engine recorded this.
    pub at: u64,
    /// Event time: when it actually happened, when known (backlog imports,
    /// "last month", story time). Absent = it happened at write time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub occurred_at: Option<u64>,
    pub text: String,
    #[serde(default)]
    pub tags: Vec<String>,
}

impl Episode {
    pub fn event_at(&self) -> u64 {
        self.occurred_at.unwrap_or(self.at)
    }
}

/// Belief is derived from countable events — the model classifies relations,
/// the runtime does the arithmetic. Never model-assigned numbers.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Belief {
    pub support: u32,
    pub contradict: u32,
    pub last_event_at: u64,
    /// Write time of the last event that moved AGAINST the then-current text:
    /// a contradiction, a supersession, or a genuine disposition flip. Drift
    /// arithmetic compares this to the parent's `consolidated_at` — a leaf
    /// that moved against itself after the line was written is evidence
    /// the cached line no longer follows from its children.
    #[serde(default)]
    pub last_against_at: u64,
}

impl Default for Belief {
    fn default() -> Self {
        Belief { support: 1, contradict: 0, last_event_at: 0, last_against_at: 0 }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Salience {
    pub importance: f32,
    pub retrieval_count: u32,
    pub last_retrieved_at: Option<u64>,
}

impl Default for Salience {
    fn default() -> Self {
        Salience { importance: 0.5, retrieval_count: 0, last_retrieved_at: None }
    }
}

/// A superseded value of a state fact. History of change is itself memory.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Supersession {
    pub value: String,
    pub superseded_at: u64,
}

/// One observation moving a disposition axis. The trajectory is memory too.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Nudge {
    pub dir: i8,
    pub note: String,
    pub at: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Leaf {
    pub id: Id,
    pub species: Species,
    /// Small, boring, atomic. States: the current value sentence.
    /// Dispositions: the axis statement ("tends to overspend late-month").
    pub text: String,
    /// Distillant ids this leaf hangs under (multi-parent DAG).
    pub parents: Vec<Id>,
    #[serde(default)]
    pub salience: Salience,
    #[serde(default)]
    pub belief: Belief,
    /// States only: prior values, oldest first.
    #[serde(default)]
    pub history: Vec<Supersession>,
    /// Dispositions only: EMA position in [-1, 1].
    #[serde(default)]
    pub axis: f32,
    /// Dispositions only: the nudge trail.
    #[serde(default)]
    pub nudges: Vec<Nudge>,
    /// Episode ids backing this leaf.
    #[serde(default)]
    pub evidence: Vec<Id>,
    /// Event time of the current value, when known (distinct from write time).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub occurred_at: Option<u64>,
    pub created_at: u64,
    pub updated_at: u64,
}

impl Leaf {
    pub fn new(id: Id, species: Species, text: String, parents: Vec<Id>, at: u64) -> Self {
        Leaf {
            id,
            species,
            text,
            parents,
            salience: Salience::default(),
            belief: Belief { support: 1, contradict: 0, last_event_at: at, last_against_at: 0 },
            history: Vec::new(),
            axis: 0.0,
            nudges: Vec::new(),
            evidence: Vec::new(),
            occurred_at: None,
            created_at: at,
            updated_at: at,
        }
    }
}

/// A distillant: the distilled layer of the tree. Simultaneously index
/// (routing) and understanding (line) — the behavioral median of what
/// hangs below it.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Distillant {
    pub id: Id,
    pub tree: Tree,
    pub label: String,
    /// One line. A cached judgment — consolidation keeps it fresh.
    /// (alias: pre-rename saves call this field "distillant")
    #[serde(alias = "distillant")]
    pub line: String,
    /// Compiled recall vocabulary: words/phrases that route messages here.
    #[serde(default)]
    pub routing: Vec<String>,
    /// Parent distillant ids; empty = crown (root) node.
    #[serde(default)]
    pub parents: Vec<Id>,
    /// Residual counter: facts that landed here for lack of anywhere better.
    #[serde(default)]
    pub misc_count: u32,
    /// Write time of the last consolidation pass over this distillant (0 = never).
    /// The automatic trigger only re-fires once new material lands after this.
    #[serde(default)]
    pub consolidated_at: u64,
    /// Write time of the last MATERIAL line change (0 = never). Stamped
    /// only when the text actually differs — a rewrite that comes back
    /// identical proves the line absorbed the churn, and the cascade stops
    /// there. Parents read this for drift: a child line that changed after
    /// the parent's pass means the parent summarized something that no
    /// longer exists.
    #[serde(default, alias = "distillant_changed_at")]
    pub line_changed_at: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Graph {
    #[serde(default)]
    pub episodes: Vec<Episode>,
    #[serde(default, alias = "midpoints")]
    pub distillants: BTreeMap<Id, Distillant>,
    #[serde(default)]
    pub leaves: BTreeMap<Id, Leaf>,
}

/// Soft cap: a stream past this is due for a distilled digest pass —
/// consolidation owns it, and the model sees the full episode texts while
/// they are still alive.
pub const STREAM_CAP: usize = 1000;
/// How many of the oldest episodes one digest absorbs when nothing chooses
/// otherwise: the mechanical hard-cap fold, and the model's default cut.
pub const COMPRESS_BATCH: usize = 100;
/// How many of the oldest episodes a distilled digest pass shows the model —
/// the cut window. The model picks the actual fold size within it, so a
/// digest can end at a period boundary instead of an arbitrary count.
pub const DIGEST_WINDOW: usize = COMPRESS_BATCH + COMPRESS_BATCH / 2;
/// Smallest fold the model may cut: digests of tiny periods thrash the pass.
pub const DIGEST_MIN_CUT: usize = COMPRESS_BATCH / 2;
/// Hard cap: `push_episode` folds mechanically past this, so the stream stays
/// bounded even keyless/offline. The soft→hard window is consolidation's
/// chance to distill a batch before the lossy fold destroys it.
pub const STREAM_HARD_CAP: usize = STREAM_CAP + COMPRESS_BATCH;
/// Character budget for a digest episode's text.
const DIGEST_TEXT_BUDGET: usize = 1500;
/// Text marker of a mechanical (fragments-only) digest. Consolidation
/// polishes episodes carrying it; a distilled digest never starts with this.
pub const MECH_DIGEST_PREFIX: &str = "[digest of ";

impl Graph {
    /// The near-universal crown. Personal signature grows in the middle layers;
    /// the harvester creates those. Roots converge fast for everyone.
    pub fn seed() -> Graph {
        let mut g = Graph::default();
        let crown = [
            (Tree::Registry, "people", "People", "Family, friends, contacts, counterparties.",
             vec!["friend", "family", "sister", "brother", "mom", "dad", "partner", "wife", "husband", "boss", "landlord"]),
            (Tree::Registry, "money", "Money", "Accounts, obligations, income, spending.",
             vec!["money", "pay", "paid", "payment", "rent", "bill", "bills", "budget", "salary", "price", "cost", "owe", "bought", "buy"]),
            (Tree::Registry, "work", "Work", "Job, projects, professional life.",
             vec!["work", "job", "project", "meeting", "deadline", "office", "client"]),
            (Tree::Registry, "life", "Life", "Health, home, habits, interests.",
             vec!["home", "health", "gym", "trip", "travel", "hobby", "weekend"]),
            (Tree::Profile, "communication", "Communication style", "How they like to be spoken to.",
             vec![]),
            (Tree::Profile, "money-style", "Money style", "How they handle money: discipline, risk, planning.",
             vec![]),
            (Tree::Profile, "temperament", "Temperament", "Disposition and decision-making style.",
             vec![]),
        ];
        for (tree, id, label, line, routing) in crown {
            g.distillants.insert(
                id.to_string(),
                Distillant {
                    id: id.to_string(),
                    tree,
                    label: label.to_string(),
                    line: line.to_string(),
                    routing: routing.into_iter().map(|s| s.to_string()).collect(),
                    parents: Vec::new(),
                    misc_count: 0,
                    consolidated_at: 0,
                    line_changed_at: 0,
                },
            );
        }
        g
    }

    pub fn leaves_under(&self, distillant_id: &str) -> Vec<&Leaf> {
        self.leaves
            .values()
            .filter(|l| l.parents.iter().any(|p| p == distillant_id))
            .collect()
    }

    pub fn child_distillants(&self, distillant_id: &str) -> Vec<&Distillant> {
        self.distillants
            .values()
            .filter(|m| m.parents.iter().any(|p| p == distillant_id))
            .collect()
    }

    pub fn roots(&self, tree: Tree) -> Vec<&Distillant> {
        self.distillants
            .values()
            .filter(|m| m.tree == tree && m.parents.is_empty())
            .collect()
    }

    pub fn push_episode(
        &mut self,
        text: String,
        tags: Vec<String>,
        at: u64,
        occurred_at: Option<u64>,
    ) -> Id {
        let mut n = self.episodes.len() + 1;
        let mut id = format!("ep-{n}");
        while self.episodes.iter().any(|e| e.id == id) {
            n += 1;
            id = format!("ep-{n}");
        }
        self.episodes.push(Episode { id: id.clone(), at, occurred_at, text, tags });
        while self.episodes.len() > STREAM_HARD_CAP {
            self.fold_oldest(COMPRESS_BATCH, None);
        }
        id
    }

    /// Fold the oldest `take` episodes into one digest episode, remapping
    /// leaf evidence so pointers stay resolvable. With `text` (a distilled
    /// period summary, written from the full batch) the digest is final;
    /// without, the fold is lossy but mechanical — fragments under
    /// MECH_DIGEST_PREFIX that consolidation can still polish, though the
    /// unquoted remainder of the batch is gone.
    pub fn fold_oldest(&mut self, take: usize, text: Option<String>) -> Id {
        let take = take.min(self.episodes.len());
        let batch: Vec<Episode> = self.episodes.drain(0..take).collect();
        let mut tags: Vec<String> = Vec::new();
        for e in &batch {
            for t in &e.tags {
                if !tags.contains(t) {
                    tags.push(t.clone());
                }
            }
        }
        let text = text.unwrap_or_else(|| {
            let mut text = format!("{MECH_DIGEST_PREFIX}{} earlier episodes] ", batch.len());
            let mut omitted = 0usize;
            for e in &batch {
                if text.len() < DIGEST_TEXT_BUDGET {
                    if !text.ends_with("] ") {
                        text.push_str("; ");
                    }
                    text.extend(e.text.chars().take(100));
                } else {
                    omitted += 1;
                }
            }
            if omitted > 0 {
                text.push_str(&format!(" (+{omitted} more)"));
            }
            text
        });
        let mut n = 1;
        let mut id = format!("ep-digest-{n}");
        while self.episodes.iter().any(|e| e.id == id) {
            n += 1;
            id = format!("ep-digest-{n}");
        }
        let digest = Episode {
            id: id.clone(),
            at: batch.last().map(|e| e.at).unwrap_or(0),
            occurred_at: batch.first().map(|e| e.event_at()),
            text,
            tags,
        };
        self.episodes.insert(0, digest);
        let dropped: Vec<&str> = batch.iter().map(|e| e.id.as_str()).collect();
        for leaf in self.leaves.values_mut() {
            let before = leaf.evidence.len();
            leaf.evidence.retain(|e| !dropped.contains(&e.as_str()));
            if leaf.evidence.len() != before && !leaf.evidence.contains(&id) {
                leaf.evidence.push(id.clone());
            }
        }
        id
    }

    /// True when `anc` is reachable from `of` by walking parent links —
    /// the cycle guard for structural ops (reparent / merge).
    pub fn is_ancestor(&self, anc: &str, of: &str) -> bool {
        let mut stack: Vec<&str> = vec![of];
        let mut seen: Vec<&str> = Vec::new();
        while let Some(id) = stack.pop() {
            if let Some(m) = self.distillants.get(id) {
                for p in &m.parents {
                    if p == anc {
                        return true;
                    }
                    if !seen.contains(&p.as_str()) {
                        seen.push(p);
                        stack.push(p);
                    }
                }
            }
        }
        false
    }
}

pub fn slugify(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || c == '/' {
            out.push(c.to_ascii_lowercase());
        } else if (c == ' ' || c == '-' || c == '_') && !out.ends_with('-') && !out.ends_with('/') && !out.is_empty() {
            out.push('-');
        }
    }
    out.trim_matches('-').to_string()
}

/// Human-readable age, for prompt rendering ("3d ago", "2mo ago").
pub fn age_str(now: u64, then: u64) -> String {
    let secs = now.saturating_sub(then);
    let days = secs / 86_400;
    if days == 0 {
        "today".to_string()
    } else if days < 30 {
        format!("{days}d ago")
    } else if days < 365 {
        format!("{}mo ago", days / 30)
    } else {
        format!("{:.1}y ago", days as f32 / 365.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_has_crown_roots() {
        let g = Graph::seed();
        assert!(g.distillants.contains_key("money"));
        assert!(g.distillants.contains_key("communication"));
        assert_eq!(g.roots(Tree::Registry).len(), 4);
        assert_eq!(g.roots(Tree::Profile).len(), 3);
    }

    #[test]
    fn episode_ids_unique_and_hard_capped() {
        let mut g = Graph::default();
        let a = g.push_episode("one".into(), vec![], 1, None);
        let b = g.push_episode("two".into(), vec![], 2, None);
        assert_ne!(a, b);
        for i in 0..(STREAM_HARD_CAP + 10) {
            g.push_episode(format!("e{i}"), vec![], i as u64, None);
        }
        assert!(g.episodes.len() <= STREAM_HARD_CAP);
        assert!(
            g.episodes[0].id.starts_with("ep-digest-"),
            "overflow compresses into a digest instead of dropping"
        );
        assert!(g.episodes[0].text.starts_with(MECH_DIGEST_PREFIX), "the emergency fold is marked mechanical");
        assert!(g.episodes[0].text.contains("one"), "digest carries the oldest texts");
    }

    #[test]
    fn soft_window_does_not_compress() {
        let mut g = Graph::default();
        for i in 0..(STREAM_CAP + 50) {
            g.push_episode(format!("e{i}"), vec![], i as u64, None);
        }
        assert_eq!(g.episodes.len(), STREAM_CAP + 50, "between soft and hard cap nothing folds");
        assert!(
            !g.episodes[0].id.starts_with("ep-digest-"),
            "the window belongs to consolidation, not the mechanical fold"
        );
    }

    #[test]
    fn compression_remaps_leaf_evidence_to_digest() {
        let mut g = Graph::default();
        let first = g.push_episode("scammed by X".into(), vec![], 1, None);
        let mut l = Leaf::new("scam".into(), Species::State, "Got scammed once.".into(), vec![], 1);
        l.evidence.push(first.clone());
        g.leaves.insert(l.id.clone(), l);
        for i in 0..(STREAM_HARD_CAP + 1) {
            g.push_episode(format!("e{i}"), vec![format!("tag-{}", i % 3)], i as u64, None);
        }
        let leaf = &g.leaves["scam"];
        assert!(!leaf.evidence.contains(&first), "dropped id no longer referenced");
        assert_eq!(leaf.evidence, vec!["ep-digest-1"], "evidence remapped to the digest");
        assert!(g.episodes[0].tags.contains(&"tag-0".to_string()), "digest unions tags");
    }

    #[test]
    fn event_time_preferred_over_write_time() {
        let mut g = Graph::default();
        g.push_episode("imported".into(), vec![], 1_000_000, Some(5));
        assert_eq!(g.episodes[0].event_at(), 5);
        g.push_episode("live".into(), vec![], 1_000_000, None);
        assert_eq!(g.episodes[1].event_at(), 1_000_000);
    }

    #[test]
    fn ancestor_walk_detects_chains_and_self() {
        let mut g = Graph::seed();
        for (id, parent) in [("life/island", "life"), ("life/island/animals", "life/island")] {
            g.distillants.insert(
                id.into(),
                Distillant {
                    id: id.into(),
                    tree: Tree::Registry,
                    label: id.into(),
                    line: String::new(),
                    routing: vec![],
                    parents: vec![parent.into()],
                    misc_count: 0,
                    consolidated_at: 0,
                    line_changed_at: 0,
                },
            );
        }
        assert!(g.is_ancestor("life", "life/island/animals"));
        assert!(g.is_ancestor("life/island", "life/island/animals"));
        assert!(!g.is_ancestor("life/island/animals", "life"));
        assert!(!g.is_ancestor("money", "life/island"));
    }

    #[test]
    fn slugify_basic() {
        assert_eq!(slugify("My Sister Lisa"), "my-sister-lisa");
        assert_eq!(slugify("money/rent"), "money/rent");
        assert_eq!(slugify("  Hello,  World! "), "hello-world");
    }
}
