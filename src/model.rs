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
    /// Write time of the last forget that removed a leaf or child under
    /// this distillant (0 = never). A line written over something that has
    /// since been forgotten is suspect regardless of what remains — drift
    /// reads this like a leaf that moved against its text.
    #[serde(default)]
    pub forgotten_at: u64,
}

impl Distillant {
    /// A distillant is bare when nothing has written its judgment. Every
    /// write that means one — a harvest declaring the distillant, a pass
    /// rewriting it — stamps `line_changed_at`, so an unstamped line is
    /// text that was never a judgment: a seed description, a stub named
    /// by a leaf before anything distilled it. A stamped line that is
    /// empty or only repeats the label says nothing either. Bare is a
    /// state, not a line: every render says so instead of printing a
    /// description as if it were a judgment, and consolidation treats it
    /// as due the moment there is material to judge.
    pub fn is_bare(&self) -> bool {
        let never_written = self.line_changed_at == 0;
        let line = self.line.trim();
        let says_nothing = line.is_empty() || line.eq_ignore_ascii_case(self.label.trim());
        never_written || says_nothing
    }

    /// The one-line text every render shows for this distillant: its line,
    /// or its label marked as still unwritten.
    pub fn headline(&self) -> String {
        if self.is_bare() {
            format!("{} (no line yet)", self.label)
        } else {
            self.line.clone()
        }
    }

    /// A distillant nothing has judged yet: no line, no stamps.
    pub fn bare(id: &str, tree: Tree, label: &str, parents: Vec<Id>, routing: Vec<Id>) -> Distillant {
        Distillant {
            id: id.to_string(),
            tree,
            label: label.to_string(),
            line: String::new(),
            routing,
            parents,
            misc_count: 0,
            consolidated_at: 0,
            line_changed_at: 0,
            forgotten_at: 0,
        }
    }
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

/// The profile tree's single root: the whole-person estimate. Its line is
/// the character sketch, distilled from the axis lines below it; every
/// profile axis hangs under it, so the pinned tier opens with who this
/// person is before it lists how they tend.
pub const PROFILE_APEX: &str = "character";

impl Graph {
    /// The near-universal crown. Personal signature grows in the middle layers;
    /// the harvester creates those. Roots converge fast for everyone.
    ///
    /// Every seed is born bare: a crown's line is a judgment about this
    /// person, and nothing has judged yet. The first pass over a crown
    /// with material under it writes the first line.
    pub fn seed() -> Graph {
        let mut g = Graph::default();
        let crown = [
            (Tree::Registry, "people", "People", Vec::new(),
             vec!["friend", "family", "sister", "brother", "mom", "dad", "partner", "wife", "husband", "boss", "landlord"]),
            (Tree::Registry, "money", "Money", Vec::new(),
             vec!["money", "pay", "paid", "payment", "rent", "bill", "bills", "budget", "salary", "price", "cost", "owe", "bought", "buy"]),
            (Tree::Registry, "work", "Work", Vec::new(),
             vec!["work", "job", "project", "meeting", "deadline", "office", "client"]),
            (Tree::Registry, "life", "Life", Vec::new(),
             vec!["home", "health", "gym", "trip", "travel", "hobby", "weekend"]),
            (Tree::Profile, PROFILE_APEX, "Character", Vec::new(), vec![]),
            (Tree::Profile, "communication", "Communication style", vec![PROFILE_APEX], vec![]),
            (Tree::Profile, "money-style", "Money style", vec![PROFILE_APEX], vec![]),
            (Tree::Profile, "temperament", "Temperament", vec![PROFILE_APEX], vec![]),
        ];
        let owned = |ids: Vec<&str>| ids.into_iter().map(str::to_string).collect();
        for (tree, id, label, parents, routing) in crown {
            g.distillants.insert(id.to_string(), Distillant::bare(id, tree, label, owned(parents), owned(routing)));
        }
        g
    }

    /// Where a distillant hangs. Its parents become an antichain; in the
    /// profile tree they are never nowhere — an axis with no parent of its
    /// own is an axis of the apex, and the apex alone is a root.
    pub fn home_parents(&self, id: &str, tree: Tree, parents: Vec<Id>) -> Vec<Id> {
        let is_apex = id == PROFILE_APEX;
        if is_apex {
            return Vec::new();
        }
        let homes = self.antichain(parents);
        let axis_without_parent = tree == Tree::Profile && homes.is_empty();
        if axis_without_parent {
            return vec![PROFILE_APEX.to_string()];
        }
        homes
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

    /// A parent set names the finest homes only: deduplicated, no empties,
    /// and no parent that is an ancestor of another parent in the set — an
    /// ancestor beside its own descendant says nothing the descendant does
    /// not, and renders the node twice.
    pub fn antichain(&self, parents: Vec<Id>) -> Vec<Id> {
        let mut distinct: Vec<Id> = Vec::new();
        for p in parents {
            if !p.is_empty() && !distinct.contains(&p) {
                distinct.push(p);
            }
        }
        let is_ancestor_of_another =
            |p: &Id| distinct.iter().any(|q| q != p && self.is_ancestor(p, q));
        distinct.iter().filter(|p| !is_ancestor_of_another(p)).cloned().collect()
    }

    /// Forget by id — a leaf, or a distillant with everything under it.
    /// Forgetting is the user's authority over what is held about them, so
    /// the content leaves every store: the leaves go; the stream episodes
    /// that were evidence for nothing else go with them (an episode
    /// restating the content would keep it recallable); a forgotten
    /// distillant's children re-home to its parents and episode tags naming
    /// it are dropped. Every surviving distillant that held something
    /// removed is stamped `forgotten_at = now`, so its line reads as
    /// drifted until a pass rewrites it. None when nothing has that id.
    pub fn forget(&mut self, id: &str, now: u64) -> Option<Forgotten> {
        let distillant_ids = self.forgotten_distillant_ids(id);
        let forgets_a_distillant = !distillant_ids.is_empty();
        let leaf_ids: Vec<Id> = if forgets_a_distillant {
            self.leaves
                .values()
                .filter(|l| l.parents.iter().all(|p| distillant_ids.contains(p)))
                .map(|l| l.id.clone())
                .collect()
        } else if self.leaves.contains_key(id) {
            vec![id.to_string()]
        } else {
            return None;
        };

        let mut removed_leaves: Vec<Leaf> = Vec::new();
        for lid in &leaf_ids {
            if let Some(l) = self.leaves.remove(lid) {
                removed_leaves.push(l);
            }
        }
        let mut held_something_removed: Vec<Id> =
            removed_leaves.iter().flat_map(|l| l.parents.iter().cloned()).collect();
        // A leaf that hung under a forgotten distillant AND somewhere else
        // keeps living at its other homes.
        for l in self.leaves.values_mut() {
            l.parents.retain(|p| !distillant_ids.contains(p));
        }

        let mut new_homes: Vec<Id> = Vec::new();
        for did in &distillant_ids {
            if let Some(d) = self.distillants.remove(did) {
                new_homes.extend(d.parents);
            }
        }
        held_something_removed.extend(new_homes.iter().cloned());
        for pid in held_something_removed {
            if let Some(p) = self.distillants.get_mut(&pid) {
                p.forgotten_at = now;
            }
        }
        let child_ids: Vec<Id> = self
            .distillants
            .values()
            .filter(|d| d.parents.iter().any(|p| distillant_ids.contains(p)))
            .map(|d| d.id.clone())
            .collect();
        for cid in child_ids {
            let Some(child) = self.distillants.get(&cid) else { continue };
            let mut parents: Vec<Id> =
                child.parents.iter().filter(|p| !distillant_ids.contains(p)).cloned().collect();
            parents.extend(new_homes.iter().cloned());
            let parents = self.antichain(parents);
            if let Some(child) = self.distillants.get_mut(&cid) {
                child.parents = parents;
            }
        }

        let still_evidence: Vec<&Id> = self.leaves.values().flat_map(|l| l.evidence.iter()).collect();
        let sole_evidence: Vec<Id> = removed_leaves
            .iter()
            .flat_map(|l| l.evidence.iter())
            .filter(|e| !still_evidence.contains(e))
            .cloned()
            .collect();
        let tagged_only_here = |e: &Episode| {
            forgets_a_distillant && !e.tags.is_empty() && e.tags.iter().all(|t| distillant_ids.contains(t))
        };
        let mut episodes: Vec<Id> = Vec::new();
        self.episodes.retain(|e| {
            let goes = sole_evidence.contains(&e.id) || tagged_only_here(e);
            if goes {
                episodes.push(e.id.clone());
            }
            !goes
        });
        for e in &mut self.episodes {
            e.tags.retain(|t| !distillant_ids.contains(t));
        }

        Some(Forgotten {
            leaves: removed_leaves.into_iter().map(|l| l.id).collect(),
            distillants: distillant_ids,
            episodes,
        })
    }

    /// Restore the structural invariants over the whole graph: the profile
    /// tree has the apex as its one root with every axis under it, and
    /// every leaf's and distillant's parents form an antichain. Ops keep
    /// both as they go; this is for graphs written before they held.
    pub fn normalize(&mut self) -> Normalized {
        let mut out = Normalized::default();
        if !self.distillants.contains_key(PROFILE_APEX) {
            self.distillants.insert(
                PROFILE_APEX.to_string(),
                Distillant::bare(PROFILE_APEX, Tree::Profile, "Character", Vec::new(), Vec::new()),
            );
            out.apex_created = true;
        }
        let distillant_ids: Vec<Id> = self.distillants.keys().cloned().collect();
        for id in distillant_ids {
            let m = &self.distillants[&id];
            let parents = m.parents.clone();
            let homes = self.home_parents(&id, m.tree, parents.clone());
            if homes == parents {
                continue;
            }
            let homed_under_apex = parents.is_empty() && homes == [PROFILE_APEX];
            self.distillants.get_mut(&id).expect("listed above").parents = homes;
            if homed_under_apex {
                out.homed_under_apex.push(id);
            } else {
                out.antichained.push(id);
            }
        }
        let leaf_ids: Vec<Id> = self.leaves.keys().cloned().collect();
        for id in leaf_ids {
            let parents = self.leaves[&id].parents.clone();
            let homes = self.antichain(parents.clone());
            if homes != parents {
                self.leaves.get_mut(&id).expect("listed above").parents = homes;
                out.antichained.push(id);
            }
        }
        out
    }

    /// The distillant `id` and every distillant reachable below it (a
    /// forgotten distillant takes its subtree). Empty when `id` is no
    /// distillant.
    fn forgotten_distillant_ids(&self, id: &str) -> Vec<Id> {
        if !self.distillants.contains_key(id) {
            return Vec::new();
        }
        let mut out: Vec<Id> = vec![id.to_string()];
        let mut i = 0;
        while i < out.len() {
            for c in self.child_distillants(&out[i]) {
                if !out.contains(&c.id) {
                    out.push(c.id.clone());
                }
            }
            i += 1;
        }
        out
    }
}

/// What `Graph::normalize` repaired on a graph written before the current
/// invariants held. Empty on a graph that already holds them.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Normalized {
    /// Ids (leaves and distillants) whose parent set was reduced to an antichain.
    pub antichained: Vec<Id>,
    /// Profile distillants that stood as roots and now hang under the apex.
    pub homed_under_apex: Vec<Id>,
    /// Whether the apex itself was missing and had to be created.
    pub apex_created: bool,
}

impl Normalized {
    pub fn is_empty(&self) -> bool {
        self.antichained.is_empty() && self.homed_under_apex.is_empty() && !self.apex_created
    }
}

/// What one `Graph::forget` removed, by id — the trace names it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Forgotten {
    pub leaves: Vec<Id>,
    pub distillants: Vec<Id>,
    pub episodes: Vec<Id>,
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
        let profile_roots = g.roots(Tree::Profile);
        assert_eq!(profile_roots.len(), 1, "the apex is the profile's one root");
        assert_eq!(profile_roots[0].id, PROFILE_APEX);
        let mut axes: Vec<&str> = g.child_distillants(PROFILE_APEX).iter().map(|c| c.id.as_str()).collect();
        axes.sort();
        assert_eq!(axes, vec!["communication", "money-style", "temperament"]);
        assert!(g.distillants.values().all(|m| m.is_bare()), "every seed is born bare — nothing has judged yet");
    }

    #[test]
    fn a_line_is_a_judgment_only_once_something_stamped_it() {
        let mut g = Graph::seed();
        // The legacy seed shape: a description in the line slot, never stamped.
        g.distillants.get_mut("money-style").unwrap().line = "How they handle money: discipline, risk, planning.".into();
        assert!(g.distillants["money-style"].is_bare(), "an unstamped description is not a judgment");
        assert_eq!(g.distillants["money-style"].headline(), "Money style (no line yet)");
        g.distillants.get_mut("money-style").unwrap().line_changed_at = 7;
        assert!(!g.distillants["money-style"].is_bare(), "a stamped line is one");
        g.distillants.get_mut("money-style").unwrap().line = String::new();
        assert!(g.distillants["money-style"].is_bare(), "a stamped empty line still says nothing");
    }

    #[test]
    fn home_parents_keeps_the_profile_rooted_at_the_apex() {
        let g = Graph::seed();
        assert_eq!(g.home_parents("risk-appetite", Tree::Profile, vec![]), vec![PROFILE_APEX], "a parentless axis is an axis of the apex");
        assert_eq!(g.home_parents("communication/bluntness", Tree::Profile, vec!["communication".into()]), vec!["communication"]);
        assert_eq!(g.home_parents(PROFILE_APEX, Tree::Profile, vec!["temperament".into()]), Vec::<Id>::new(), "the apex is never re-homed");
        assert_eq!(g.home_parents("people", Tree::Registry, vec![]), Vec::<Id>::new(), "registry crowns stay roots");
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
                    forgotten_at: 0,
                },
            );
        }
        assert!(g.is_ancestor("life", "life/island/animals"));
        assert!(g.is_ancestor("life/island", "life/island/animals"));
        assert!(!g.is_ancestor("life/island/animals", "life"));
        assert!(!g.is_ancestor("money", "life/island"));
    }

    fn bare_distillant(g: &mut Graph, id: &str, parents: &[&str]) {
        let label = id.rsplit('/').next().unwrap_or(id).replace('-', " ");
        let parents = parents.iter().map(|p| p.to_string()).collect();
        g.distillants.insert(id.into(), Distillant::bare(id, Tree::Registry, &label, parents, Vec::new()));
    }

    fn leaf_with_evidence(g: &mut Graph, id: &str, parents: &[&str], evidence: &[&str]) {
        let mut l = Leaf::new(
            id.into(),
            Species::State,
            format!("fact {id}"),
            parents.iter().map(|p| p.to_string()).collect(),
            1,
        );
        l.evidence = evidence.iter().map(|e| e.to_string()).collect();
        g.leaves.insert(l.id.clone(), l);
    }

    #[test]
    fn antichain_drops_ancestors_duplicates_and_empties() {
        let mut g = Graph::seed();
        bare_distillant(&mut g, "work/video", &["work"]);
        bare_distillant(&mut g, "work/video/prompt", &["work/video"]);
        let homes = g.antichain(vec![
            "work".into(),
            "work/video".into(),
            "".into(),
            "work/video".into(),
            "people".into(),
        ]);
        assert_eq!(homes, vec!["work/video", "people"], "the ancestor `work` is implied by `work/video`");
        assert_eq!(g.antichain(vec!["work/video/prompt".into(), "work".into()]), vec!["work/video/prompt"]);
        assert_eq!(g.antichain(vec!["money".into()]), vec!["money"], "a single parent stands");
    }

    #[test]
    fn forget_leaf_takes_only_its_sole_evidence_episodes() {
        let mut g = Graph::seed();
        let shared = g.push_episode("ate lunch and paid rent".into(), vec!["life".into()], 1, None);
        let sole = g.push_episode("ate katsu curry".into(), vec!["life".into()], 2, None);
        let unrelated = g.push_episode("rent paid".into(), vec!["money".into()], 3, None);
        leaf_with_evidence(&mut g, "katsu", &["life"], &[&shared, &sole]);
        leaf_with_evidence(&mut g, "rent", &["money"], &[&shared, &unrelated]);

        let gone = g.forget("katsu", 9).expect("leaf exists");
        assert_eq!(g.distillants["life"].forgotten_at, 9, "the parent that held the leaf is stamped");
        assert_eq!(g.distillants["money"].forgotten_at, 0);
        assert_eq!(gone.leaves, vec!["katsu"]);
        assert!(gone.distillants.is_empty());
        assert_eq!(gone.episodes, vec![sole.clone()], "only the episode nothing else cites leaves the stream");
        assert!(!g.leaves.contains_key("katsu"));
        assert!(g.episodes.iter().any(|e| e.id == shared), "an episode still cited elsewhere stays");
        assert!(g.episodes.iter().all(|e| e.id != sole));
        assert_eq!(g.leaves["rent"].evidence, vec![shared, unrelated]);
    }

    #[test]
    fn forget_distillant_takes_subtree_rehomes_children_and_drops_tags() {
        let mut g = Graph::seed();
        bare_distillant(&mut g, "work/hack", &["work"]);
        bare_distillant(&mut g, "work/hack/inner", &["work/hack"]);
        bare_distillant(&mut g, "work/hack/kept-elsewhere", &["work/hack", "people"]);
        let only_here = g.push_episode("hack built".into(), vec!["work/hack".into()], 1, None);
        let also_money = g.push_episode("hack sold".into(), vec!["work/hack".into(), "money".into()], 2, None);
        leaf_with_evidence(&mut g, "inner-fact", &["work/hack/inner"], &[]);
        leaf_with_evidence(&mut g, "dual-home", &["work/hack", "money"], &[]);

        let gone = g.forget("work/hack", 9).expect("distillant exists");
        assert_eq!(g.distillants["work"].forgotten_at, 9, "the surviving parent of a forgotten distillant is stamped");
        assert_eq!(gone.distillants, vec!["work/hack", "work/hack/inner", "work/hack/kept-elsewhere"]);
        assert_eq!(gone.leaves, vec!["inner-fact"], "a leaf with another home survives");
        assert_eq!(gone.episodes, vec![only_here], "an episode tagged only inside the forgotten subtree goes");
        assert_eq!(g.leaves["dual-home"].parents, vec!["money"]);
        let sold = g.episodes.iter().find(|e| e.id == also_money).expect("shared-tag episode stays");
        assert_eq!(sold.tags, vec!["money"], "the forgotten tag is dropped");
        assert!(!g.distillants.contains_key("work/hack/kept-elsewhere"), "the subtree goes even where a child had a second parent");
    }

    #[test]
    fn forget_distillant_rehomes_grandchildren_to_the_parents() {
        let mut g = Graph::seed();
        bare_distillant(&mut g, "work/a", &["work"]);
        bare_distillant(&mut g, "work/a/b", &["work/a"]);
        // Forgetting only the middle node: its child climbs to `work`.
        let ids = g.forgotten_distillant_ids("work/a");
        assert_eq!(ids, vec!["work/a", "work/a/b"], "a forget takes the whole subtree, never just the middle");
        assert!(g.forget("nope", 9).is_none());
    }

    #[test]
    fn headline_marks_bare_distillants() {
        let mut g = Graph::seed();
        bare_distillant(&mut g, "work/x-report", &["work"]);
        assert!(g.distillants["work/x-report"].is_bare());
        assert_eq!(g.distillants["work/x-report"].headline(), "x report (no line yet)");
        let money = g.distillants.get_mut("money").unwrap();
        money.line = "Runs one wallet on a tight budget.".into();
        money.line_changed_at = 5;
        assert_eq!(g.distillants["money"].headline(), g.distillants["money"].line);
        // The legacy stub shape: a line that only repeats the label is no line.
        let stub = g.distillants.get_mut("work/x-report").unwrap();
        stub.line = "X Report".into();
        stub.line_changed_at = 5;
        assert!(g.distillants["work/x-report"].is_bare());
        g.distillants.get_mut("work/x-report").unwrap().line = "Runs the X report for friends.".into();
        assert!(!g.distillants["work/x-report"].is_bare());
    }

    #[test]
    fn normalize_repairs_a_graph_written_before_the_invariants() {
        let mut g = Graph::seed();
        bare_distillant(&mut g, "work/video", &["work"]);
        bare_distillant(&mut g, "work/video/prompt", &["work", "work/video"]);
        leaf_with_evidence(&mut g, "take", &["work/video/prompt", "work"], &[]);
        leaf_with_evidence(&mut g, "fine", &["money"], &[]);
        // The pre-apex profile shape: axes standing as roots, no apex at all.
        g.distillants.remove(PROFILE_APEX);
        for axis in ["communication", "money-style", "temperament"] {
            g.distillants.get_mut(axis).unwrap().parents.clear();
        }
        let repaired = g.normalize();
        assert_eq!(repaired.antichained, vec!["work/video/prompt", "take"]);
        assert!(repaired.apex_created);
        assert_eq!(repaired.homed_under_apex, vec!["communication", "money-style", "temperament"]);
        assert_eq!(g.distillants["work/video/prompt"].parents, vec!["work/video"]);
        assert_eq!(g.leaves["take"].parents, vec!["work/video/prompt"]);
        assert_eq!(g.roots(Tree::Profile).len(), 1);
        assert!(g.distillants[PROFILE_APEX].is_bare());
        assert!(g.normalize().is_empty(), "idempotent");
    }

    #[test]
    fn slugify_basic() {
        assert_eq!(slugify("My Sister Lisa"), "my-sister-lisa");
        assert_eq!(slugify("money/rent"), "money/rent");
        assert_eq!(slugify("  Hello,  World! "), "hello-world");
    }
}
