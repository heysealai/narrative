//! Belief and salience arithmetic. Pure functions over countable events.
//!
//! The model classifies (supports / contradicts / supersedes); this module
//! moves the numbers. "Why did this belief move?" always has an answer like
//! "four contradicting episodes in sixty days" — never model vibes.

use crate::model::{Belief, Leaf, Nudge, Salience, Supersession};

/// Laplace-smoothed belief strength in (0, 1).
pub fn strength(b: &Belief) -> f32 {
    (b.support as f32 + 1.0) / ((b.support + b.contradict) as f32 + 2.0)
}

pub fn support(b: &mut Belief, at: u64) {
    b.support += 1;
    b.last_event_at = at;
}

pub fn contradict(b: &mut Belief, at: u64) {
    b.contradict += 1;
    b.last_event_at = at;
    b.last_against_at = at;
}

/// EMA step for a disposition axis. One outlier nudges, repeated patterns move —
/// hysteresis by inertia, the behavioral-median property applied to beliefs.
pub const NUDGE_ALPHA: f32 = 0.25;

pub fn nudge_axis(axis: f32, dir: i8) -> f32 {
    let target = dir.signum() as f32;
    (axis * (1.0 - NUDGE_ALPHA) + target * NUDGE_ALPHA).clamp(-1.0, 1.0)
}

/// The axis magnitude past which a position counts as committed to a pole.
pub const COMMIT: f32 = 0.3;

/// Crossing the commitment boundary into a pole. (Whether that crossing is a
/// *flip* depends on history — see `apply_nudge`.)
pub fn flipped(before: f32, after: f32) -> bool {
    (before > -COMMIT && after <= -COMMIT) || (before < COMMIT && after >= COMMIT)
}

/// Apply a nudge to a disposition leaf, recording the trajectory.
/// Returns true on a genuine polarity flip: the axis crossed the commitment
/// boundary against prior evidence — not a fresh leaf settling into its pole.
pub fn apply_nudge(leaf: &mut Leaf, dir: i8, note: String, at: u64) -> bool {
    let had_opposition = leaf.nudges.iter().any(|n| n.dir.signum() != dir.signum());
    let before = leaf.axis;
    leaf.axis = nudge_axis(leaf.axis, dir);
    leaf.nudges.push(Nudge { dir, note, at });
    support(&mut leaf.belief, at);
    leaf.updated_at = at;
    let flip = flipped(before, leaf.axis) && had_opposition;
    if flip {
        leaf.belief.last_against_at = at;
    }
    flip
}

/// States switch, they don't drift: the old value becomes history
/// (and the history of change is some of the most informative memory).
/// `event_at` is when the change actually happened (story time for backlog
/// imports); `now` is write time, which the arithmetic keys on.
pub fn supersede(leaf: &mut Leaf, new_text: String, event_at: u64, now: u64) {
    let old = std::mem::replace(&mut leaf.text, new_text);
    leaf.history.push(Supersession { value: old, superseded_at: event_at });
    // The fresh belief keeps the against-stamp: the VALUE moved, and any
    // line written over the old value is now suspect regardless of
    // how believed the new one is.
    leaf.belief =
        Belief { support: 1, contradict: 0, last_event_at: now, last_against_at: now };
    leaf.updated_at = now;
}

/// Salience: importance assigned at write, strengthened on retrieval,
/// decayed with disuse (half-life ~6 months). Defends exceptions from
/// median-washing — a high-importance outlier resists absorption.
pub fn effective_salience(s: &Salience, last_touch: u64, now: u64) -> f32 {
    let idle_days = now.saturating_sub(last_touch) as f32 / 86_400.0;
    let decay = 0.5_f32.powf(idle_days / 180.0);
    let reinforcement = (0.04 * s.retrieval_count as f32).min(0.2);
    (s.importance * decay + reinforcement).min(1.5)
}

fn last_touch(leaf: &Leaf) -> u64 {
    leaf.salience.last_retrieved_at.unwrap_or(0).max(leaf.updated_at)
}

/// Retrieval ranking score: relevance is handled by routing (you only score
/// leaves under an activated distillant); this combines recency, importance,
/// reinforcement, and belief strength.
pub fn score(leaf: &Leaf, now: u64) -> f32 {
    effective_salience(&leaf.salience, last_touch(leaf), now) * (0.5 + 0.5 * strength(&leaf.belief))
}

pub fn mark_retrieved(leaf: &mut Leaf, now: u64) {
    leaf.salience.retrieval_count += 1;
    leaf.salience.last_retrieved_at = Some(now);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Leaf, Species};

    fn leaf(species: Species) -> Leaf {
        Leaf::new("t".into(), species, "text".into(), vec!["money".into()], 1_000)
    }

    #[test]
    fn strength_is_laplace_smoothed() {
        let b = Belief { support: 1, contradict: 0, last_event_at: 0, last_against_at: 0 };
        assert!((strength(&b) - 2.0 / 3.0).abs() < 1e-6);
        let b = Belief { support: 4, contradict: 4, last_event_at: 0, last_against_at: 0 };
        assert!((strength(&b) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn one_nudge_does_not_flip() {
        let axis = 0.6;
        let after = nudge_axis(axis, -1);
        assert!(after > 0.0, "single contradicting nudge must not flip the axis");
        assert!(!flipped(axis, after));
    }

    #[test]
    fn repeated_nudges_eventually_flip() {
        let mut axis = 0.6;
        let mut flips = 0;
        for _ in 0..10 {
            let before = axis;
            axis = nudge_axis(axis, -1);
            if flipped(before, axis) {
                flips += 1;
            }
        }
        assert!(axis < -0.3, "ten consistent nudges should move the axis across");
        assert_eq!(flips, 1, "the flip should be reported exactly once");
    }

    #[test]
    fn supersede_keeps_history_and_resets_belief() {
        let mut l = leaf(Species::State);
        l.belief.support = 7;
        supersede(&mut l, "rent is $2,400/mo".into(), 2_000, 2_000);
        assert_eq!(l.text, "rent is $2,400/mo");
        assert_eq!(l.history.len(), 1);
        assert_eq!(l.history[0].value, "text");
        assert_eq!(l.belief.support, 1);
    }

    #[test]
    fn supersede_event_time_lands_in_history_not_arithmetic() {
        let mut l = leaf(Species::State);
        supersede(&mut l, "moved to the cave".into(), 500, 9_000);
        assert_eq!(l.history[0].superseded_at, 500, "display time is event time");
        assert_eq!(l.updated_at, 9_000, "salience freshness stays on write time");
        assert_eq!(l.belief.last_event_at, 9_000);
    }

    #[test]
    fn salience_decays_with_disuse_and_reinforces_with_retrieval() {
        let s = Salience { importance: 0.8, retrieval_count: 0, last_retrieved_at: None };
        let fresh = effective_salience(&s, 1_000_000, 1_000_000);
        let year_idle = effective_salience(&s, 1_000_000, 1_000_000 + 365 * 86_400);
        assert!(year_idle < fresh * 0.4);

        let reinforced = Salience { importance: 0.8, retrieval_count: 5, last_retrieved_at: None };
        assert!(
            effective_salience(&reinforced, 1_000_000, 1_000_000)
                > effective_salience(&s, 1_000_000, 1_000_000)
        );
    }
}
