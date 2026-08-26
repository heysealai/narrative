//! Mechanical projection: the read-time hot path. No LLM here, ever.
//!
//! Write-time intelligence (the harvester) compiles aliases and keywords into
//! each distillant's routing map; this module matches an outgoing message
//! against that table lexically. Ambiguity is permissive — every matching
//! distillant activates; over-supply costs tokens, not correctness.

use std::collections::BTreeSet;

use crate::model::{Graph, Id};

pub struct RoutingTable {
    /// (normalized term, distillant id)
    entries: Vec<(String, Id)>,
}

/// Lowercase; every non-alphanumeric becomes a space ("lisa's" -> "lisa s").
pub fn normalize(text: &str) -> String {
    text.chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect()
}

/// Evaluative-facet vocabulary (rule 13's facets and the kin the discovery
/// run surfaced), grouped by inflection family. The line is the gate:
/// a single-word facet term may live in a distillant's routing only while
/// some word of its family appears in that distillant's line. This is
/// the mechanical edge of the prompt-level doctrine — enforced when routing
/// is added and re-checked whenever a line is rewritten (orphan
/// pruning), so prompt drift cannot leave facet routing unbacked.
pub const FACET_FAMILIES: &[&[&str]] = &[
    &["regret", "regrets", "regretted", "regretting"],
    &["trust", "trusts", "trusted", "trusting", "trusty"],
    &["fear", "fears", "feared", "fearful", "afraid", "dread", "dreads", "terrified", "terrifying"],
    &["pride", "proud", "proudly"],
    &["conflict", "conflicts", "conflicted"],
    &["loyal", "loyalty", "loyalist", "loyally"],
    &["faithful", "faithfulness", "faithfully"],
    &["devoted", "devotion"],
    &["grateful", "gratitude"],
    &["jealous", "jealousy"],
    &["shame", "ashamed", "shameful"],
    &["weep", "weeps", "weeping", "wept", "cried", "cries", "tears"],
    &["aversion", "aversions", "averse"],
    &["ache", "aches", "aching"],
];

/// The family of a single-word facet term; None for everything else.
/// Multi-word terms are referring expressions ("faithful boy" names a
/// person, not a facet) and are never gated.
pub fn facet_family(term: &str) -> Option<&'static [&'static str]> {
    let t = term.trim().to_lowercase();
    if t.contains(char::is_whitespace) {
        return None;
    }
    FACET_FAMILIES.iter().copied().find(|fam| fam.contains(&t.as_str()))
}

/// Does the line carry this facet family? Token-exact over the
/// normalized line; family breadth (fear/afraid/dread) is the synonym
/// bridge, not stemming.
pub fn distillant_carries(family: &[&str], line: &str) -> bool {
    let norm = normalize(line);
    let tokens: BTreeSet<&str> = norm.split_whitespace().collect();
    family.iter().any(|w| tokens.contains(w))
}

impl RoutingTable {
    pub fn build(graph: &Graph) -> Self {
        let mut entries = Vec::new();
        for m in graph.distillants.values() {
            let mut terms: BTreeSet<String> = BTreeSet::new();
            terms.insert(normalize(&m.label).trim().to_string());
            // The last path segment of the id is recall vocabulary too.
            if let Some(seg) = m.id.rsplit('/').next() {
                terms.insert(normalize(&seg.replace('-', " ")).trim().to_string());
            }
            for t in &m.routing {
                terms.insert(normalize(t).trim().to_string());
            }
            for t in terms {
                if !t.is_empty() {
                    entries.push((t, m.id.clone()));
                }
            }
        }
        RoutingTable { entries }
    }

    /// All distillants whose routing vocabulary appears in the message.
    /// Single-word terms match whole tokens; multi-word terms match as phrases.
    /// Ranked: strongest lexical match first.
    pub fn matches(&self, text: &str) -> Vec<Id> {
        self.matches_scored(text).into_iter().map(|(mid, _)| mid).collect()
    }

    /// Matches with their lexical score: each distinct single-word hit
    /// counts 1, each phrase hit 2 (a matched phrase is rarer, hence more
    /// specific). Sorted by score descending; ties keep table order (≈ id
    /// order) so the ranking is deterministic.
    pub fn matches_scored(&self, text: &str) -> Vec<(Id, u32)> {
        let norm = normalize(text);
        let tokens: BTreeSet<&str> = norm.split_whitespace().collect();
        let phrase_haystack = {
            let joined: Vec<&str> = norm.split_whitespace().collect();
            format!(" {} ", joined.join(" "))
        };
        let mut hits: Vec<(Id, u32)> = Vec::new();
        for (term, mid) in &self.entries {
            let weight = if term.contains(' ') {
                if phrase_haystack.contains(&format!(" {term} ")) { 2 } else { 0 }
            } else if tokens.contains(term.as_str()) {
                1
            } else {
                0
            };
            if weight == 0 {
                continue;
            }
            match hits.iter_mut().find(|(m, _)| m == mid) {
                Some((_, s)) => *s += weight,
                None => hits.push((mid.clone(), weight)),
            }
        }
        hits.sort_by(|(_, a), (_, b)| b.cmp(a)); // stable: ties keep table order
        hits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Graph, Distillant, Tree};

    fn graph_with(distillants: Vec<(&str, Tree, &str, Vec<&str>)>) -> Graph {
        let mut g = Graph::default();
        for (id, tree, label, routing) in distillants {
            g.distillants.insert(
                id.to_string(),
                Distillant {
                    id: id.to_string(),
                    tree,
                    label: label.to_string(),
                    line: String::new(),
                    routing: routing.into_iter().map(|s| s.to_string()).collect(),
                    parents: Vec::new(),
                    misc_count: 0,
                    consolidated_at: 0,
                    line_changed_at: 0,
                    forgotten_at: 0,
                },
            );
        }
        g
    }

    #[test]
    fn matches_alias_and_label() {
        let g = graph_with(vec![(
            "people/lisa",
            Tree::Registry,
            "Lisa",
            vec!["lisa.eth", "my sister"],
        )]);
        let t = RoutingTable::build(&g);
        assert_eq!(t.matches("send Lisa the usual"), vec!["people/lisa"]);
        assert_eq!(t.matches("pay my sister back"), vec!["people/lisa"]);
        assert_eq!(t.matches("transfer to lisa.eth now"), vec!["people/lisa"]);
        assert!(t.matches("totally unrelated message").is_empty());
    }

    #[test]
    fn possessive_punctuation_still_matches() {
        let g = graph_with(vec![("people/lisa", Tree::Registry, "Lisa", vec![])]);
        let t = RoutingTable::build(&g);
        assert_eq!(t.matches("what's lisa's wallet?"), vec!["people/lisa"]);
    }

    #[test]
    fn ambiguous_mention_activates_all_matches() {
        let g = graph_with(vec![
            ("people/lisa-sister", Tree::Registry, "Lisa (sister)", vec!["lisa"]),
            ("people/lisa-coworker", Tree::Registry, "Lisa (coworker)", vec!["lisa"]),
        ]);
        let t = RoutingTable::build(&g);
        let hits = t.matches("lunch with lisa tomorrow");
        assert_eq!(hits.len(), 2, "permissive: both Lisas activate");
    }

    #[test]
    fn single_word_terms_do_not_match_substrings() {
        let g = graph_with(vec![("money/rent", Tree::Registry, "Rent", vec!["rent"])]);
        let t = RoutingTable::build(&g);
        assert!(t.matches("the current situation").is_empty());
        assert_eq!(t.matches("rent is due"), vec!["money/rent"]);
    }

    #[test]
    fn id_segment_is_vocabulary() {
        let g = graph_with(vec![("money/dust-sweep", Tree::Registry, "Dust sweeping", vec![])]);
        let t = RoutingTable::build(&g);
        assert_eq!(t.matches("run the dust sweep"), vec!["money/dust-sweep"]);
    }

    #[test]
    fn matches_rank_by_lexical_score_with_phrases_weighing_double() {
        let g = graph_with(vec![
            ("money/aaa", Tree::Registry, "Aaa", vec!["rent"]),
            ("money/zzz", Tree::Registry, "Zzz", vec!["rent", "late fee"]),
        ]);
        let t = RoutingTable::build(&g);
        let scored = t.matches_scored("is the rent late fee due?");
        assert_eq!(scored[0], ("money/zzz".to_string(), 3), "1 token + 1 phrase×2");
        assert_eq!(scored[1], ("money/aaa".to_string(), 1));
        assert_eq!(t.matches("is the rent late fee due?"), vec!["money/zzz", "money/aaa"]);
        // Ties keep table order: deterministic, no reordering on equal score.
        assert_eq!(t.matches("rent is due"), vec!["money/aaa", "money/zzz"]);
    }

    #[test]
    fn facet_family_lookup_and_the_distillant_gate() {
        assert!(facet_family("regrets").is_some());
        assert!(facet_family("Regret").is_some(), "case-insensitive");
        assert!(facet_family("plantation").is_none(), "topical vocabulary is never gated");
        assert!(facet_family("faithful boy").is_none(), "phrases are referring expressions");

        let fear = facet_family("afraid").unwrap();
        assert!(
            distillant_carries(fear, "wolves in the night, the worst fear of his life"),
            "family breadth bridges synonyms: a fear line licenses 'afraid'"
        );
        assert!(!distillant_carries(fear, "swore faithfulness, proves loyal"));
    }
}
