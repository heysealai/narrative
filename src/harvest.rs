//! The post-turn harvester: ambient writes. An off-context model call reads
//! the finished turn and emits structured memory ops; the runtime applies
//! them. Contradiction cross-matching happens here, against comparanda
//! pre-opened by projection pointed backwards at the turn text.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::belief;
use crate::llm::{ChatRequest, Llm};
use crate::model::{age_str, Belief, Distillant, Graph, Leaf, LeafKind, Nudge, Salience, Species, Supersession, Tree, PROFILE_APEX};
use crate::projection;
use crate::routing;

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "snake_case")]
pub enum Relation {
    Novel,
    Duplicate,
    Supports,
    Contradicts,
    Supersedes,
}

/// How a rules entry relates to the standing rules: a new instruction, a
/// restatement of one, a change of one's wording, or its withdrawal. Any
/// of the first three on a withdrawn rule's id reinstates it.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "snake_case")]
pub enum RuleRelation {
    Novel,
    Duplicate,
    Supersedes,
    Retract,
}

/// One rules entry: the user's instruction in their own words. `target` is
/// the standing rule this changes ("" when novel); `instruction` is the
/// host's instruction id this answers ("" when the rule came from ordinary
/// speech).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RuleOp {
    pub id: String,
    pub text: String,
    pub relation: RuleRelation,
    pub target: String,
    #[serde(default)]
    pub instruction: String,
    #[serde(default)]
    pub occurred_at: Option<u64>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Op {
    /// Stream: something that happened, time-anchored, immutable.
    Episode { text: String, tags: Vec<String>, occurred_at: Option<u64> },
    /// Registry: a state fact — current value with supersession semantics.
    State {
        id: String,
        distillants: Vec<String>,
        text: String,
        relation: Relation,
        /// Existing leaf id this relates to ("" when relation is novel).
        target: String,
        importance: f32,
        /// New recall vocabulary observed this turn ("my sister", "lisa.eth").
        aliases: Vec<String>,
        /// Event time: when the fact became true, when that differs from now.
        occurred_at: Option<u64>,
    },
    /// Profile: a disposition nudge along an axis.
    Disposition {
        id: String,
        distillants: Vec<String>,
        text: String,
        /// -1 or 1: which pole this observation pushes toward.
        dir: i8,
        note: String,
        importance: f32,
    },
    /// Rules: a standing instruction — binding on first occurrence, in the
    /// user's own words; a later instruction supersedes or retracts it.
    Rule(RuleOp),
    /// Create or refresh a distillant (the line layer).
    Distillant {
        id: String,
        tree: Tree,
        label: String,
        line: String,
        routing: Vec<String>,
        parents: Vec<String>,
    },
    /// New supporting evidence for an existing leaf, nothing to restate.
    Reinforce { target: String },
    /// Add recall vocabulary to an existing distillant.
    Alias { distillant: String, add: Vec<String> },
    /// Rewrite a distillant's one-line line.
    Distill { distillant: String, line: String },
    /// Re-home a leaf: replace its parent set (e.g. under a finer distillant).
    Move { leaf: String, parents: Vec<String> },
    /// Re-home a distillant under different parents (empty = promote to root).
    Reparent { distillant: String, parents: Vec<String> },
    /// Absorb a duplicate leaf into the canonical one; evidence and counts combine.
    MergeLeaf { from: String, into: String },
    /// Merge two distillants that turned out to be the same thing.
    MergeDistillant { from: String, into: String },
    /// The user asked to forget something: the leaf or distillant that holds
    /// it leaves every store (`Graph::forget`). Authoritative — applied
    /// after everything else in the batch, so nothing the same harvest
    /// wrote about the content survives it.
    Forget { target: String },
    /// Field manual: one durable operating lesson keyed by the tool and
    /// service it applies to, distilled from server-authored run-digest
    /// blocks in the input. HOST-APPLIED: the driver stores these as typed
    /// rows outside the graph; `apply_ops` touches nothing for them.
    ManualUpsert { tool: String, service: String, lesson: String },
    /// Retire a field-manual entry whose lesson the evidence now
    /// contradicts. Host-applied like [`Op::ManualUpsert`].
    ManualRetire { tool: String, service: String },
}

/// Wire format for the harvester's structured output. Named homogeneous
/// sections rather than a tagged union: grammar-constrained decoding
/// degenerates on `anyOf` discriminators (observed live: a single empty
/// episode), while homogeneous arrays keep the model filling each section
/// deliberately. Section order in the schema is also the apply order.
/// Sections may be omitted in hand-composed ops files; unknown keys are
/// rejected (catches the old tagged-union shape and decoding soup).
#[derive(Deserialize, Debug, Default)]
#[serde(default, deny_unknown_fields)]
struct HarvestOut {
    distillants: Vec<DistillantOp>,
    episodes: Vec<EpisodeOp>,
    states: Vec<StateOp>,
    dispositions: Vec<DispositionOp>,
    rules: Vec<RuleOp>,
    reinforces: Vec<ReinforceOp>,
    aliases: Vec<AliasOp>,
    distills: Vec<DistillOp>,
    moves: Vec<MoveOp>,
    reparents: Vec<ReparentOp>,
    merge_leaves: Vec<MergeOp>,
    merge_distillants: Vec<MergeOp>,
    forgets: Vec<ForgetOp>,
    manual_upserts: Vec<ManualUpsertOp>,
    manual_retires: Vec<ManualRetireOp>,
}

#[derive(Deserialize, Debug)]
struct EpisodeOp {
    text: String,
    tags: Vec<String>,
    #[serde(default)]
    occurred_at: Option<u64>,
}

#[derive(Deserialize, Debug)]
struct StateOp {
    id: String,
    distillants: Vec<String>,
    text: String,
    relation: Relation,
    target: String,
    importance: f32,
    aliases: Vec<String>,
    #[serde(default)]
    occurred_at: Option<u64>,
}

#[derive(Deserialize, Debug)]
struct DispositionOp {
    id: String,
    distillants: Vec<String>,
    text: String,
    dir: i8,
    note: String,
    importance: f32,
}

#[derive(Deserialize, Debug)]
struct DistillantOp {
    id: String,
    tree: Tree,
    label: String,
    line: String,
    routing: Vec<String>,
    parents: Vec<String>,
}

#[derive(Deserialize, Debug)]
struct ReinforceOp {
    target: String,
}

#[derive(Deserialize, Debug)]
struct AliasOp {
    distillant: String,
    add: Vec<String>,
}

#[derive(Deserialize, Debug)]
struct DistillOp {
    distillant: String,
    line: String,
}

#[derive(Deserialize, Debug)]
struct MoveOp {
    leaf: String,
    parents: Vec<String>,
}

#[derive(Deserialize, Debug)]
struct ReparentOp {
    distillant: String,
    parents: Vec<String>,
}

#[derive(Deserialize, Debug)]
struct MergeOp {
    from: String,
    into: String,
}

#[derive(Deserialize, Debug)]
struct ForgetOp {
    target: String,
}

#[derive(Deserialize, Debug)]
struct ManualUpsertOp {
    tool: String,
    service: String,
    lesson: String,
}

#[derive(Deserialize, Debug)]
struct ManualRetireOp {
    tool: String,
    service: String,
}

impl HarvestOut {
    fn into_ops(self) -> Vec<Op> {
        let mut ops = Vec::new();
        for m in self.distillants {
            ops.push(Op::Distillant {
                id: m.id,
                tree: m.tree,
                label: m.label,
                line: m.line,
                routing: m.routing,
                parents: m.parents,
            });
        }
        for e in self.episodes {
            if !e.text.trim().is_empty() {
                ops.push(Op::Episode { text: e.text, tags: e.tags, occurred_at: e.occurred_at });
            }
        }
        for s in self.states {
            ops.push(Op::State {
                id: s.id,
                distillants: s.distillants,
                text: s.text,
                relation: s.relation,
                target: s.target,
                importance: s.importance,
                aliases: s.aliases,
                occurred_at: s.occurred_at,
            });
        }
        for d in self.dispositions {
            ops.push(Op::Disposition {
                id: d.id,
                distillants: d.distillants,
                text: d.text,
                dir: d.dir,
                note: d.note,
                importance: d.importance,
            });
        }
        for r in self.rules {
            ops.push(Op::Rule(r));
        }
        for r in self.reinforces {
            ops.push(Op::Reinforce { target: r.target });
        }
        for a in self.aliases {
            ops.push(Op::Alias { distillant: a.distillant, add: a.add });
        }
        for d in self.distills {
            ops.push(Op::Distill { distillant: d.distillant, line: d.line });
        }
        for m in self.moves {
            ops.push(Op::Move { leaf: m.leaf, parents: m.parents });
        }
        for r in self.reparents {
            ops.push(Op::Reparent { distillant: r.distillant, parents: r.parents });
        }
        for m in self.merge_leaves {
            ops.push(Op::MergeLeaf { from: m.from, into: m.into });
        }
        for m in self.merge_distillants {
            ops.push(Op::MergeDistillant { from: m.from, into: m.into });
        }
        for f in self.forgets {
            ops.push(Op::Forget { target: f.target });
        }
        for m in self.manual_upserts {
            if !m.lesson.trim().is_empty() {
                ops.push(Op::ManualUpsert { tool: m.tool, service: m.service, lesson: m.lesson });
            }
        }
        for m in self.manual_retires {
            ops.push(Op::ManualRetire { tool: m.tool, service: m.service });
        }
        ops
    }
}

pub fn ops_schema() -> Value {
    let relation = json!({"type": "string", "enum": ["novel", "duplicate", "supports", "contradicts", "supersedes"]});
    let rule_relation = json!({"type": "string", "enum": ["novel", "duplicate", "supersedes", "retract"]});
    let tree = json!({"type": "string", "enum": ["registry", "profile"]});
    let strings = json!({"type": "array", "items": {"type": "string"}});
    json!({
        "type": "object",
        "properties": {
            "distillants": {
                "description": "New or refreshed distillants. Applied before leaves; declare a distillant here before hanging leaves under it.",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "path-like kebab id under a crown root, e.g. people/lisa or money/rent"},
                        "tree": tree,
                        "label": {"type": "string"},
                        "line": {"type": "string", "description": "one line: the behavioral median of what lives here; advertise evaluative facets the leaves carry (trust, regret, fear, pride) — retrieval descends by this line"},
                        "routing": {"type": "array", "items": {"type": "string"}, "description": "words/short phrases likely to appear in future messages about this — referring expressions and topics; evaluative vocabulary is derived from the line automatically"},
                        "parents": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["id", "tree", "label", "line", "routing", "parents"],
                    "additionalProperties": false
                }
            },
            "episodes": {
                "description": "Events worth remembering as events: time-anchored, past tense, one sentence.",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "involved distillant ids"},
                        "occurred_at": {"type": ["integer", "null"], "description": "unix seconds when the event actually happened, when the turn says so ('last month', a date, imported backlog); null = it happened now"}
                    },
                    "required": ["text", "tags", "occurred_at"],
                    "additionalProperties": false
                }
            },
            "states": {
                "description": "Registry facts: noun-shaped, current-value, supersession semantics.",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "kebab-case slug, stable handle"},
                        "distillants": {"type": "array", "items": {"type": "string"}, "description": "distillant ids this fact hangs under (multi-parent allowed)"},
                        "text": {"type": "string", "description": "the fact, one small atomic sentence"},
                        "relation": relation,
                        "target": {"type": "string", "description": "existing leaf id this relates to; empty string when novel"},
                        "importance": {"type": "number", "description": "0..1; 0.9+ for money rules and critical facts"},
                        "aliases": {"type": "array", "items": {"type": "string"}, "description": "ways the user refers to the THING this fact is about (names, nicknames, handles, addresses), added to the first distillant's routing — never words lifted from the fact itself"},
                        "occurred_at": {"type": ["integer", "null"], "description": "unix seconds when the fact became true / the change happened, when stated; null = now"}
                    },
                    "required": ["id", "distillants", "text", "relation", "target", "importance", "aliases", "occurred_at"],
                    "additionalProperties": false
                }
            },
            "dispositions": {
                "description": "Profile axis nudges: traits drift, one observation never flips an axis.",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "the EXISTING axis id from the pinned profile when the observation is about a tendency already tracked (that is a nudge); a new kebab-case id only for a genuinely new axis"},
                        "distillants": {"type": "array", "items": {"type": "string"}},
                        "text": {"type": "string", "description": "the axis statement, e.g. \"keeps spending tightly controlled\""},
                        "dir": {"type": "integer", "enum": [-1, 1], "description": "+1 pushes toward the statement, -1 against it"},
                        "note": {"type": "string", "description": "the observation behind the nudge"},
                        "importance": {"type": "number"}
                    },
                    "required": ["id", "distillants", "text", "dir", "note", "importance"],
                    "additionalProperties": false
                }
            },
            "rules": {
                "description": "Standing instructions: what the user asked for that should hold on every future turn, in their own words. Binding on first occurrence; never weighed or merged. An instruction that changes a standing rule shown to you supersedes or retracts it by id — never adds a second; one that gives a withdrawn rule again names that rule's id and reinstates it.",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "kebab-case slug, stable handle; the existing rule's id when relation is supersedes, duplicate or retract, and the withdrawn rule's id when the user gives one again"},
                        "text": {"type": "string", "description": "the instruction in the user's own words, trimmed to the instruction, present tense; a procedure keeps its trigger and every step, in order; empty for retract"},
                        "relation": rule_relation,
                        "target": {"type": "string", "description": "existing rule id this relates to; empty string when novel"},
                        "instruction": {"type": "string", "description": "the id from the 'Instructions to resolve' section this entry answers; empty string when the rule came from ordinary speech"},
                        "occurred_at": {"type": ["integer", "null"], "description": "unix seconds when the instruction was given, when the turn says so; null = now"}
                    },
                    "required": ["id", "text", "relation", "target", "instruction", "occurred_at"],
                    "additionalProperties": false
                }
            },
            "reinforces": {
                "description": "New supporting evidence for an existing leaf with nothing to restate — lighter than a supports state.",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "existing leaf id the evidence supports"}
                    },
                    "required": ["target"],
                    "additionalProperties": false
                }
            },
            "aliases": {
                "description": "Extra recall vocabulary for existing distillants.",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "distillant": {"type": "string"},
                        "add": strings
                    },
                    "required": ["distillant", "add"],
                    "additionalProperties": false
                }
            },
            "distills": {
                "description": "Refresh a distillant's one-line line when it has gone stale — including when this turn adds an evaluative facet (trust, regret, fear, pride) the old line does not advertise.",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "distillant": {"type": "string"},
                        "line": {"type": "string"}
                    },
                    "required": ["distillant", "line"],
                    "additionalProperties": false
                }
            },
            "moves": {
                "description": "Re-home leaves: replace a leaf's parent distillants, e.g. move old leaves under a finer distillant created this batch.",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "leaf": {"type": "string", "description": "existing leaf id"},
                        "parents": {"type": "array", "items": {"type": "string"}, "description": "the leaf's new full parent set"}
                    },
                    "required": ["leaf", "parents"],
                    "additionalProperties": false
                }
            },
            "reparents": {
                "description": "Re-home a distillant under different parent distillants.",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "distillant": {"type": "string"},
                        "parents": {"type": "array", "items": {"type": "string"}, "description": "new parent distillant ids; empty promotes to crown root"}
                    },
                    "required": ["distillant", "parents"],
                    "additionalProperties": false
                }
            },
            "merge_leaves": {
                "description": "Absorb a duplicate leaf into the canonical one: evidence, belief counts, and parents combine; 'from' is deleted.",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "from": {"type": "string", "description": "leaf id to absorb (deleted)"},
                        "into": {"type": "string", "description": "leaf id that remains"}
                    },
                    "required": ["from", "into"],
                    "additionalProperties": false
                }
            },
            "forgets": {
                "description": "The user explicitly asked to forget, delete, or stop remembering something: the ids of the leaves or distillants that hold it (they appear in the comparanda or the directory). Authoritative and applied last.",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "existing leaf or distillant id"}
                    },
                    "required": ["target"],
                    "additionalProperties": false
                }
            },
            "manual_upserts": {
                "description": "Field-manual lessons, distilled ONLY from worker run digest blocks in the input — never from the user's own words. One durable operating lesson per (tool, service): what wall exists and what works instead. An upsert REPLACES the existing entry for its (tool, service) shown in the field-manual section — refine it with the new evidence, never restate it. A one-off transient failure with nothing durable to teach emits nothing.",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool": {"type": "string", "description": "the tool name exactly as the digest line names it"},
                        "service": {"type": "string", "description": "the digest line's service anchor (a URL host, a service id); empty string for a tool-wide lesson"},
                        "lesson": {"type": "string", "description": "one or two sentences of plain operating prose: the wall and the working alternative"}
                    },
                    "required": ["tool", "service", "lesson"],
                    "additionalProperties": false
                }
            },
            "manual_retires": {
                "description": "Retire a field-manual entry whose lesson the input's evidence now contradicts (what the entry warns about demonstrably works).",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool": {"type": "string"},
                        "service": {"type": "string", "description": "the entry's service key; empty string for a tool-wide entry"}
                    },
                    "required": ["tool", "service"],
                    "additionalProperties": false
                }
            },
            "merge_distillants": {
                "description": "Merge two distillants that turned out to be the same thing: leaves, children, and routing move to 'into'; 'from' is deleted.",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "from": {"type": "string", "description": "distillant id to absorb (deleted)"},
                        "into": {"type": "string", "description": "distillant id that remains"}
                    },
                    "required": ["from", "into"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["distillants", "episodes", "states", "dispositions", "rules", "reinforces", "aliases", "distills", "moves", "reparents", "merge_leaves", "merge_distillants", "forgets", "manual_upserts", "manual_retires"],
        "additionalProperties": false
    })
}

const HARVESTER_SYSTEM: &str = r#"You are the memory architect for a personal assistant. After each conversation turn you distill what is durable into a structured memory with three stores and a list of standing rules:

- STREAM (episodes): things that happened — time-anchored, immutable. "Paid rent late in May."
- REGISTRY (states): noun-shaped facts with a current value — "rent is $2,200/mo", "sister = Lisa". States SWITCH: when a value changes, it is superseded, never blended.
- PROFILE (dispositions): trait axes that DRIFT — "tends to overspend late-month". One observation nudges an axis; it never flips it.
- RULES: the user's standing instructions in their own words — "keep replies to five lines", "ask before any spend over $20". A rule BINDS from the moment it is given: it is never weighed, nudged, or merged; only a later instruction supersedes, retracts, or reinstates it.

Rules:
1. Distill only what is durable. Chitchat, pleasantries, and one-off questions leave no trace. All sections empty is a perfectly good harvest.
2. One fact per leaf. Small, boring, atomic sentences. Never bundle.
3. Distillant creation is your judgment call: a durable participant in the user's life (a person, an obligation, a project) deserves a distillant (e.g. people/lisa, money/rent); an incidental mention stays a tag on an episode. Distillant ids are path-like, under the existing crown roots. Declare new distillants in the distillants section; they are applied before leaves. The user themself is never a distillant: the whole graph already models them — their history, doings, and possessions live under the topical crowns, their tendencies in the profile. people/* is for the OTHER people in their life.
4. Aliases are how future recall works: record the ways the user refers to a THING ("my sister", "lisa.eth", "the landlord") as routing vocabulary — on the state's aliases field or in the aliases section. Routing is a referring-expression index, not a word list: never lift words out of the fact itself, episode detail (numbers, one-off phrasings), or generic phrases a message about anything could contain — every stray term routes unrelated messages here. Single-user memory: possessives like "my sister" are stable aliases. Wallet addresses, handles and emails are exact anchors — always record them. Matching is exact-token, no stemming: include the inflected forms a future message would actually contain ("payment" AND "payments"). Evaluative vocabulary (trust, regret, fear, pride, conflict) is derived from lines automatically — spend aliases and routing on referring expressions, never on facet words.
5. Cross-match against the comparanda you are given. Classify each state: novel (nothing like it exists), duplicate (already stored, restated), supports (new evidence for an existing leaf — set target), contradicts (casts doubt, no clear replacement — set target), supersedes (clear new value replacing an old one — set target). Never store the same knowledge twice as novel. When the turn merely adds evidence for an existing leaf and there is nothing to restate, emit a reinforces entry instead of a supports state.
6. You classify; the runtime does the arithmetic. Never hedge text with probabilities.
7. importance: 0.9+ safety-critical facts, money facts, explicit "remember this"; ~0.5 ordinary facts; ~0.2 minor color. High-importance exceptions ("got scammed by X once") deserve their own leaf — never average them away.
8. Dispositions: the leaf text states the +1 pole of the axis. dir=+1 pushes toward the statement, dir=-1 against it. The whole profile is shown to you every time (pinned): an observation about a tendency already tracked is a nudge on that existing id — a new leaf only for a genuinely new axis. Keep profile axes few and broad. The profile's root, `character`, is the whole-person estimate consolidation distills from the axes: never hang a leaf there — a disposition always belongs to an axis under it.
9. Episodes: log events worth remembering as events (payments, decisions, incidents, plans made). Tag with involved distillant ids. The leaf ops you emit alongside will be wired to them as evidence automatically.
10. Use distill to refresh a distillant's one-line summary when what you learned makes the old line stale.
11. Time: everything is stamped with write time automatically. When the turn says WHEN something actually happened or changed ("last month", "back in 2019", dated backlog text), set occurred_at to unix seconds; otherwise null. Recall renders ages from it — "changed 2mo ago" should mean two months of the user's life, not two months since you wrote it.
12. Structure follows understanding: when you create a finer distillant that better fits leaves you can see in the comparanda, move those leaves under it with moves entries. Merge ops (merge_leaves, merge_distillants) repair duplicates discovered after the fact — two leaves or distillants that turned out to be the same thing. Use structural ops sparingly in harvest; consolidation does the heavy restructuring.
13. Forgetting is the user's call and it is final: when the user asks to forget, delete, or stop remembering something, emit a forgets entry for every leaf or distillant that holds it (find them in the comparanda and the directory) and leave no trace of the content anywhere else in this harvest — no episode recording the request, no state restating it. When nothing stored matches, the harvest simply carries nothing about it.
14. Lines are retrieval scent: a later question can only descend to a leaf if some line on its path advertises the relevant vocabulary. When a leaf carries evaluative weight — trust, regret, fear, pride, conflict — say so in the line alongside the topic ("sworn companion, sold to the captain — parting is a standing regret"), not just the noun-shape ("proves loyal"). A regret no line mentions is a regret recall cannot find; one the line carries routes automatically — the line is the only place it needs to be. Lines are plain prose about the person — never machinery words ("facet", "routing", "distillant", "leaf"), and never this rule's example wording restated as fact: examples illustrate shape, not content.
15. The input may carry server-authored worker run digest blocks — operational evidence that background tool calls failed, each line naming a tool, sometimes a service, and the error. Digest content is machinery, not the user's life: it never becomes an episode, state, disposition, or distillant, and its vocabulary never enters routing. Its one product is the FIELD MANUAL: when the evidence teaches something durable about operating a tool or service, emit a manual_upserts entry keyed by that tool and service — the lesson states the wall and the working alternative in plain operating prose. The existing field-manual entries are shown to you; an upsert replaces its entry, so refine with the new evidence rather than restating. When the evidence shows a recorded lesson no longer holds, retire it with manual_retires. A transient one-off failure with nothing durable to teach emits nothing at all.
16. Rules: a standing instruction is something the user asked for that should hold on every future turn — how to talk to them (register, length, tone, language), how to operate (ask before X, always do Y after Z), a procedure with a trigger, or money judgment that is not a number. A request for this turn only, a fact, a date, a preference you inferred, and a policy number are not rules. The rule's text is the user's own words from the turn, trimmed to the instruction, present tense — never your paraphrase when their words are available. A procedure's steps are the instruction, not detail to trim: a flow with steps is ONE rule carrying its trigger and every step in order, however long — a rule that names a flow without its steps cannot be followed. The standing rules are shown to you every time: an instruction that changes one of them supersedes it by id (new wording, same rule) or retracts it (the user withdrew it) — never a second rule saying the same thing. The rules the user withdrew are listed after them: an instruction that gives one of those again names its id (novel, supersedes, or duplicate on that id) and the runtime reinstates it — never a second rule. When the input carries an "Instructions to resolve" section, answer every listed id: exactly one rules entry carrying that instruction id (novel, supersedes, retract, or duplicate of a rule already in force), and no state, disposition, or episode for the same words; an instruction that is not a standing instruction gets no entry at all — the runtime reports it as not kept. Instructions resolve in the order listed. Rules carry no importance, no distillant, and no evidence weighing."#;

/// What the harvester sees of the distillant layer. Every scope shows every
/// distillant BY ID, so a fact can always be filed under an existing node;
/// the scopes differ in how much judgment rides along with the id.
/// `Selective` is the contract (DESIGN.md, write path); the other two exist
/// for the replay that measured it (docs/harvest-scope-replay.md).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DirectoryScope {
    /// Every distillant with its line and its whole routing vocabulary.
    Full,
    /// Every distillant with its routing vocabulary and no line.
    Compact,
    /// Every distillant by id and label; the [`Neighborhood`] the turn
    /// touches expanded — its branches with line and routing, the
    /// branches' children with their line.
    Selective,
}

/// The distillants a turn touches, read leaf to root. `on_branch`: the
/// routing hits and every ancestor up to the root — the branches the turn
/// is on, where the harvester files and whose routing it extends. `beside`:
/// every hit's children — the level the turn is on, the siblings a new
/// fact lands among, which the harvester recognizes by what they say (a
/// child named only by its label is a twin waiting to happen). Every hit
/// counts, a hit that another hit sits under included: the turn is on that
/// level too, and a sibling of the deeper hit is where its fact may belong.
/// A distillant in neither set is an index line.
#[derive(Default, Debug, PartialEq, Eq)]
pub struct Neighborhood {
    pub on_branch: BTreeSet<String>,
    pub beside: BTreeSet<String>,
}

impl Neighborhood {
    pub fn open(graph: &Graph, table: &routing::RoutingTable, turn_text: &str) -> Neighborhood {
        let hits = table.matches(turn_text);
        let mut on_branch = BTreeSet::new();
        let mut pending = hits.clone();
        while let Some(id) = pending.pop() {
            let Some(m) = graph.distillants.get(&id) else { continue };
            if !on_branch.insert(id) {
                continue;
            }
            pending.extend(m.parents.iter().cloned());
        }
        let beside = hits
            .iter()
            .flat_map(|hit| graph.child_distillants(hit))
            .map(|child| child.id.clone())
            .filter(|id| !on_branch.contains(id))
            .collect();
        Neighborhood { on_branch, beside }
    }
}

impl DirectoryScope {
    pub fn parse(spelling: &str) -> Option<Self> {
        match spelling {
            "full" => Some(Self::Full),
            "compact" => Some(Self::Compact),
            "selective" => Some(Self::Selective),
            _ => None,
        }
    }
}

/// One directory entry: the id, tree and label always; the line and the
/// routing vocabulary when the scope grants them.
fn directory_entry(m: &Distillant, with_line: bool, with_routing: bool) -> String {
    let tree = match m.tree {
        Tree::Registry => "registry",
        Tree::Profile => "profile",
    };
    let mut entry = format!("- {} [{}] \"{}\"", m.id, tree, m.label);
    if with_line {
        let _ = write!(entry, " — {}", m.headline());
    }
    if with_routing {
        let _ = write!(entry, " | routing: {}", m.routing.join(", "));
    }
    entry.push('\n');
    entry
}

fn render_directory_scoped(
    graph: &Graph,
    scope: DirectoryScope,
    table: &routing::RoutingTable,
    turn_text: &str,
) -> String {
    let neighborhood = match scope {
        DirectoryScope::Full | DirectoryScope::Compact => Neighborhood::default(),
        DirectoryScope::Selective => Neighborhood::open(graph, table, turn_text),
    };
    let mut out = String::new();
    for m in graph.distillants.values() {
        let (with_line, with_routing) = match scope {
            DirectoryScope::Full => (true, true),
            DirectoryScope::Compact => (false, true),
            DirectoryScope::Selective => {
                let on_branch = neighborhood.on_branch.contains(&m.id);
                let beside = neighborhood.beside.contains(&m.id);
                (on_branch || beside, on_branch)
            }
        };
        out.push_str(&directory_entry(m, with_line, with_routing));
    }
    out
}

/// The harvester's cross-match window is wider than the agent's recall caps:
/// a hidden comparandum makes the harvester store duplicates as novel.
const HARVEST_PER_MIDPOINT_CAP: usize = 12;
const HARVEST_TOTAL_CAP: usize = 48;

/// The standing rules, every one by id, for the harvester to supersede or
/// retract by id rather than mint a second rule saying the same thing —
/// pinned for the harvester as the profile is. The withdrawn rules follow
/// by id, so one the user gives again is reinstated rather than minted.
fn render_pinned_rules(graph: &Graph) -> String {
    fn by_id(mut rules: Vec<&Leaf>) -> Vec<&Leaf> {
        rules.sort_by(|a, b| a.id.cmp(&b.id));
        rules
    }
    let mut out = String::new();
    for l in by_id(graph.rules()) {
        let _ = writeln!(out, "- [{}] {}", l.id, l.text);
    }
    if out.is_empty() {
        out.push_str("(none)\n");
    }
    let withdrawn = by_id(graph.withdrawn_rules());
    if !withdrawn.is_empty() {
        out.push_str("Withdrawn (out of force; an instruction giving one of these again names its id and reinstates it):\n");
        for l in withdrawn {
            let _ = writeln!(out, "- [{}] {}", l.id, l.text);
        }
    }
    out
}

/// The host's instructions awaiting memory, each by id with the user's
/// words and the assistant's reading of them.
fn render_instructions(instructions: &[Instruction]) -> String {
    let mut out = String::new();
    for i in instructions {
        let _ = writeln!(out, "- [{}] user said: \"{}\" — read as: {}", i.id, i.utterance.trim(), i.paraphrase.trim());
    }
    out
}

fn render_comparanda(graph: &Graph, table: &routing::RoutingTable, turn_text: &str, now: u64) -> String {
    let p = projection::project_with_table(
        graph,
        table,
        turn_text,
        now,
        HARVEST_PER_MIDPOINT_CAP,
        HARVEST_TOTAL_CAP,
    );
    let mut out = String::new();
    let mut seen: Vec<&str> = Vec::new();
    for o in &p.opened {
        for id in &o.leaf_ids {
            if seen.contains(&id.as_str()) {
                continue;
            }
            seen.push(id);
            if let Some(l) = graph.leaves.get(id) {
                let _ = writeln!(
                    out,
                    "- [{}] ({}, under {}) {}",
                    l.id,
                    l.species().name(),
                    l.parents.join("+"),
                    l.text
                );
            }
        }
    }
    if out.is_empty() {
        out.push_str("(none)\n");
    }
    out
}

/// The profile is pinned for the harvester exactly as it is pinned for
/// recall: every disposition, every time, regardless of what the turn
/// routes to. Cross-matching against a sample would let the harvester
/// mint a near-duplicate of an axis it was never shown — belief strength
/// can only accumulate on an axis the harvester can see.
fn render_pinned_profile(graph: &Graph) -> String {
    let mut out = String::new();
    let mut roots = graph.roots(Tree::Profile);
    roots.sort_by(|a, b| a.id.cmp(&b.id));
    let mut stack: Vec<String> = roots.iter().rev().map(|r| r.id.clone()).collect();
    while let Some(id) = stack.pop() {
        let mut leaves = graph.leaves_under(&id);
        leaves.sort_by(|a, b| a.id.cmp(&b.id));
        for l in leaves {
            match &l.kind {
                LeafKind::Disposition(d) => {
                    let _ = writeln!(out, "- [{}] (under {}) {} (axis {:+.2}, {} observations)", l.id, id, l.text, d.axis, d.nudges.len());
                }
                LeafKind::State(_) | LeafKind::Rule(_) => {
                    let _ = writeln!(out, "- [{}] (under {}) {}", l.id, id, l.text);
                }
            }
        }
        let mut children = graph.child_distillants(&id);
        children.sort_by(|a, b| b.id.cmp(&a.id));
        stack.extend(children.into_iter().map(|c| c.id.clone()));
    }
    if out.is_empty() {
        out.push_str("(no axes yet)\n");
    }
    out
}

/// A rule-shaped ask the assistant acknowledged that memory has not yet
/// resolved: the host's id for it, the user's words, the assistant's reading.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Instruction {
    pub id: String,
    pub utterance: String,
    pub paraphrase: String,
}

/// A saved document the host holds: what to call it, the text the
/// harvester reads, and when it was written if the host knows. The engine
/// frames it for the harvester and accepts what the harvester cuts from
/// it; the document itself, and the record of its import, stay the
/// host's. A host that wants the import on the user's record pushes an
/// [`Op::Episode`] of its own into the batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Document {
    pub name: String,
    pub text: String,
    /// When the document was written, unix seconds: the age the harvester
    /// reads its words at, so a two-year-old note does not read as today's
    /// words. None when the host does not know.
    pub written_at: Option<u64>,
}

/// What the harvester reads of a finished turn and pending source material.
/// `field_manual` is the host-rendered upsert compare point. `instructions`
/// receive Rule-only verdicts; `documents` can contribute every memory kind.
/// Empty inputs omit their sections.
#[derive(Clone, Debug)]
pub struct HarvestInput<'a> {
    pub user_text: &'a str,
    pub assistant_text: &'a str,
    pub field_manual: &'a str,
    pub instructions: &'a [Instruction],
    pub documents: &'a [Document],
}

impl<'a> HarvestInput<'a> {
    /// The bare turn: no field manual, nothing awaiting memory.
    pub fn turn(user_text: &'a str, assistant_text: &'a str) -> Self {
        HarvestInput { user_text, assistant_text, field_manual: "", instructions: &[], documents: &[] }
    }
}

pub fn build_user_message(graph: &Graph, input: &HarvestInput, now: u64, scope: DirectoryScope) -> String {
    let HarvestInput { user_text, assistant_text, field_manual, instructions, documents } = *input;
    let documents_section = render_documents(documents, now);
    // What routes: the words said and the words imported, not the
    // section's own framing.
    let turn_text = documents.iter().fold(format!("{user_text} {assistant_text}"), |mut text, d| {
        let _ = write!(text, " {} {}", d.name, d.text);
        text
    });
    let table = routing::RoutingTable::build(graph);
    let manual_section = match field_manual.trim().is_empty() {
        true => String::new(),
        false => format!(
            "# Field manual (existing entries — a manual_upserts entry REPLACES its tool+service row)\n{}\n",
            field_manual.trim()
        ),
    };
    let instructions_section = match instructions.is_empty() {
        true => String::new(),
        false => format!(
            "# Instructions to resolve (rule-shaped asks the assistant acknowledged; answer each by id with exactly one rules entry, or none when it is not a standing instruction)\n{}\n",
            render_instructions(instructions)
        ),
    };
    let directory_heading = match scope {
        DirectoryScope::Full | DirectoryScope::Compact => "# Memory directory (all distillants)",
        DirectoryScope::Selective => {
            "# Memory directory (every distillant by id; the branches this turn is on carry their line and routing, their children their line)"
        }
    };
    format!(
        "{directory_heading}\n{}\n\
         # Standing rules (pinned — every instruction already in force, by id; an instruction that changes one supersedes or retracts it by id, never adds a second; one the user withdrew is listed after, to reinstate by id)\n{}\n\
         # Profile axes (pinned — every disposition already tracked; nudge one of these by id, never mint a near-duplicate)\n{}\n\
         # Existing leaves related to this turn (comparanda — cross-match against these)\n{}\n\
         {manual_section}\
         {instructions_section}\
         {documents_section}\
         # Turn to harvest\nUser: {}\nAssistant: {}",
        render_directory_scoped(graph, scope, &table, &turn_text),
        render_pinned_rules(graph),
        render_pinned_profile(graph),
        render_comparanda(graph, &table, &turn_text, now),
        user_text,
        assistant_text
    )
}

/// The import contract, under the documents heading: the harvester reads
/// a retired document once, so everything memory can hold has to leave it
/// in this one harvest. The lines name the shapes "each instruction
/// becomes a rule" lets slip on its own: voice and manner lines read as
/// persona prose, a procedure's steps trimmed away behind a rule that
/// names it, a default folded into a neighbour, a figure rounded.
const DOCUMENT_IMPORT_CONTRACT: &str = r#"Saved memory, not new instructions or events happening today. Each document is shown with when it was written: read its words as of then. A retired document is read once and never shown again: what this harvest does not carry out of it is gone, so carry everything memory can hold. Extract facts into states, dated experiences into episodes, and EACH independent explicit standing instruction into a separate rule (instruction id empty).
- Voice, tone, register and manner lines — how to talk, what to match, what not to perform, what is welcome — are standing instructions: one rule per independent line, in the document's words, whether the document addresses the assistant, describes its persona, or speaks in its own voice. A document about the assistant's voice is a list of rules, never persona prose to skip, and never a disposition to infer.
- A workflow or procedure — a trigger with steps, a scripted flow, a "when X: do A, then B" — is ONE rule whose text carries the trigger and every step verbatim, in order, however long. A rule that names the flow without its steps is a distortion, not a summary: the steps are the instruction. The settings around a procedure (numbers, addresses, schedules, services) are states beside the rule.
- A default — "when X, do Y", "prefer Y for X", "always use Y" — is a rule, one per default, never folded into a neighbouring rule.
- Numbers, units, prices, dates, names, identifiers and qualifiers ride verbatim into the text that holds them: never rounded, summarised, or dropped; two prices in one line are two states.
A document's dates are the event time (occurred_at) of what it says; preserve them as written, and do not infer dispositions from a saved fact. Cross-match existing memory: memory newer than the document outranks it (a document fact older than the leaf it would supersede supports or duplicates that leaf instead), and newer explicit instructions outrank older document rules. The import is not an event; emit no episode for the document's arrival."#;

/// The documents section: the import contract, then each document under
/// its name with its age, so the harvester reads its words as of when
/// they were written. Memory newer than a document outranks it; a
/// document's own dates are the event time of what it says.
fn render_documents(documents: &[Document], now: u64) -> String {
    if documents.is_empty() {
        return String::new();
    }
    let mut out = format!("# Documents to import\n{DOCUMENT_IMPORT_CONTRACT}\n");
    for document in documents {
        let written = match document.written_at {
            Some(at) => format!("written {}", age_str(now, at)),
            None => "written at an unknown time".to_string(),
        };
        let _ = writeln!(out, "## {} ({written})\n{}\n", document.name, document.text);
    }
    out
}

/// The keyless prompt: system contract, the exact output schema, and the
/// rendered input — self-contained, like `render_digest_prompt` and
/// `render_redistill_prompt`. A driver holding only this render can play
/// the harvester role. The directory is the design's, [`DirectoryScope::Selective`].
pub fn render_harvest_prompt(graph: &Graph, input: &HarvestInput, now: u64) -> String {
    render_harvest_prompt_scoped(graph, input, now, DirectoryScope::Selective)
}

/// The keyless prompt under a chosen [`DirectoryScope`], for the replay
/// that compares them.
pub fn render_harvest_prompt_scoped(graph: &Graph, input: &HarvestInput, now: u64, scope: DirectoryScope) -> String {
    format!(
        "# System\n{HARVESTER_SYSTEM}\n\n# Output schema (reply with one JSON object matching it)\n{}\n\n# Input\n{}",
        serde_json::to_string_pretty(&ops_schema()).expect("static schema serializes"),
        build_user_message(graph, input, now, scope)
    )
}

pub fn parse_ops(text: &str) -> Result<Vec<Op>> {
    let out: HarvestOut = serde_json::from_str(text)
        .with_context(|| format!("harvester returned unparseable ops: {text}"))?;
    Ok(out.into_ops())
}

/// Run the harvester over one finished turn and apply what it found.
pub fn run(llm: &dyn Llm, graph: &mut Graph, input: &HarvestInput, now: u64) -> Result<Applied> {
    let req = ChatRequest {
        system: HARVESTER_SYSTEM.to_string(),
        messages: vec![json!({
            "role": "user",
            "content": build_user_message(graph, input, now, DirectoryScope::Selective)
        })],
        tools: vec![],
        output_schema: Some(ops_schema()),
        max_tokens: 4096,
        thinking: false,
    };
    // Constrained decoding occasionally degenerates mid-output; one fresh
    // attempt recovers it. Parse failure twice in a row is a real error.
    let mut last_err = None;
    for _ in 0..2 {
        let resp = llm.chat(&req)?;
        if std::env::var("NARRATIVE_DEBUG").is_ok() {
            eprintln!("[debug] raw harvest: {}", resp.text());
        }
        match parse_ops(&resp.text()) {
            Ok(ops) => return Ok(apply_ops(graph, ops, now)),
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.expect("loop ran"))
}

/// The reserved open id is no distillant: nothing creates one under it.
fn reserved(id: &str, traces: &mut Vec<String>) -> bool {
    let reserved = id == projection::RULES;
    if reserved {
        traces.push(format!("⚠ \"{id}\" is the open id of the standing instructions, not a distillant; skipped"));
    }
    reserved
}

/// Bring a distillant a leaf op names into being when it is missing.
/// False when the id is no distillant at all, so the leaf does not hang
/// under it.
fn ensure_distillant(graph: &mut Graph, id: &str, tree: Tree, traces: &mut Vec<String>) -> bool {
    if reserved(id, traces) {
        return false;
    }
    if graph.distillants.contains_key(id) {
        return true;
    }
    let label = id.rsplit('/').next().unwrap_or(id).replace('-', " ");
    let named_parent = match id.rsplit_once('/') {
        Some((prefix, _)) if graph.distillants.contains_key(prefix) => vec![prefix.to_string()],
        _ => Vec::new(),
    };
    let parents = graph.home_parents(id, tree, named_parent);
    // A bare stub has no line: nothing distilled it. Leaving the line empty
    // is what makes it visibly bare — every render says "(no line yet)" and
    // consolidation treats it as due (`consolidate::pressure`), where the
    // pass writes its first line or merges it into a same-named distillant.
    graph.distillants.insert(id.to_string(), Distillant::bare(id, tree, &label, parents, Vec::new()));
    traces.push(format!("⚠ auto-created bare distillant {id} (harvester skipped the distillant op)"));
    true
}

pub(crate) fn add_routing(
    graph: &mut Graph,
    distillant_id: &str,
    terms: &[String],
    traces: &mut Vec<String>,
) {
    let Some(m) = graph.distillants.get_mut(distillant_id) else { return };
    let mut added = Vec::new();
    let mut gated = Vec::new();
    for t in terms {
        let t = t.trim().to_lowercase();
        if t.is_empty() || m.routing.iter().any(|r| r.eq_ignore_ascii_case(&t)) {
            continue;
        }
        // The line gates facet vocabulary: a facet term may route
        // only while the line advertises its family. Referring
        // expressions, anchors and topical words pass untouched.
        if let Some(fam) = routing::facet_family(&t)
            && !routing::distillant_carries(fam, &m.line)
        {
            gated.push(t);
            continue;
        }
        m.routing.push(t.clone());
        added.push(t);
    }
    if !added.is_empty() {
        traces.push(format!("≈ routing {} ← {}", distillant_id, added.join(", ")));
    }
    if !gated.is_empty() {
        traces.push(format!(
            "✋ routing {} ✗ {} (facet the line does not carry)",
            distillant_id,
            gated.join(", ")
        ));
    }
}

/// Facet routing is derived, not model-written: sync the single-word facet
/// vocabulary in routing to exactly the families the line carries —
/// prune what the rewritten line dropped, mirror in the full family
/// (inflections and synonyms: a "fear" line routes "afraid") for
/// what it carries. The model's judgment lives in the line; the
/// runtime compiles it down. Phrases, aliases, anchors and topical terms
/// are never touched. Runs wherever a line is set or rewritten.
pub(crate) fn sync_facet_routing(graph: &mut Graph, distillant_id: &str, traces: &mut Vec<String>) {
    let Some(m) = graph.distillants.get_mut(distillant_id) else { return };
    let line = m.line.clone();
    let mut pruned = Vec::new();
    m.routing.retain(|t| match routing::facet_family(t) {
        Some(fam) if !routing::distillant_carries(fam, &line) => {
            pruned.push(t.clone());
            false
        }
        _ => true,
    });
    let mut mirrored = Vec::new();
    for fam in routing::FACET_FAMILIES {
        if !routing::distillant_carries(fam, &line) {
            continue;
        }
        for w in *fam {
            if !m.routing.iter().any(|r| r.eq_ignore_ascii_case(w)) {
                m.routing.push((*w).to_string());
                mirrored.push(*w);
            }
        }
    }
    if !pruned.is_empty() {
        traces.push(format!(
            "✂ routing {} − {} (line no longer carries it)",
            distillant_id,
            pruned.join(", ")
        ));
    }
    if !mirrored.is_empty() {
        traces.push(format!("≈ routing {} ⇐ line: {}", distillant_id, mirrored.join(", ")));
    }
}

/// What one rules entry did to the standing rules.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RuleEffect {
    /// A new rule is in force.
    Kept { id: String, text: String },
    /// A standing rule's wording changed.
    Superseded { id: String, from: String, to: String },
    /// A standing rule was withdrawn: out of force, on record. Also the
    /// answer to withdrawing one already withdrawn.
    Retracted { id: String, text: String },
    /// The instruction restated a rule already in force.
    Duplicate { id: String, text: String },
    /// A withdrawn rule is in force again, with the words it was given
    /// again in; earlier words stay in its history.
    Reinstated { id: String, text: String },
}

/// One rules entry's effect, with the host's instruction it answered when
/// it answered one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuleOutcome {
    pub instruction: Option<String>,
    pub effect: RuleEffect,
}

/// What applying a harvest did: the traces, and every rules entry's effect.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Applied {
    pub traces: Vec<String>,
    pub rules: Vec<RuleOutcome>,
}

/// How memory answered one instruction the host sent to be resolved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Resolution {
    Rule(RuleEffect),
    /// No rules entry answered it: the harvester judged it not a standing
    /// instruction, or found nothing to retract.
    NotARule,
}

/// Every instruction the host sent, resolved: the last rules entry that
/// answered it, or [`Resolution::NotARule`] when none did.
pub fn resolve(instructions: &[Instruction], applied: &Applied) -> Vec<(String, Resolution)> {
    instructions
        .iter()
        .map(|i| {
            let answered = applied
                .rules
                .iter()
                .rev()
                .find(|o| o.instruction.as_deref() == Some(i.id.as_str()))
                .map(|o| Resolution::Rule(o.effect.clone()));
            (i.id.clone(), answered.unwrap_or(Resolution::NotARule))
        })
        .collect()
}

pub fn apply_ops(graph: &mut Graph, ops: Vec<Op>, now: u64) -> Applied {
    let mut traces = Vec::new();
    let mut rules = Vec::new();

    // Every apply starts on the structural invariants, so a graph written
    // before they held is repaired the first time it is touched.
    let repaired = graph.normalize();
    if repaired.apex_created {
        traces.push(format!("⇢ profile apex {} created", PROFILE_APEX));
    }
    if !repaired.homed_under_apex.is_empty() {
        traces.push(format!("⇢ axes homed under the apex: {}", repaired.homed_under_apex.join(", ")));
    }
    if !repaired.antichained.is_empty() {
        traces.push(format!("⇢ parent sets antichained: {}", repaired.antichained.join(", ")));
    }

    // Distillants first so leaves have somewhere to hang.
    for op in &ops {
        if let Op::Distillant { id, tree, label, line, routing, parents } = op {
            if reserved(id, &mut traces) {
                continue;
            }
            let existed = graph.distillants.contains_key(id);
            let parents = graph.home_parents(id, *tree, parents.clone());
            let entry = graph.distillants.entry(id.clone()).or_insert(Distillant {
                id: id.clone(),
                tree: *tree,
                label: label.clone(),
                line: line.clone(),
                routing: Vec::new(),
                parents,
                misc_count: 0,
                consolidated_at: 0,
                // A brand-new line is changed material from any parent's view.
                line_changed_at: now,
                forgotten_at: 0,
            });
            let rewritten = existed && !line.is_empty();
            if existed {
                if rewritten {
                    if entry.line != *line {
                        entry.line_changed_at = now;
                    }
                    entry.line = line.clone();
                }
            } else {
                traces.push(format!("✚ distillant {} — {}", id, line));
            }
            if rewritten || !existed {
                sync_facet_routing(graph, id, &mut traces);
            }
            let routing = routing.clone();
            add_routing(graph, id, &routing, &mut traces);
        }
    }

    // Episodes next; they become evidence for this batch's leaves.
    let mut evidence: Vec<String> = Vec::new();
    for op in &ops {
        if let Op::Episode { text, tags, occurred_at } = op {
            let id = graph.push_episode(text.clone(), tags.clone(), now, *occurred_at);
            traces.push(format!("◦ episode {id}: {text}"));
            evidence.push(id);
        }
    }

    for op in ops {
        match op {
            Op::Episode { .. } | Op::Distillant { .. } => {}
            Op::State { id, distillants, text, relation, target, importance, aliases, occurred_at } => {
                let distillants: Vec<String> =
                    distillants.into_iter().filter(|m| ensure_distillant(graph, m, Tree::Registry, &mut traces)).collect();
                if let Some(first) = distillants.first() {
                    add_routing(graph, first, &aliases, &mut traces);
                }
                let target_id = if target.is_empty() { id.clone() } else { target.clone() };
                let importance = importance.clamp(0.0, 1.0);
                if !graph.leaves.contains_key(&target_id) {
                    // Fresh insert — the novel path, and the self-healing
                    // fallback when a non-novel relation names a missing leaf.
                    // The fallback lands on `id`, which the target check did
                    // not cover: an occupied id is someone else's record.
                    if let Some(occupant) = graph.leaves.get(&id) {
                        traces.push(format!(
                            "⚠ {relation:?} on missing leaf [{target_id}] would overwrite the {} [{id}]; skipped",
                            occupant.species().name()
                        ));
                        continue;
                    }
                    if relation != Relation::Novel {
                        traces.push(format!(
                            "⚠ {relation:?} on missing leaf [{target_id}]; storing as novel"
                        ));
                    }
                    let homes = graph.antichain(distillants.clone());
                    let mut leaf = Leaf::state(id.clone(), text.clone(), homes, importance, now);
                    leaf.evidence = evidence.clone();
                    leaf.occurred_at = occurred_at;
                    // Residual pressure: a fact hanging directly off a crown root
                    // is gradient signal for the next consolidation pass.
                    for p in &leaf.parents {
                        if let Some(m) = graph.distillants.get_mut(p)
                            && m.parents.is_empty()
                        {
                            m.misc_count += 1;
                        }
                    }
                    traces.push(format!("+ state [{}] {} (under {})", id, text, distillants.join("+")));
                    graph.leaves.insert(id.clone(), leaf);
                } else {
                    let leaf = graph.leaves.get_mut(&target_id).expect("checked above");
                    let Leaf { text: leaf_text, kind, evidence: leaf_evidence, occurred_at: leaf_occurred_at, updated_at, .. } = leaf;
                    let LeafKind::State(state) = kind else {
                        traces.push(format!("⚠ state op targets the {} [{target_id}]; skipped", kind.species().name()));
                        continue;
                    };
                    match relation {
                        Relation::Novel if *leaf_text == text => {
                            belief::support(&mut state.belief, now);
                            traces.push(format!("↑ re-affirmed [{}]", target_id));
                        }
                        // Novel on an occupied id with other words is a
                        // supersession the harvester did not name: states switch.
                        Relation::Novel | Relation::Supersedes => {
                            let old = std::mem::replace(leaf_text, text.clone());
                            belief::supersede(state, old, occurred_at.unwrap_or(now), now);
                            *leaf_occurred_at = occurred_at;
                            *updated_at = now;
                            traces.push(format!("⇄ superseded [{}] → {}", target_id, text));
                        }
                        Relation::Duplicate => {
                            // No weight, no wording, no clock moves — but the
                            // batch's source is still where this was said.
                            traces.push(format!("≡ duplicate of [{}] (no change)", target_id));
                        }
                        Relation::Supports => {
                            belief::support(&mut state.belief, now);
                            *updated_at = now;
                            traces.push(format!(
                                "↑ supports [{}] ({} for / {} against)",
                                target_id, state.belief.support, state.belief.contradict
                            ));
                        }
                        Relation::Contradicts => {
                            belief::contradict(&mut state.belief, now);
                            *updated_at = now;
                            let s = belief::strength(&state.belief);
                            traces.push(format!(
                                "↓ contradicts [{}] (strength now {:.2})",
                                target_id, s
                            ));
                        }
                    }
                    // Every accepted relation cites the batch: what a leaf
                    // is evidence of is also what it can be forgotten with.
                    attach_evidence(leaf_evidence, &evidence);
                }
            }

            Op::Disposition { id, distillants, text, dir, note, importance } => {
                let distillants: Vec<String> =
                    distillants.into_iter().filter(|m| ensure_distillant(graph, m, Tree::Profile, &mut traces)).collect();
                let importance = importance.clamp(0.0, 1.0);
                if let Some(leaf) = graph.leaves.get_mut(&id) {
                    let Leaf { text: leaf_text, kind, updated_at, .. } = leaf;
                    let LeafKind::Disposition(d) = kind else {
                        traces.push(format!("⚠ nudge targets the {} [{id}]; skipped", kind.species().name()));
                        continue;
                    };
                    let flipped = belief::apply_nudge(d, dir, note, now);
                    *updated_at = now;
                    if !text.is_empty() && *leaf_text != text {
                        *leaf_text = text;
                    }
                    let mut line = format!("~ nudge [{}] {:+} (axis {:+.2})", id, dir, d.axis);
                    if flipped {
                        line.push_str(" — POLARITY FLIPPED");
                    }
                    traces.push(line);
                } else {
                    let homes = graph.antichain(distillants);
                    // Born with one supporting event (the birth itself) and its
                    // first nudge as the whole trajectory.
                    let mut leaf = Leaf::disposition(id.clone(), text.clone(), homes, importance, now);
                    let LeafKind::Disposition(d) = &mut leaf.kind else { unreachable!("built as a disposition") };
                    d.axis = belief::nudge_axis(0.0, dir);
                    d.nudges.push(Nudge { dir, note, at: now });
                    let axis = d.axis;
                    leaf.evidence = evidence.clone();
                    traces.push(format!("+ disposition [{}] {} (axis {:+.2})", id, text, axis));
                    graph.leaves.insert(id, leaf);
                }
            }
            Op::Rule(op) => apply_rule(graph, op, now, &evidence, &mut traces, &mut rules),
            Op::Reinforce { target } => {
                let Some(leaf) = graph.leaves.get_mut(&target) else {
                    traces.push(format!("⚠ reinforce on missing leaf [{target}]; skipped"));
                    continue;
                };
                let Leaf { kind, evidence: leaf_evidence, updated_at, .. } = leaf;
                let Some(belief) = kind.belief_mut() else {
                    traces.push(format!("⚠ reinforce on rule [{target}]; a rule carries no belief; skipped"));
                    continue;
                };
                belief::support(belief, now);
                attach_evidence(leaf_evidence, &evidence);
                *updated_at = now;
                traces.push(format!(
                    "↑ reinforced [{}] ({} for / {} against)",
                    target, belief.support, belief.contradict
                ));
            }
            Op::Alias { distillant, add } => {
                if graph.distillants.contains_key(&distillant) {
                    add_routing(graph, &distillant, &add, &mut traces);
                } else {
                    traces.push(format!("⚠ alias for unknown distillant {distillant}; skipped"));
                }
            }
            Op::Distill { distillant, line } => {
                if let Some(m) = graph.distillants.get_mut(&distillant) {
                    if m.line != line {
                        m.line_changed_at = now;
                    }
                    m.line = line.clone();
                    traces.push(format!("✎ distill {} — {}", distillant, line));
                    sync_facet_routing(graph, &distillant, &mut traces);
                } else {
                    traces.push(format!("⚠ distill for unknown distillant {distillant}; skipped"));
                }
            }
            Op::Move { leaf, parents } => apply_move(graph, leaf, parents, now, &mut traces),
            Op::Reparent { distillant, parents } => {
                apply_reparent(graph, distillant, parents, &mut traces)
            }
            Op::MergeLeaf { from, into } => apply_merge_leaf(graph, from, into, now, &mut traces),
            Op::MergeDistillant { from, into } => apply_merge_distillant(graph, from, into, &mut traces),
            Op::Forget { target } => apply_forget(graph, target, now, &mut evidence, &mut traces),
            // Host-applied: the driver stores field-manual entries as typed
            // rows outside the graph. Reaching this arm means the driver
            // did not peel them — trace it, touch nothing.
            Op::ManualUpsert { tool, service, .. } | Op::ManualRetire { tool, service } => {
                traces.push(format!(
                    "⚠ field-manual op for {tool}/{service} reached the graph apply — host-applied, dropped"
                ));
            }
        }
    }
    Applied { traces, rules }
}

/// Cite the batch's episodes on a leaf, once each — several ops in one
/// batch may land on the same leaf.
fn attach_evidence(leaf_evidence: &mut Vec<String>, evidence: &[String]) {
    for e in evidence {
        if !leaf_evidence.contains(e) {
            leaf_evidence.push(e.clone());
        }
    }
}

/// The words a rules entry writes: the instruction text, the host
/// instruction it answers, and when it was given.
struct RuleWords {
    text: String,
    instruction: Option<String>,
    occurred_at: Option<u64>,
}

/// A new rule in force: the user's words under no distillant, the batch's
/// episodes as evidence.
fn keep_rule(graph: &mut Graph, id: String, words: RuleWords, now: u64, evidence: &[String], traces: &mut Vec<String>) -> RuleEffect {
    let RuleWords { text, instruction, occurred_at } = words;
    let mut leaf = Leaf::rule(id.clone(), text.clone(), instruction, now);
    leaf.evidence = evidence.to_vec();
    leaf.occurred_at = occurred_at;
    traces.push(format!("+ rule [{}] {}", id, text));
    graph.leaves.insert(id.clone(), leaf);
    RuleEffect::Kept { id, text }
}

/// A standing rule reworded: the old words become history, the instruction
/// that changed it is the one it now answers.
fn supersede_rule(leaf: &mut Leaf, words: RuleWords, now: u64, evidence: &[String], traces: &mut Vec<String>) -> RuleEffect {
    let RuleWords { text, instruction, occurred_at } = words;
    let Leaf { id, text: leaf_text, kind, evidence: leaf_evidence, occurred_at: leaf_occurred_at, updated_at, .. } = leaf;
    let LeafKind::Rule(rule) = kind else { unreachable!("a standing rule is a rule leaf") };
    let from = std::mem::replace(leaf_text, text.clone());
    rule.history.push(Supersession { value: from.clone(), superseded_at: occurred_at.unwrap_or(now) });
    if instruction.is_some() {
        rule.instruction = instruction;
    }
    *leaf_occurred_at = occurred_at;
    *updated_at = now;
    attach_evidence(leaf_evidence, evidence);
    traces.push(format!("⇄ rule superseded [{}] → {}", id, text));
    RuleEffect::Superseded { id: id.clone(), from, to: text }
}

/// The user withdrew a standing rule: out of force, out of every prompt,
/// on record under its id with the batch that withdrew it cited.
fn retract_rule(leaf: &mut Leaf, at: u64, now: u64, evidence: &[String], traces: &mut Vec<String>) -> RuleEffect {
    let Leaf { id, text, kind, evidence: leaf_evidence, updated_at, .. } = leaf;
    let LeafKind::Rule(rule) = kind else { unreachable!("a standing rule is a rule leaf") };
    rule.retracted_at = Some(at);
    *updated_at = now;
    attach_evidence(leaf_evidence, evidence);
    traces.push(format!("✗ rule withdrawn [{}] — {}", id, text));
    RuleEffect::Retracted { id: id.clone(), text: text.clone() }
}

/// The user gave a withdrawn rule again: in force under the same id, in
/// the words given now; words that differ from the withdrawn ones go to
/// its history like any rewording.
fn reinstate_rule(leaf: &mut Leaf, words: RuleWords, now: u64, evidence: &[String], traces: &mut Vec<String>) -> RuleEffect {
    let RuleWords { text, instruction, occurred_at } = words;
    let Leaf { id, text: leaf_text, kind, evidence: leaf_evidence, occurred_at: leaf_occurred_at, updated_at, .. } = leaf;
    let LeafKind::Rule(rule) = kind else { unreachable!("a withdrawn rule is a rule leaf") };
    rule.retracted_at = None;
    if *leaf_text != text {
        let from = std::mem::replace(leaf_text, text.clone());
        rule.history.push(Supersession { value: from, superseded_at: occurred_at.unwrap_or(now) });
    }
    if instruction.is_some() {
        rule.instruction = instruction;
    }
    *leaf_occurred_at = occurred_at;
    *updated_at = now;
    attach_evidence(leaf_evidence, evidence);
    traces.push(format!("↺ rule reinstated [{}] — {}", id, text));
    RuleEffect::Reinstated { id: id.clone(), text }
}

/// What a rules entry's target id names.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RuleTarget {
    /// A rule in force.
    Standing,
    /// A rule the user withdrew: on record, to reinstate.
    Withdrawn,
    /// No rule; the words are new.
    Absent,
}

/// One rules entry against the rules. A rule id is never shared with a
/// state or disposition, and a rule is never weighed, so every relation
/// resolves by what its target names: a rule in force, one withdrawn, or
/// nothing.
fn apply_rule(
    graph: &mut Graph,
    op: RuleOp,
    now: u64,
    evidence: &[String],
    traces: &mut Vec<String>,
    outcomes: &mut Vec<RuleOutcome>,
) {
    let RuleOp { id, text, relation, target, instruction, occurred_at } = op;
    let instruction = (!instruction.trim().is_empty()).then(|| instruction.trim().to_string());
    let target_id = if target.is_empty() { id.clone() } else { target.clone() };
    let text = text.trim().to_string();
    let target = match graph.leaves.get(&target_id) {
        Some(leaf) if leaf.is_standing_rule() => RuleTarget::Standing,
        Some(leaf) if leaf.is_withdrawn_rule() => RuleTarget::Withdrawn,
        Some(leaf) => {
            traces.push(format!("⚠ rule op targets the {} [{target_id}]; skipped", leaf.species().name()));
            return;
        }
        None => RuleTarget::Absent,
    };
    let names_nothing_to_keep = text.is_empty() && relation != RuleRelation::Retract;
    if names_nothing_to_keep {
        traces.push(format!("⚠ rule [{target_id}] with no text; skipped"));
        return;
    }
    // A relation on no rule keeps the words under `id`, which the target
    // check did not cover: an occupied id is someone else's record,
    // whatever its species.
    if target == RuleTarget::Absent
        && relation != RuleRelation::Retract
        && let Some(occupant) = graph.leaves.get(&id)
    {
        traces.push(format!(
            "⚠ {relation:?} on missing rule [{target_id}] would overwrite the {} [{id}]; skipped",
            occupant.species().name()
        ));
        return;
    }
    let answers = instruction.clone();
    let words = RuleWords { text, instruction, occurred_at };
    let effect = match (relation, target) {
        (RuleRelation::Novel, RuleTarget::Absent) => keep_rule(graph, id, words, now, evidence, traces),
        (RuleRelation::Novel, RuleTarget::Standing) => {
            // Id collision: new words on a standing id supersede; the same
            // words restate it.
            let leaf = graph.leaves.get_mut(&target_id).expect("standing");
            if leaf.text == words.text {
                attach_evidence(&mut leaf.evidence, evidence);
                traces.push(format!("≡ rule [{target_id}] re-affirmed (no change)"));
                RuleEffect::Duplicate { id: target_id, text: words.text }
            } else {
                supersede_rule(leaf, words, now, evidence, traces)
            }
        }
        (RuleRelation::Supersedes, RuleTarget::Standing) => {
            let leaf = graph.leaves.get_mut(&target_id).expect("standing");
            supersede_rule(leaf, words, now, evidence, traces)
        }
        (RuleRelation::Supersedes, RuleTarget::Absent) | (RuleRelation::Duplicate, RuleTarget::Absent) => {
            // The self-healing fallback: a relation naming no rule keeps
            // the words as a new rule.
            traces.push(format!("⚠ {relation:?} on missing rule [{target_id}]; storing as novel"));
            keep_rule(graph, id, words, now, evidence, traces)
        }
        (RuleRelation::Duplicate, RuleTarget::Standing) => {
            // The words stand as they are; the batch's source joins what
            // the rule cites, so a forget takes it too.
            let leaf = graph.leaves.get_mut(&target_id).expect("standing");
            attach_evidence(&mut leaf.evidence, evidence);
            traces.push(format!("≡ duplicate of rule [{target_id}] (no change)"));
            RuleEffect::Duplicate { id: target_id, text: leaf.text.clone() }
        }
        // The user gave a withdrawn rule again, in whichever relation the
        // harvester read it as: it is in force again under its own id.
        (RuleRelation::Novel | RuleRelation::Supersedes | RuleRelation::Duplicate, RuleTarget::Withdrawn) => {
            let leaf = graph.leaves.get_mut(&target_id).expect("withdrawn");
            reinstate_rule(leaf, words, now, evidence, traces)
        }
        (RuleRelation::Retract, RuleTarget::Standing) => {
            // The user withdrew it: out of force and out of every prompt,
            // on record under its id. Nothing else moves; only a forget
            // erases.
            let leaf = graph.leaves.get_mut(&target_id).expect("standing");
            retract_rule(leaf, words.occurred_at.unwrap_or(now), now, evidence, traces)
        }
        (RuleRelation::Retract, RuleTarget::Withdrawn) => {
            // Withdrawing it again changes nothing and answers the same.
            let leaf = graph.leaves.get_mut(&target_id).expect("withdrawn");
            attach_evidence(&mut leaf.evidence, evidence);
            traces.push(format!("≡ rule [{target_id}] already withdrawn (no change)"));
            RuleEffect::Retracted { id: target_id, text: leaf.text.clone() }
        }
        (RuleRelation::Retract, RuleTarget::Absent) => {
            traces.push(format!("⚠ retract of unknown rule [{target_id}]; skipped"));
            return;
        }
    };
    outcomes.push(RuleOutcome { instruction: answers, effect });
}

/// The user asked to forget: what holds the content leaves every store
/// (`Graph::forget`). A forget may take this batch's own episodes with
/// it, so `evidence` is pruned to what the stream still holds: nothing
/// applied after it cites what left.
fn apply_forget(graph: &mut Graph, target: String, now: u64, evidence: &mut Vec<String>, traces: &mut Vec<String>) {
    let Some(gone) = graph.forget(&target, now) else {
        traces.push(format!("⚠ forget of unknown id [{target}]; skipped"));
        return;
    };
    evidence.retain(|e| !gone.episodes.contains(e));
    traces.push(format!(
        "✗ forgot [{}] — {} leaves, {} distillants, {} episodes left the graph",
        target,
        gone.leaves.len(),
        gone.distillants.len(),
        gone.episodes.len()
    ));
}

fn apply_move(graph: &mut Graph, leaf_id: String, parents: Vec<String>, now: u64, traces: &mut Vec<String>) {
    let Some(species) = graph.leaves.get(&leaf_id).map(|l| l.species()) else {
        traces.push(format!("⚠ move of unknown leaf [{leaf_id}]; skipped"));
        return;
    };
    let tree = match species {
        Species::State => Tree::Registry,
        Species::Disposition => Tree::Profile,
        Species::Rule => {
            traces.push(format!("⚠ move of rule [{leaf_id}]; a rule hangs under nothing; skipped"));
            return;
        }
    };
    let parents = graph.antichain(parents);
    let parents: Vec<String> = parents.into_iter().filter(|p| ensure_distillant(graph, p, tree, traces)).collect();
    if parents.is_empty() {
        traces.push(format!("⚠ move of [{leaf_id}] with no parents; skipped"));
        return;
    }
    let leaf = graph.leaves.get_mut(&leaf_id).expect("checked above");
    leaf.parents = parents;
    leaf.updated_at = now;
    traces.push(format!("→ moved [{}] under {}", leaf_id, leaf.parents.join("+")));
}

fn apply_reparent(graph: &mut Graph, distillant_id: String, parents: Vec<String>, traces: &mut Vec<String>) {
    let Some(tree) = graph.distillants.get(&distillant_id).map(|m| m.tree) else {
        traces.push(format!("⚠ reparent of unknown distillant {distillant_id}; skipped"));
        return;
    };
    let mut kept = Vec::new();
    for p in parents {
        if p == distillant_id || graph.is_ancestor(&distillant_id, &p) {
            traces.push(format!("⚠ reparent {distillant_id} under {p} would cycle; parent dropped"));
        } else if graph.distillants.get(&p).map(|m| m.tree) != Some(tree) {
            traces.push(format!("⚠ reparent {distillant_id}: parent {p} missing or wrong tree; dropped"));
        } else if !kept.contains(&p) {
            kept.push(p);
        }
    }
    let kept = graph.home_parents(&distillant_id, tree, kept);
    let m = graph.distillants.get_mut(&distillant_id).expect("checked above");
    m.parents = kept;
    let dest = if m.parents.is_empty() { "(crown root)".to_string() } else { m.parents.join("+") };
    traces.push(format!("→ reparented {distillant_id} under {dest}"));
}

/// A merge combines two weighed leaves of one species. A rule is never
/// merged: two rules saying the same thing are one supersession away from
/// one.
fn mergeable(from: &Leaf, into: &Leaf) -> bool {
    matches!(
        (&from.kind, &into.kind),
        (LeafKind::State(_), LeafKind::State(_)) | (LeafKind::Disposition(_), LeafKind::Disposition(_))
    )
}

fn merge_weight(dst_belief: &mut Belief, dst_salience: &mut Salience, src_belief: &Belief, src_salience: &Salience) {
    dst_belief.support += src_belief.support;
    dst_belief.contradict += src_belief.contradict;
    dst_belief.last_event_at = dst_belief.last_event_at.max(src_belief.last_event_at);
    dst_salience.importance = dst_salience.importance.max(src_salience.importance);
    dst_salience.retrieval_count += src_salience.retrieval_count;
    dst_salience.last_retrieved_at = dst_salience.last_retrieved_at.max(src_salience.last_retrieved_at);
}

fn apply_merge_leaf(graph: &mut Graph, from: String, into: String, now: u64, traces: &mut Vec<String>) {
    let valid = match (graph.leaves.get(&from), graph.leaves.get(&into)) {
        (Some(src), Some(dst)) => from != into && mergeable(src, dst),
        _ => false,
    };
    if !valid {
        traces.push(format!("⚠ merge_leaves [{from}] → [{into}] invalid (missing, same id, a rule, or species mismatch); skipped"));
        return;
    }
    let src = graph.leaves.remove(&from).expect("checked above");
    let dst = graph.leaves.get_mut(&into).expect("checked above");
    match (&mut dst.kind, src.kind) {
        (LeafKind::State(d), LeafKind::State(s)) => {
            merge_weight(&mut d.belief, &mut d.salience, &s.belief, &s.salience);
            d.history.extend(s.history);
            d.history.sort_by_key(|h| h.superseded_at);
        }
        (LeafKind::Disposition(d), LeafKind::Disposition(s)) => {
            merge_weight(&mut d.belief, &mut d.salience, &s.belief, &s.salience);
            // Replay the combined trajectory so the axis stays derived, not blended.
            d.nudges.extend(s.nudges);
            d.nudges.sort_by_key(|n| n.at);
            d.axis = d.nudges.iter().fold(0.0, |a, n| belief::nudge_axis(a, n.dir));
        }
        _ => unreachable!("mergeable read these kinds"),
    }
    for e in src.evidence {
        if !dst.evidence.contains(&e) {
            dst.evidence.push(e);
        }
    }
    let mut homes = dst.parents.clone();
    homes.extend(src.parents);
    dst.created_at = dst.created_at.min(src.created_at);
    dst.occurred_at = match (dst.occurred_at, src.occurred_at) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    };
    dst.updated_at = now;
    let homes = graph.antichain(homes);
    graph.leaves.get_mut(&into).expect("checked above").parents = homes;
    traces.push(format!("⊕ merged [{}] into [{}] (absorbed: {})", from, into, src.text));
}

fn apply_merge_distillant(graph: &mut Graph, from: String, into: String, traces: &mut Vec<String>) {
    if from == into
        || !graph.distillants.contains_key(&from)
        || !graph.distillants.contains_key(&into)
        || graph.distillants[&from].tree != graph.distillants[&into].tree
    {
        traces.push(format!("⚠ merge_distillants {from} → {into} invalid (missing, same id, or tree mismatch); skipped"));
        return;
    }
    let src = graph.distillants.remove(&from).expect("checked above");
    for leaf in graph.leaves.values_mut() {
        if leaf.parents.contains(&from) {
            leaf.parents.retain(|p| *p != from);
            if !leaf.parents.contains(&into) {
                leaf.parents.push(into.clone());
            }
        }
    }
    let child_ids: Vec<String> = graph
        .distillants
        .values()
        .filter(|m| m.parents.contains(&from))
        .map(|m| m.id.clone())
        .collect();
    for c in child_ids {
        // Re-homing a child that `into` descends from would create a cycle —
        // the child just loses the merged parent instead.
        let cycles = c == into || graph.is_ancestor(&c, &into);
        let child = graph.distillants.get_mut(&c).expect("from child list");
        child.parents.retain(|p| *p != from);
        if !cycles && !child.parents.contains(&into) {
            child.parents.push(into.clone());
        } else if cycles {
            traces.push(format!("⚠ child {c} of {from} not re-homed under {into} (cycle)"));
        }
    }
    // The absorbed distillant's vocabulary keeps routing here.
    let mut terms = src.routing.clone();
    terms.push(src.label.clone());
    if let Some(seg) = src.id.rsplit('/').next() {
        terms.push(seg.replace('-', " "));
    }
    add_routing(graph, &into, &terms, traces);
    if let Some(dst) = graph.distillants.get_mut(&into) {
        dst.misc_count += src.misc_count;
    }
    for e in &mut graph.episodes {
        if e.tags.contains(&from) {
            e.tags.retain(|t| *t != from);
            if !e.tags.contains(&into) {
                e.tags.push(into.clone());
            }
        }
    }
    traces.push(format!("⊕ merged distillant {from} into {into}"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::DispositionKind;
    use crate::routing::RoutingTable;

    fn disposition_of<'g>(g: &'g Graph, id: &str) -> &'g DispositionKind {
        let LeafKind::Disposition(d) = &g.leaves[id].kind else { panic!("{id} is not a disposition") };
        d
    }

    fn rule_op(id: &str, text: &str, relation: RuleRelation, target: &str, instruction: &str) -> Op {
        Op::Rule(RuleOp {
            id: id.into(),
            text: text.into(),
            relation,
            target: target.into(),
            instruction: instruction.into(),
            occurred_at: None,
        })
    }

    fn instruction(id: &str, utterance: &str, paraphrase: &str) -> Instruction {
        Instruction { id: id.into(), utterance: utterance.into(), paraphrase: paraphrase.into() }
    }

    #[test]
    fn a_rule_is_kept_superseded_and_retracted_by_id() {
        let mut g = Graph::seed();
        let applied = apply_ops(
            &mut g,
            vec![
                Op::Episode { text: "Asked for five-line replies.".into(), tags: vec![], occurred_at: None },
                rule_op("five-lines", "keep replies to five lines", RuleRelation::Novel, "", "i-1"),
            ],
            1_000,
        );
        assert_eq!(
            applied.rules,
            vec![RuleOutcome {
                instruction: Some("i-1".into()),
                effect: RuleEffect::Kept { id: "five-lines".into(), text: "keep replies to five lines".into() }
            }]
        );
        let rule = &g.leaves["five-lines"];
        assert!(rule.is_rule() && rule.parents.is_empty());
        assert_eq!(rule.evidence.len(), 1, "the batch's episode is evidence");
        let LeafKind::Rule(kind) = &rule.kind else { panic!() };
        assert_eq!(kind.instruction.as_deref(), Some("i-1"));

        let applied = apply_ops(
            &mut g,
            vec![rule_op("five-lines", "keep replies to three lines", RuleRelation::Supersedes, "five-lines", "i-2")],
            2_000,
        );
        assert_eq!(
            applied.rules[0].effect,
            RuleEffect::Superseded { id: "five-lines".into(), from: "keep replies to five lines".into(), to: "keep replies to three lines".into() }
        );
        let rule = &g.leaves["five-lines"];
        assert_eq!(rule.text, "keep replies to three lines");
        assert_eq!(rule.kind.history()[0].value, "keep replies to five lines");
        assert_eq!(rule.updated_at, 2_000);
        let LeafKind::Rule(kind) = &rule.kind else { panic!() };
        assert_eq!(kind.instruction.as_deref(), Some("i-2"), "the rule answers the instruction that last changed it");

        // Same words again: a duplicate, whichever relation names it.
        let applied = apply_ops(&mut g, vec![rule_op("five-lines", "keep replies to three lines", RuleRelation::Novel, "", "")], 2_500);
        assert!(matches!(applied.rules[0].effect, RuleEffect::Duplicate { .. }), "{:?}", applied.rules);
        assert_eq!(applied.rules[0].instruction, None);
        assert!(g.leaves["five-lines"].kind.history().len() == 1, "no change");

        let applied = apply_ops(&mut g, vec![rule_op("five-lines", "", RuleRelation::Retract, "five-lines", "i-3")], 3_000);
        assert_eq!(
            applied.rules[0].effect,
            RuleEffect::Retracted { id: "five-lines".into(), text: "keep replies to three lines".into() }
        );
        assert!(g.rules().is_empty(), "out of force");
        assert!(g.leaves["five-lines"].is_withdrawn_rule(), "on record");
        assert_eq!(g.episodes.len(), 1, "a retract erases nothing");
        assert!(g.distillants.values().all(|d| d.forgotten_at == 0), "no distillant held it, none is stamped");
        let applied = apply_ops(&mut g, vec![rule_op("five-lines", "", RuleRelation::Retract, "five-lines", "i-4")], 3_500);
        assert!(matches!(applied.rules[0].effect, RuleEffect::Retracted { .. }), "withdrawing it again answers the same: {:?}", applied.rules);
        assert_eq!(applied.rules[0].instruction.as_deref(), Some("i-4"));
        let applied = apply_ops(&mut g, vec![rule_op("never-was", "", RuleRelation::Retract, "never-was", "i-5")], 3_600);
        assert!(applied.rules.is_empty(), "retracting nothing answers nothing");
        assert!(applied.traces.iter().any(|t| t.contains("retract of unknown rule")), "{:?}", applied.traces);
    }

    #[test]
    fn a_withdrawn_rule_is_out_of_every_prompt_and_on_record() {
        let mut g = Graph::seed();
        apply_ops(&mut g, vec![rule_op("no-emoji", "stop using emoji", RuleRelation::Novel, "", "")], 1_000);
        apply_ops(&mut g, vec![rule_op("five-lines", "keep replies to five lines", RuleRelation::Novel, "", "")], 1_000);
        apply_ops(&mut g, vec![rule_op("no-emoji", "", RuleRelation::Retract, "no-emoji", "")], 2_000);

        let pinned = projection::render_rules(&g).expect("one rule stands");
        assert!(pinned.contains("[five-lines]") && !pinned.contains("no-emoji"), "the pinned block lists rules in force only: {pinned}");
        assert_eq!(projection::render_rules(&g), projection::render_rules(&g), "and stays byte-stable");
        let open = projection::render_open(&g, projection::RULES, 3_000 + 86_400 * 3);
        assert!(open.contains("- [five-lines] keep replies to five lines (given"), "{open}");
        assert!(open.contains("Withdrawn"), "{open}");
        assert!(open.contains("- [no-emoji] stop using emoji (withdrawn 3d ago)"), "{open}");
        assert!(open.find("[five-lines]").unwrap() < open.find("[no-emoji]").unwrap(), "in force before withdrawn");
        let harvester = build_user_message(&g, &HarvestInput::turn("hi", "hello"), 3_000, DirectoryScope::Selective);
        let rules_section = harvester.split("# Standing rules").nth(1).unwrap().split("# Profile axes").next().unwrap();
        assert!(rules_section.contains("- [five-lines] keep replies to five lines\nWithdrawn (out of force"), "{rules_section}");
        assert!(rules_section.contains("- [no-emoji] stop using emoji"), "the harvester sees withdrawn rules by id: {rules_section}");
        assert!(crate::consolidate::stats(&g).contains("(1 rules, 1 withdrawn)"), "{}", crate::consolidate::stats(&g));

        apply_ops(&mut g, vec![rule_op("no-emoji", "", RuleRelation::Retract, "no-emoji", "")], 2_500);
        assert_eq!(g.withdrawn_rules().len(), 1, "no duplicate on record");
        let gone = g.forget("no-emoji", 4_000).expect("on record");
        assert_eq!(gone.leaves, vec!["no-emoji"], "the user's forget erases a withdrawn rule");
        assert!(g.withdrawn_rules().is_empty());
    }

    #[test]
    fn giving_a_withdrawn_rule_again_reinstates_it_under_its_id() {
        for relation in [RuleRelation::Novel, RuleRelation::Supersedes, RuleRelation::Duplicate] {
            let mut g = Graph::seed();
            apply_ops(&mut g, vec![rule_op("no-emoji", "stop using emoji", RuleRelation::Novel, "", "i-1")], 1_000);
            apply_ops(&mut g, vec![rule_op("no-emoji", "", RuleRelation::Retract, "no-emoji", "i-2")], 2_000);
            let target = if relation == RuleRelation::Novel { "" } else { "no-emoji" };
            let ep = Op::Episode { text: "asked for no emoji again".into(), tags: vec![], occurred_at: None };
            let applied = apply_ops(&mut g, vec![ep, rule_op("no-emoji", "no emoji, please", relation, target, "i-3")], 3_000);
            assert_eq!(
                applied.rules[0].effect,
                RuleEffect::Reinstated { id: "no-emoji".into(), text: "no emoji, please".into() },
                "{relation:?}"
            );
            assert_eq!(applied.rules[0].instruction.as_deref(), Some("i-3"), "{relation:?}");
            let rule = &g.leaves["no-emoji"];
            assert!(rule.is_standing_rule(), "{relation:?}: in force again");
            assert_eq!(rule.text, "no emoji, please");
            assert_eq!(rule.kind.history()[0].value, "stop using emoji", "{relation:?}: the withdrawn words are history");
            assert_eq!(rule.evidence.len(), 1, "{relation:?}: cites the turn that gave it again");
            assert_eq!(rule.updated_at, 3_000);
            let LeafKind::Rule(kind) = &rule.kind else { unreachable!() };
            assert_eq!(kind.instruction.as_deref(), Some("i-3"));
            assert_eq!(g.rules().len(), 1, "{relation:?}: one rule, not a second");
            assert!(projection::render_rules(&g).unwrap().contains("- [no-emoji] no emoji, please (was: stop using emoji)"));
        }
        // The same words again: reinstated with no history entry.
        let mut g = Graph::seed();
        apply_ops(&mut g, vec![rule_op("no-emoji", "stop using emoji", RuleRelation::Novel, "", "")], 1_000);
        apply_ops(&mut g, vec![rule_op("no-emoji", "", RuleRelation::Retract, "no-emoji", "")], 2_000);
        apply_ops(&mut g, vec![rule_op("no-emoji", "stop using emoji", RuleRelation::Novel, "", "")], 3_000);
        assert!(g.leaves["no-emoji"].is_standing_rule());
        assert!(g.leaves["no-emoji"].kind.history().is_empty(), "same words: nothing superseded");
    }

    #[test]
    fn document_import_preserves_facts_dates_and_each_independent_rule() {
        let documents = vec![Document {
            name: "USER.md".into(),
            text: "Lives in Lisbon. Opened North studio on 2024-01-02. Never use exclamation marks. Keep replies short.".into(),
            written_at: Some(86_400 * 100),
        }];
        let input = HarvestInput { documents: &documents, ..HarvestInput::turn("", "") };
        let mut graph = Graph::seed();
        let now = 86_400 * 500;
        let prompt = render_harvest_prompt(&graph, &input, now);
        assert!(prompt.contains("# Documents to import"));
        assert!(!prompt.contains("# Instructions to resolve"));
        assert!(prompt.contains("EACH independent explicit standing instruction"));
        assert!(prompt.contains("## USER.md (written 1.1y ago)"), "the harvester reads the words as of when they were written: {prompt}");
        assert!(prompt.contains("memory newer than the document outranks it"));
        // The contract rides once, under the heading, before the first document.
        let section = prompt.split("# Documents to import\n").nth(1).unwrap();
        let contract = section.split("## USER.md").next().unwrap();
        assert!(contract.contains("A retired document is read once and never shown again"), "{contract}");
        assert!(contract.contains("Voice, tone, register and manner lines"), "{contract}");
        assert!(contract.contains("are standing instructions: one rule per independent line"), "{contract}");
        assert!(contract.contains("never persona prose to skip, and never a disposition to infer"), "{contract}");
        assert!(contract.contains("is ONE rule whose text carries the trigger and every step verbatim, in order, however long"), "{contract}");
        assert!(contract.contains("A rule that names the flow without its steps is a distortion, not a summary"), "{contract}");
        assert!(contract.contains("is a rule, one per default, never folded into a neighbouring rule"), "{contract}");
        assert!(contract.contains("Numbers, units, prices, dates, names, identifiers and qualifiers ride verbatim"), "{contract}");
        assert!(contract.contains("two prices in one line are two states"), "{contract}");
        // The system contract and the schema say the same of a procedure, for
        // a flow dictated in conversation as much as one imported.
        let system = prompt.split("# Output schema").next().unwrap();
        assert!(system.contains("A procedure's steps are the instruction, not detail to trim"), "{system}");
        assert!(prompt.contains("a procedure keeps its trigger and every step, in order"), "the schema's rule text says so too");
        let ops = parse_ops(&json!({
            "states": [{"id": "city", "distillants": ["reg.home"], "text": "Lives in Lisbon", "relation": "novel", "target": "", "importance": 0.7, "aliases": ["Lisbon"]}],
            "episodes": [{"text": "Opened North studio on 2024-01-02", "tags": [], "occurred_at": 1704153600}],
            "rules": [
                {"id": "punctuation", "text": "Never use exclamation marks", "relation": "novel", "target": ""},
                {"id": "length", "text": "Keep replies short", "relation": "novel", "target": ""}
            ]
        }).to_string()).unwrap();
        let applied = apply_ops(&mut graph, ops, 1_000);
        assert_eq!(graph.leaves["city"].text, "Lives in Lisbon");
        assert_eq!(graph.rules().len(), 2);
        assert_eq!(applied.rules.len(), 2);
        assert_eq!(graph.episodes.len(), 1, "the dated experience is the only episode: the import itself is not an event");
        assert_eq!(graph.episodes[0].occurred_at, Some(1704153600));
        assert!(resolve(&[], &applied).is_empty(), "imports have no Rule-only verdict");

        let undated = vec![Document { name: "notes.txt".into(), text: "A quiet day.".into(), written_at: None }];
        let prompt = render_harvest_prompt(&graph, &HarvestInput { documents: &undated, ..HarvestInput::turn("", "") }, 1_000);
        assert!(prompt.contains("## notes.txt (written at an unknown time)"), "{prompt}");
    }

    /// A voice document, a workflows document and a preferences document
    /// of the shapes a host retires: a manner section of one-line
    /// instructions, a scripted flow with numbered steps beside a one-line
    /// rule that points at it, a research default, priced settings, a
    /// preference with a qualifier.
    fn retired_documents() -> Vec<Document> {
        vec![
            Document {
                name: "VOICE.md".into(),
                text: "# Voice\n## Vibe\n- Keep it short by default.\n- Match Mara's energy — she is casual, fast, uses shorthand. Don't over-formalize.\n- Humor is welcome. Warmth over polish.\n- Don't perform being an assistant. Just be present.\n".into(),
                written_at: Some(86_400 * 300),
            },
            Document {
                name: "WORKFLOWS.md".into(),
                text: "# Saved Workflows\n## Research\n- Trigger: the user asks to research a topic.\n- Steps: check the reading catalog first before any web search.\n- Defaults: prefer newsletter sources as the primary research tool.\n\n## Plant Tracker — Demo Conversation Flow\n- Trigger: the user says \"make a plant tracker\".\n- This is a SCRIPTED DEMO FLOW. Follow it exactly, in order:\n  Step 1 — Assistant asks: \"which plants, and how often do you water them?\"\n  Step 2 — User names the plants; assistant asks whether to send a reminder by email.\n  Step 3 — User says yes; assistant proposes a schedule and asks to confirm.\n  Step 4 — User confirms; assistant creates the standing order and renders the plant journal card in the same reply.\n- Timezone for reminders: Europe/Lisbon (7am WET)\n- Reminder service: Pingly (~$0.02 per reminder)\n".into(),
                written_at: Some(86_400 * 400),
            },
            Document {
                name: "USER.md".into(),
                text: "# Preferences\n- Wants short answers — three sentences max, even for reflective questions.\n- Prices seen: image generation runs $0.04 per image on Flux; a voice call costs about $0.84 per minute on Ringo.\n".into(),
                written_at: Some(86_400 * 450),
            },
        ]
    }

    #[test]
    fn a_retired_document_loses_nothing_memory_can_hold() {
        let documents = retired_documents();
        let input = HarvestInput { documents: &documents, ..HarvestInput::turn("", "") };
        let mut graph = Graph::seed();
        let prompt = render_harvest_prompt(&graph, &input, 86_400 * 500);
        assert!(prompt.contains("## VOICE.md (written 6mo ago)"), "{prompt}");
        assert!(prompt.contains("Step 4 — User confirms"), "the flow rides whole into the prompt: {prompt}");
        // What the contract asks for, in the ops it asks for: every manner
        // line its own rule, the flow one rule with every step, the default
        // its own rule, every price verbatim, the qualifier kept.
        let flow = "When the user says \"make a plant tracker\", follow this scripted demo flow exactly, in order:\nStep 1 — Assistant asks: \"which plants, and how often do you water them?\"\nStep 2 — User names the plants; assistant asks whether to send a reminder by email.\nStep 3 — User says yes; assistant proposes a schedule and asks to confirm.\nStep 4 — User confirms; assistant creates the standing order and renders the plant journal card in the same reply.";
        let ops = parse_ops(&json!({
            "rules": [
                {"id": "short-by-default", "text": "Keep it short by default", "relation": "novel", "target": ""},
                {"id": "match-energy", "text": "Match Mara's energy — she is casual, fast, uses shorthand. Don't over-formalize", "relation": "novel", "target": ""},
                {"id": "humor-welcome", "text": "Humor is welcome. Warmth over polish", "relation": "novel", "target": ""},
                {"id": "be-present", "text": "Don't perform being an assistant. Just be present", "relation": "novel", "target": ""},
                {"id": "research-catalog-first", "text": "When asked to research a topic, check the reading catalog first before any web search", "relation": "novel", "target": ""},
                {"id": "research-newsletters", "text": "Prefer newsletter sources as the primary research tool", "relation": "novel", "target": ""},
                {"id": "plant-tracker-demo-flow", "text": flow, "relation": "novel", "target": ""},
                {"id": "three-sentences", "text": "Wants short answers — three sentences max, even for reflective questions", "relation": "novel", "target": ""}
            ],
            "states": [
                {"id": "plant-reminder-timezone", "distillants": ["life"], "text": "Plant reminders go out in Europe/Lisbon time, at 7am WET", "relation": "novel", "target": "", "importance": 0.6, "aliases": []},
                {"id": "pingly-price", "distillants": ["money"], "text": "Pingly reminders cost ~$0.02 per reminder", "relation": "novel", "target": "", "importance": 0.6, "aliases": ["Pingly"]},
                {"id": "flux-price", "distillants": ["money"], "text": "Image generation runs $0.04 per image on Flux", "relation": "novel", "target": "", "importance": 0.6, "aliases": ["Flux"]},
                {"id": "ringo-price", "distillants": ["money"], "text": "A voice call costs about $0.84 per minute on Ringo", "relation": "novel", "target": "", "importance": 0.6, "aliases": ["Ringo"]}
            ]
        }).to_string()).unwrap();
        let applied = apply_ops(&mut graph, ops, 86_400 * 500);
        assert_eq!(applied.rules.len(), 8, "{:?}", applied.traces);
        assert_eq!(graph.rules().len(), 8, "one rule per manner line, per default, per flow, per preference");
        assert_eq!(graph.leaves["plant-tracker-demo-flow"].text, flow, "the rule holds the flow's every step, unchanged");
        assert_eq!(graph.leaves["three-sentences"].text, "Wants short answers — three sentences max, even for reflective questions");
        assert_eq!(graph.leaves["flux-price"].text, "Image generation runs $0.04 per image on Flux");
        assert_eq!(graph.leaves["ringo-price"].text, "A voice call costs about $0.84 per minute on Ringo");
        assert_eq!(graph.leaves["pingly-price"].text, "Pingly reminders cost ~$0.02 per reminder");
        // The pinned tier carries the whole flow, so the rule can be
        // followed from the block alone; and the harvester's own pinned
        // rules show it whole too, to supersede by id rather than re-mint.
        let pinned = projection::render_rules(&graph).unwrap();
        for step in ["Step 1 — Assistant asks", "Step 2 — User names the plants", "Step 3 — User says yes", "Step 4 — User confirms"] {
            assert!(pinned.contains(step), "{step} rides pinned: {pinned}");
        }
        assert!(pinned.contains("- [match-energy] Match Mara's energy"), "{pinned}");
        assert!(render_pinned_rules(&graph).contains("Step 4 — User confirms"));
        assert!(graph.episodes.is_empty(), "nothing here happened; the import is not an event");
    }

    #[test]
    fn a_document_routes_on_its_words_not_the_section_framing() {
        let mut g = Graph::seed();
        let ops = vec![Op::Distillant {
            id: "machinery".into(),
            tree: Tree::Registry,
            label: "machinery".into(),
            line: "how memory is put together".into(),
            routing: vec!["episodes".into(), "dispositions".into(), "standing instruction".into()],
            parents: vec![],
        }];
        apply_ops(&mut g, ops, 1_000);
        let documents = vec![Document { name: "USER.md".into(), text: "Lives in Lisbon.".into(), written_at: None }];
        let input = HarvestInput { documents: &documents, ..HarvestInput::turn("", "") };
        let prompt = build_user_message(&g, &input, 1_000, DirectoryScope::Selective);
        assert!(prompt.contains("# Documents to import"), "{prompt}");
        let directory = prompt.split("# Standing rules").next().unwrap();
        assert!(
            !directory.contains("how memory is put together"),
            "the section's own framing must not open a branch: {directory}"
        );
    }

    fn strength_of(g: &Graph, id: &str) -> f32 {
        belief::strength(g.leaves[id].kind.belief().expect("weighed leaf"))
    }

    /// The host's own record of an import, pushed into the batch as an
    /// ordinary episode: what the batch's leaves then cite.
    fn host_import_episode(name: &str) -> Op {
        Op::Episode { text: format!("Imported notes from {name}"), tags: vec![], occurred_at: None }
    }

    #[test]
    fn a_duplicate_only_import_is_forgotten_with_the_fact_it_restated() {
        let mut g = Graph::seed();
        apply_ops(&mut g, vec![state_op("city", "Lives in Lisbon", Relation::Novel, "")], 1_000);
        let before = g.leaves["city"].clone();
        // Two ops in one batch name the same fact: the host's episode is cited once.
        let ops = vec![
            host_import_episode("USER.md"),
            state_op("city-again", "Lives in Lisbon", Relation::Duplicate, "city"),
            state_op("city-thrice", "Lives in Lisbon", Relation::Duplicate, "city"),
        ];
        apply_ops(&mut g, ops, 2_000);
        assert_eq!(g.episodes.len(), 1, "the host's episode is the only one");
        let source = g.episodes[0].id.clone();
        let after = &g.leaves["city"];
        assert_eq!(after.evidence, vec![source.clone()], "a duplicate cites its source, once");
        assert_eq!(after.text, before.text);
        assert_eq!(after.updated_at, before.updated_at, "no clock moves on a duplicate");
        assert_eq!(after.occurred_at, before.occurred_at);
        assert_eq!(strength_of(&g, "city"), belief::strength(before.kind.belief().unwrap()), "no weight moves on a duplicate");
        assert!(!g.leaves.contains_key("city-again") && !g.leaves.contains_key("city-thrice"));

        apply_ops(&mut g, vec![Op::Forget { target: "city".into() }], 3_000);
        assert!(!g.leaves.contains_key("city"));
        assert!(g.episodes.is_empty(), "the source was evidence for nothing else; it goes with the fact");
    }

    #[test]
    fn a_duplicate_only_import_cites_the_rule_it_restated() {
        for relation in [RuleRelation::Duplicate, RuleRelation::Novel] {
            let mut g = Graph::seed();
            apply_ops(&mut g, vec![rule_op("five-lines", "keep replies to five lines", RuleRelation::Novel, "", "i-1")], 1_000);
            let before = g.leaves["five-lines"].clone();
            let target = if relation == RuleRelation::Duplicate { "five-lines" } else { "" };
            let ops = vec![host_import_episode("USER.md"), rule_op("five-lines", "keep replies to five lines", relation, target, "")];
            let applied = apply_ops(&mut g, ops, 2_000);
            assert!(matches!(applied.rules[0].effect, RuleEffect::Duplicate { .. }), "{relation:?}: {:?}", applied.rules);
            assert_eq!(g.episodes.len(), 1);
            let source = g.episodes[0].id.clone();
            let after = &g.leaves["five-lines"];
            assert_eq!(after.evidence, vec![source], "{relation:?}: the duplicate cites its source");
            assert_eq!(after.text, before.text);
            assert_eq!(after.updated_at, before.updated_at, "{relation:?}: no clock moves");
            assert_eq!(after.occurred_at, before.occurred_at);
            assert_eq!(after.kind.history().len(), 0);

            apply_ops(&mut g, vec![rule_op("five-lines", "", RuleRelation::Retract, "five-lines", "")], 3_000);
            assert!(g.leaves["five-lines"].is_withdrawn_rule());
            assert_eq!(g.episodes.len(), 1, "{relation:?}: a retract withdraws the rule, not the record of the import");
        }
    }

    #[test]
    fn a_source_cited_by_a_surviving_leaf_outlives_the_forgotten_one() {
        let mut g = Graph::seed();
        apply_ops(
            &mut g,
            vec![
                state_op("city", "Lives in Lisbon", Relation::Novel, ""),
                rule_op("five-lines", "keep replies to five lines", RuleRelation::Novel, "", ""),
            ],
            1_000,
        );
        let ops = vec![
            host_import_episode("USER.md"),
            state_op("city-again", "Lives in Lisbon", Relation::Duplicate, "city"),
            rule_op("five-lines", "keep replies to five lines", RuleRelation::Duplicate, "five-lines", ""),
        ];
        apply_ops(&mut g, ops, 2_000);
        let source = g.episodes[0].id.clone();
        assert_eq!(g.leaves["city"].evidence, vec![source.clone()]);
        assert_eq!(g.leaves["five-lines"].evidence, vec![source.clone()]);

        apply_ops(&mut g, vec![Op::Forget { target: "city".into() }], 3_000);
        assert!(g.episodes.iter().any(|e| e.id == source), "the rule still cites the source");
        apply_ops(&mut g, vec![Op::Forget { target: "five-lines".into() }], 4_000);
        assert!(g.episodes.is_empty(), "the user forgot its last citer; so does it");
    }

    #[test]
    fn nothing_after_a_forget_cites_what_it_took() {
        let mut g = Graph::seed();
        apply_ops(&mut g, vec![state_op("city", "Lives in Lisbon", Relation::Novel, "")], 1_000);
        let ops = vec![
            Op::Episode { text: "moved on".into(), tags: vec![], occurred_at: None },
            Op::Reinforce { target: "city".into() },
            Op::Forget { target: "city".into() },
            state_op("town", "Lives in Porto", Relation::Novel, ""),
        ];
        apply_ops(&mut g, ops, 2_000);
        assert!(g.episodes.is_empty(), "the forgotten fact was the episode's only citer");
        assert!(g.leaves["town"].evidence.is_empty(), "nothing cites an episode the stream no longer holds");
    }

    #[test]
    fn the_open_id_of_the_rules_is_no_distillant() {
        let mut g = Graph::seed();
        let ops = vec![
            Op::Distillant {
                id: projection::RULES.into(),
                tree: Tree::Registry,
                label: "rules".into(),
                line: "house rules".into(),
                routing: vec![],
                parents: vec![],
            },
            Op::State {
                id: "no-shoes".into(),
                distillants: vec![projection::RULES.into(), "money".into()],
                text: "No shoes indoors.".into(),
                relation: Relation::Novel,
                target: "".into(),
                importance: 0.5,
                aliases: vec![],
                occurred_at: None,
            },
            Op::Move { leaf: "no-shoes".into(), parents: vec![projection::RULES.into()] },
        ];
        let applied = apply_ops(&mut g, ops, 1_000);
        assert!(!g.distillants.contains_key(projection::RULES), "{:?}", applied.traces);
        assert_eq!(g.leaves["no-shoes"].parents, vec!["money"], "the leaf hangs under what is a distillant");
        assert_eq!(applied.traces.iter().filter(|t| t.contains("not a distillant")).count(), 3, "{:?}", applied.traces);
        assert!(applied.traces.iter().any(|t| t.contains("move of [no-shoes] with no parents")), "{:?}", applied.traces);
        assert!(projection::render_open(&g, projection::RULES, 1_000).contains("standing instructions"));
    }

    #[test]
    fn a_state_novel_collision_cites_the_batch_either_way() {
        for text in ["Rent is $2,200/mo.", "Rent is $2,400/mo."] {
            let mut g = Graph::seed();
            apply_ops(&mut g, vec![state_op("rent", "Rent is $2,200/mo.", Relation::Novel, "")], 1_000);
            let ep = Op::Episode { text: "talked rent".into(), tags: vec![], occurred_at: None };
            apply_ops(&mut g, vec![ep, state_op("rent", text, Relation::Novel, "")], 2_000);
            assert_eq!(g.leaves["rent"].evidence.len(), 1, "{text}: the collision cites the batch");
            assert_eq!(g.leaves["rent"].text, text);
        }
    }

    #[test]
    fn a_missing_target_fallback_never_lands_on_an_occupied_id() {
        let mut g = Graph::seed();
        apply_ops(
            &mut g,
            vec![
                Op::Episode { text: "seeded".into(), tags: vec![], occurred_at: None },
                state_op("rent", "Rent is $2,200/mo.", Relation::Novel, ""),
                Op::Disposition { id: "spend".into(), distillants: vec!["money-style".into()], text: "keeps spending tight".into(), dir: 1, note: "n".into(), importance: 0.5 },
                rule_op("five-lines", "keep replies to five lines", RuleRelation::Novel, "", ""),
            ],
            1_000,
        );
        let snapshot = g.clone();
        let applied = apply_ops(
            &mut g,
            vec![
                // A state landing on a rule's id, and on a state's id.
                state_op("five-lines", "five", Relation::Supports, "nope"),
                state_op("rent", "Rent is $9/mo.", Relation::Supersedes, "nope"),
                // A rule landing on a state's, a disposition's, and a rule's id.
                rule_op("rent", "rent words", RuleRelation::Supersedes, "nope", ""),
                rule_op("spend", "spend words", RuleRelation::Duplicate, "nope", ""),
                rule_op("five-lines", "other words", RuleRelation::Supersedes, "nope", ""),
                rule_op("five-lines", "other words", RuleRelation::Novel, "nope", ""),
            ],
            2_000,
        );
        assert!(applied.rules.is_empty(), "{:?}", applied.rules);
        let refusals = applied.traces.iter().filter(|t| t.contains("would overwrite the")).count();
        assert_eq!(refusals, 6, "{:?}", applied.traces);
        for id in ["rent", "spend", "five-lines"] {
            let (was, is) = (&snapshot.leaves[id], &g.leaves[id]);
            assert_eq!((&is.text, is.species(), is.updated_at, &is.evidence), (&was.text, was.species(), was.updated_at, &was.evidence), "{id} untouched");
            assert_eq!(is.kind.history().len(), was.kind.history().len(), "{id} untouched");
        }
        assert_eq!(g.leaves.len(), snapshot.leaves.len(), "nothing landed anywhere");

        // With the target present, same-species supersession and duplicates
        // resolve on the target as before.
        let applied = apply_ops(
            &mut g,
            vec![
                state_op("rent-new", "Rent is $2,400/mo.", Relation::Supersedes, "rent"),
                rule_op("three-lines", "keep replies to three lines", RuleRelation::Supersedes, "five-lines", ""),
                rule_op("again", "keep replies to three lines", RuleRelation::Duplicate, "five-lines", ""),
            ],
            3_000,
        );
        assert_eq!(g.leaves["rent"].text, "Rent is $2,400/mo.");
        assert_eq!(g.leaves["five-lines"].text, "keep replies to three lines");
        assert!(!g.leaves.contains_key("rent-new") && !g.leaves.contains_key("three-lines") && !g.leaves.contains_key("again"));
        assert!(matches!(applied.rules[0].effect, RuleEffect::Superseded { .. }));
        assert!(matches!(applied.rules[1].effect, RuleEffect::Duplicate { .. }));
    }

    #[test]
    fn instructions_resolve_in_order_and_unanswered_ones_are_not_rules() {
        let mut g = Graph::seed();
        let pending = vec![
            instruction("i-1", "keep it to five lines", "reply in at most five lines"),
            instruction("i-2", "actually, forget the length thing", "drop the length rule"),
            instruction("i-3", "book the usual table tonight", "one-off reservation"),
        ];
        let prompt = build_user_message(&g, &HarvestInput { instructions: &pending, ..HarvestInput::turn("…", "…") }, 1_000, DirectoryScope::Full);
        assert!(prompt.contains("# Instructions to resolve"), "{prompt}");
        assert!(prompt.contains("- [i-2] user said: \"actually, forget the length thing\" — read as: drop the length rule"), "{prompt}");
        assert!(render_harvest_prompt(&g, &HarvestInput::turn("…", "…"), 1_000).contains("16. Rules:"));

        let raw = json!({
            "rules": [
                {"id": "five-lines", "text": "keep it to five lines", "relation": "novel", "target": "", "instruction": "i-1", "occurred_at": null},
                {"id": "five-lines", "text": "", "relation": "retract", "target": "five-lines", "instruction": "i-2", "occurred_at": null}
            ]
        })
        .to_string();
        let ops = parse_ops(&raw).unwrap();
        assert_eq!(ops.len(), 2);
        let applied = apply_ops(&mut g, ops, 1_000);
        let resolved = resolve(&pending, &applied);
        assert_eq!(resolved[0], ("i-1".into(), Resolution::Rule(RuleEffect::Kept { id: "five-lines".into(), text: "keep it to five lines".into() })));
        assert_eq!(resolved[1], ("i-2".into(), Resolution::Rule(RuleEffect::Retracted { id: "five-lines".into(), text: "keep it to five lines".into() })));
        assert_eq!(resolved[2], ("i-3".into(), Resolution::NotARule));
        assert!(g.rules().is_empty(), "kept then retracted in the order given");
    }

    #[test]
    fn a_rule_id_is_refused_by_every_weighing_op_and_hides_from_every_walk() {
        let mut g = Graph::seed();
        apply_ops(
            &mut g,
            vec![
                state_op("rent-amount", "Rent is $2,200/mo.", Relation::Novel, ""),
                rule_op("ask-first", "ask before any spend over $20", RuleRelation::Novel, "", ""),
            ],
            1_000,
        );
        let traces = apply_ops(
            &mut g,
            vec![
                state_op("x", "twenty", Relation::Supports, "ask-first"),
                Op::Disposition { id: "ask-first".into(), distillants: vec!["money-style".into()], text: "x".into(), dir: 1, note: "n".into(), importance: 0.5 },
                Op::Reinforce { target: "ask-first".into() },
                Op::Move { leaf: "ask-first".into(), parents: vec!["money".into()] },
                Op::MergeLeaf { from: "ask-first".into(), into: "rent-amount".into() },
                rule_op("rent-amount", "rent words", RuleRelation::Novel, "", ""),
            ],
            2_000,
        )
        .traces;
        for refused in ["state op targets the rule", "nudge targets the rule", "reinforce on rule", "move of rule", "a rule, or species mismatch", "rule op targets the state"] {
            assert!(traces.iter().any(|t| t.contains(refused)), "{refused}: {traces:?}");
        }
        let rule = &g.leaves["ask-first"];
        assert_eq!(rule.text, "ask before any spend over $20");
        assert!(rule.parents.is_empty() && rule.updated_at == 1_000, "untouched");
        assert!(g.leaves["rent-amount"].kind.belief().unwrap().support == 1);
        assert!(routing::RoutingTable::build(&g).matches("ask before any spend").is_empty(), "no routing reaches a rule");
        assert!(projection::project(&g, "spend over $20?", 3_000).opened.iter().all(|o| !o.leaf_ids.contains(&"ask-first".to_string())));
        assert!(!render_comparanda(&g, &routing::RoutingTable::build(&g), "spend over $20", 3_000).contains("ask-first"));
        assert!(render_pinned_rules(&g).contains("- [ask-first] ask before any spend over $20"));
        let stats = crate::consolidate::stats(&g);
        assert!(!stats.contains("ask-first"), "a rule builds no pressure anywhere: {stats}");
        assert!(stats.contains("(1 rules, 0 withdrawn)"), "{stats}");
        // The user's forget covers a rule like any leaf.
        let ops = parse_ops(&json!({"forgets": [{"target": "ask-first"}]}).to_string()).unwrap();
        apply_ops(&mut g, ops, 4_000);
        assert!(!g.leaves.contains_key("ask-first"));
    }

    #[test]
    fn a_rule_sections_parse_and_apply_after_dispositions() {
        let raw = json!({
            "dispositions": [{"id": "spend", "distillants": ["money-style"], "text": "keeps spending tight", "dir": 1, "note": "obs", "importance": 0.5}],
            "rules": [{"id": "ask-first", "text": "ask before any spend over $20", "relation": "novel", "target": "", "instruction": "", "occurred_at": 77}],
            "reinforces": [{"target": "spend"}]
        })
        .to_string();
        let ops = parse_ops(&raw).unwrap();
        assert!(matches!(ops[0], Op::Disposition { .. }));
        assert!(matches!(&ops[1], Op::Rule(RuleOp { occurred_at: Some(77), .. })));
        assert!(matches!(ops[2], Op::Reinforce { .. }));
        assert_eq!(ops_schema()["required"].as_array().unwrap().iter().filter(|s| s == &"rules").count(), 1);
        let mut g = Graph::seed();
        let applied = apply_ops(&mut g, ops, 1_000);
        assert_eq!(applied.rules.len(), 1);
        assert_eq!(g.leaves["ask-first"].occurred_at, Some(77));
    }

    fn state_op(id: &str, text: &str, relation: Relation, target: &str) -> Op {
        Op::State {
            id: id.into(),
            distillants: vec!["money/rent".into()],
            text: text.into(),
            relation,
            target: target.into(),
            importance: 0.8,
            aliases: vec!["the landlord".into()],
            occurred_at: None,
        }
    }

    #[test]
    fn novel_state_auto_creates_distillant_and_wires_evidence() {
        let mut g = Graph::seed();
        let ops = vec![
            Op::Episode {
                text: "User told me their rent.".into(),
                tags: vec!["money/rent".into()],
                occurred_at: None,
            },
            state_op("rent-amount", "Rent is $2,200/mo, due the 1st.", Relation::Novel, ""),
        ];
        let traces = apply_ops(&mut g, ops, 1_000).traces;
        assert!(g.distillants.contains_key("money/rent"), "bare distillant auto-created");
        assert_eq!(g.distillants["money/rent"].parents, vec!["money"]);
        let leaf = &g.leaves["rent-amount"];
        assert_eq!(leaf.evidence.len(), 1, "episode wired as evidence");
        assert!(traces.iter().any(|t| t.contains("auto-created")));
    }

    #[test]
    fn bare_stub_has_no_line_and_announces_itself() {
        let mut g = Graph::seed();
        apply_ops(
            &mut g,
            vec![Op::State {
                id: "x-report-run".into(),
                distillants: vec!["work/x-engagement-report".into()],
                text: "Ran the X report for Mason.".into(),
                relation: Relation::Novel,
                target: "".into(),
                importance: 0.5,
                aliases: vec![],
                occurred_at: None,
            }],
            1_000,
        );
        let stub = &g.distillants["work/x-engagement-report"];
        assert!(stub.is_bare(), "an auto-created distillant carries no line of its own");
        assert_eq!(stub.label, "x engagement report");
        assert_eq!(stub.parents, vec!["work"]);
        assert!(
            render_directory_scoped(&g, DirectoryScope::Full, &routing::RoutingTable::build(&g), "").contains("work/x-engagement-report [registry] \"x engagement report\" — x engagement report (no line yet)"),
            "the harvester's directory shows the stub as unwritten, never as a line"
        );
    }

    #[test]
    fn the_selective_directory_expands_only_the_opened_branches_and_their_ancestors() {
        let mut g = Graph::seed();
        apply_ops(
            &mut g,
            vec![
                Op::Distillant {
                    id: "money/rent".into(),
                    tree: Tree::Registry,
                    label: "Rent".into(),
                    line: "Rent and landlord dealings.".into(),
                    routing: vec!["rent".into(), "landlord".into()],
                    parents: vec!["money".into()],
                },
                Op::Distillant {
                    id: "money/rent/deposit".into(),
                    tree: Tree::Registry,
                    label: "Deposit".into(),
                    line: "The deposit the landlord holds.".into(),
                    routing: vec!["deposit".into()],
                    parents: vec!["money/rent".into()],
                },
                Op::Distillant {
                    id: "work/studio".into(),
                    tree: Tree::Registry,
                    label: "Studio".into(),
                    line: "The shared studio lease.".into(),
                    routing: vec!["studio".into()],
                    parents: vec!["work".into()],
                },
            ],
            1_000,
        );
        let turn = "the landlord raised the rent";
        let table = routing::RoutingTable::build(&g);
        let neighborhood = Neighborhood::open(&g, &table, turn);
        assert!(neighborhood.on_branch.contains("money/rent") && neighborhood.on_branch.contains("money"), "{neighborhood:?}");
        assert_eq!(neighborhood.beside.iter().collect::<Vec<_>>(), vec!["money/rent/deposit"], "{neighborhood:?}");
        // A hit that sits above another hit is a level the turn is on too:
        // its other children are beside, whatever the deeper hit.
        apply_ops(
            &mut g,
            vec![Op::Distillant {
                id: "money/taxes".into(),
                tree: Tree::Registry,
                label: "Taxes".into(),
                line: "The yearly filing.".into(),
                routing: vec!["taxes".into()],
                parents: vec!["money".into()],
            }],
            1_000,
        );
        let table = routing::RoutingTable::build(&g);
        let money_label = g.distillants["money"].label.clone();
        let both = Neighborhood::open(&g, &table, &format!("{money_label} — the landlord raised the rent"));
        assert!(both.on_branch.contains("money") && both.on_branch.contains("money/rent"), "{both:?}");
        assert!(both.beside.contains("money/taxes"), "the root is hit, so its other children are beside: {both:?}");
        assert!(both.beside.contains("money/rent/deposit"), "{both:?}");
        let selective = render_directory_scoped(&g, DirectoryScope::Selective, &table, turn);
        assert!(
            selective.contains("- money/rent [registry] \"Rent\" — Rent and landlord dealings. | routing: rent, landlord\n"),
            "the branch the turn is on carries its line and routing: {selective}"
        );
        assert!(
            selective.contains("- money/rent/deposit [registry] \"Deposit\" — The deposit the landlord holds.\n"),
            "the branch's child carries its line and no routing, though the turn never named it: {selective}"
        );
        assert!(selective.contains("- work/studio [registry] \"Studio\"\n"), "an untouched branch is an index line: {selective}");
        assert!(!selective.contains("The shared studio lease."), "{selective}");
        let money = &g.distillants["money"];
        assert!(
            selective.contains(&format!("- money [registry] \"{}\" — {}", money.label, money.headline())),
            "the branch's ancestor carries its line: {selective}"
        );
        let compact = render_directory_scoped(&g, DirectoryScope::Compact, &table, turn);
        assert!(compact.contains("- money/rent [registry] \"Rent\" | routing: rent, landlord\n"), "{compact}");
        assert!(!compact.contains("Rent and landlord dealings."), "{compact}");
        assert_eq!(
            render_directory_scoped(&g, DirectoryScope::Full, &table, turn),
            render_directory_scoped(&g, DirectoryScope::Full, &table, ""),
            "the full scope ignores the turn"
        );
    }

    #[test]
    fn forget_is_parsed_applied_last_and_leaves_no_trace() {
        let mut g = Graph::seed();
        apply_ops(
            &mut g,
            vec![
                Op::Episode { text: "Ate katsu curry for lunch.".into(), tags: vec!["life".into()], occurred_at: None },
                Op::State {
                    id: "eats-katsu-curry".into(),
                    distillants: vec!["life".into()],
                    text: "Eats katsu curry bento.".into(),
                    relation: Relation::Novel,
                    target: "".into(),
                    importance: 0.3,
                    aliases: vec![],
                    occurred_at: None,
                },
            ],
            1_000,
        );
        assert_eq!(g.episodes.len(), 1);

        // The user asks to forget it in a later turn; the same harvest also
        // (wrongly) restates the fact — the forget still wins, being last.
        let raw = json!({
            "states": [{
                "id": "eats-katsu-curry",
                "distillants": ["life"],
                "text": "Eats katsu curry bento.",
                "relation": "supports",
                "target": "eats-katsu-curry",
                "importance": 0.3,
                "aliases": [],
                "occurred_at": null
            }],
            "forgets": [{"target": "eats-katsu-curry"}, {"target": "never-existed"}]
        })
        .to_string();
        let ops = parse_ops(&raw).unwrap();
        assert!(matches!(ops.last(), Some(Op::Forget { .. })), "forgets sort last in apply order");
        let traces = apply_ops(&mut g, ops, 2_000).traces;
        assert!(!g.leaves.contains_key("eats-katsu-curry"));
        assert!(g.episodes.is_empty(), "the leaf's sole-evidence episode left the stream with it");
        assert!(traces.iter().any(|t| t.starts_with("✗ forgot [eats-katsu-curry]")), "{traces:?}");
        assert!(traces.iter().any(|t| t.contains("forget of unknown id [never-existed]")), "{traces:?}");
        let table = RoutingTable::build(&g);
        assert!(projection::project(&g, "katsu curry again?", 3_000).is_empty(), "nothing recalls it: {:?}", table.matches("katsu"));
    }

    #[test]
    fn manual_ops_parse_ride_last_and_never_touch_the_graph() {
        let raw = json!({
            "manual_upserts": [
                {"tool": "fetch", "service": "resy.com", "lesson": "Plain fetch is refused; the paid search service answers."},
                {"tool": "send_email", "service": "", "lesson": ""}
            ],
            "manual_retires": [{"tool": "fetch", "service": "old-host.dev"}],
            "episodes": [{"text": "Paid rent.", "tags": [], "occurred_at": null}]
        })
        .to_string();
        let ops = parse_ops(&raw).unwrap();
        // The empty-lesson upsert is dropped at parse; manual ops sort last.
        assert_eq!(ops.len(), 3);
        assert!(matches!(&ops[1], Op::ManualUpsert { tool, service, .. } if tool == "fetch" && service == "resy.com"));
        assert!(matches!(&ops[2], Op::ManualRetire { .. }));

        let mut g = Graph::seed();
        let before = serde_json::to_value(&g).unwrap();
        let traces = apply_ops(&mut g, vec![ops[1].clone(), ops[2].clone()], 1_000).traces;
        assert_eq!(
            serde_json::to_value(&g).unwrap(),
            before,
            "host-applied ops leave the graph byte-identical"
        );
        assert_eq!(traces.iter().filter(|t| t.contains("host-applied")).count(), 2, "{traces:?}");
    }

    #[test]
    fn the_field_manual_section_renders_only_when_entries_ride() {
        let g = Graph::seed();
        let bare = build_user_message(&g, &HarvestInput::turn("hi", ""), 1_000, DirectoryScope::Full);
        assert!(!bare.contains("# Field manual"), "empty manual renders no section");
        assert!(!bare.contains("# Instructions to resolve"), "nothing awaiting memory renders no section");
        let with = build_user_message(
            &g,
            &HarvestInput {
                user_text: "hi",
                assistant_text: "",
                field_manual: "- fetch against resy.com: plain fetch refused; use the paid search service",
                instructions: &[],
                documents: &[],
            },
            1_000,
            DirectoryScope::Full,
        );
        assert!(with.contains("# Field manual (existing entries"), "{with}");
        assert!(with.contains("plain fetch refused"), "{with}");
    }

    #[test]
    fn harvest_prompt_pins_every_disposition_regardless_of_routing() {
        let mut g = Graph::seed();
        apply_ops(
            &mut g,
            vec![Op::Disposition {
                id: "frustrated-by-tool-failures".into(),
                distillants: vec!["temperament".into()],
                text: "grows frustrated when tools fail repeatedly".into(),
                dir: 1,
                note: "swore at a failed image edit".into(),
                importance: 0.5,
            }],
            1_000,
        );
        // A turn whose words route nowhere near `temperament`.
        let prompt = render_harvest_prompt(&g, &HarvestInput::turn("ugh, broken again", "sorry about that"), 2_000);
        assert!(
            projection::project_with_caps(&g, "ugh, broken again", 2_000, 12, 48).is_empty(),
            "sanity: the turn text does not route to the profile"
        );
        assert!(prompt.contains("# Profile axes (pinned"), "{prompt}");
        assert!(
            prompt.contains("- [frustrated-by-tool-failures] (under temperament) grows frustrated when tools fail repeatedly (axis +0.25, 1 observations)"),
            "every axis is shown so the harvester nudges instead of minting: {prompt}"
        );
        assert!(prompt.contains("\"forgets\""), "the schema carries the forgets section");
        assert!(prompt.contains("13. Forgetting is the user's call"));
    }

    #[test]
    fn parent_sets_are_antichains_on_every_op_that_sets_them() {
        let mut g = Graph::seed();
        apply_ops(
            &mut g,
            vec![
                Op::Distillant {
                    id: "work/video".into(),
                    tree: Tree::Registry,
                    label: "Video".into(),
                    line: "Video generation.".into(),
                    routing: vec![],
                    parents: vec!["work".into()],
                },
                Op::Distillant {
                    id: "work/video/prompt-hack".into(),
                    tree: Tree::Registry,
                    label: "Prompt hack".into(),
                    line: "A hack that interviews for a prompt.".into(),
                    routing: vec![],
                    parents: vec!["work".into(), "work/video".into()],
                },
                Op::State {
                    id: "take-length".into(),
                    distillants: vec!["work".into(), "work/video/prompt-hack".into()],
                    text: "One continuous take.".into(),
                    relation: Relation::Novel,
                    target: "".into(),
                    importance: 0.5,
                    aliases: vec![],
                    occurred_at: None,
                },
            ],
            1_000,
        );
        assert_eq!(g.distillants["work/video/prompt-hack"].parents, vec!["work/video"], "declared under an ancestor and its descendant: the ancestor is implied");
        assert_eq!(g.leaves["take-length"].parents, vec!["work/video/prompt-hack"]);

        apply_ops(&mut g, vec![Op::Move { leaf: "take-length".into(), parents: vec!["work/video".into(), "work".into()] }], 1_100);
        assert_eq!(g.leaves["take-length"].parents, vec!["work/video"]);

        apply_ops(&mut g, vec![Op::Reparent { distillant: "work/video/prompt-hack".into(), parents: vec!["work".into(), "work/video".into()] }], 1_200);
        assert_eq!(g.distillants["work/video/prompt-hack"].parents, vec!["work/video"]);
    }

    #[test]
    fn a_pre_apex_graph_is_homed_under_the_apex_on_its_first_apply() {
        let mut g = Graph::seed();
        g.distillants.remove(PROFILE_APEX);
        for axis in ["communication", "money-style", "temperament"] {
            g.distillants.get_mut(axis).unwrap().parents.clear();
        }
        let traces = apply_ops(&mut g, vec![], 1_000).traces;
        assert!(traces.iter().any(|t| t == "⇢ profile apex character created"), "{traces:?}");
        assert!(traces.iter().any(|t| t == "⇢ axes homed under the apex: communication, money-style, temperament"), "{traces:?}");
        assert_eq!(g.roots(Tree::Profile).len(), 1);
        // A harvester-declared axis with no parent is an axis of the apex too.
        apply_ops(&mut g, vec![Op::Distillant { id: "risk-appetite".into(), tree: Tree::Profile, label: "Risk appetite".into(), line: "Takes small bets.".into(), routing: vec![], parents: vec![] }], 2_000);
        assert_eq!(g.distillants["risk-appetite"].parents, vec![PROFILE_APEX]);
        assert!(apply_ops(&mut g, vec![], 3_000).traces.is_empty(), "nothing left to repair");
    }

    #[test]
    fn aliases_compile_into_routing_and_match_later() {
        let mut g = Graph::seed();
        apply_ops(&mut g, vec![state_op("rent-amount", "Rent is $2,200/mo.", Relation::Novel, "")], 1_000);
        let table = RoutingTable::build(&g);
        assert!(table.matches("message the landlord about it").contains(&"money/rent".to_string()));
    }

    fn facet_fixture(line: &str, routing: Vec<&str>) -> Graph {
        let mut g = Graph::seed();
        g.distillants.insert(
            "people/xury".into(),
            Distillant {
                id: "people/xury".into(),
                tree: Tree::Registry,
                label: "Xury".into(),
                line: line.into(),
                routing: routing.into_iter().map(String::from).collect(),
                parents: vec!["people".into()],
                misc_count: 0,
                consolidated_at: 0,
                line_changed_at: 0,
                forgotten_at: 0,
            },
        );
        g
    }

    #[test]
    fn harvest_prompt_render_is_self_contained() {
        let g = Graph::seed();
        let p = render_harvest_prompt(&g, &HarvestInput::turn("rent went up to $2,400", "noted"), 1_000);
        assert!(p.starts_with("# System\n"), "system contract inline");
        assert!(p.contains("# Output schema"), "schema inline — no source-reading required");
        assert!(p.contains("\"merge_distillants\"") && p.contains("\"rules\""), "every section present");
        assert!(p.contains("# Standing rules (pinned"), "the rules ride pinned for the harvester: {p}");
        assert!(p.contains("# Turn to harvest\nUser: rent went up to $2,400"));
    }

    #[test]
    fn facet_routing_is_gated_by_the_distillant() {
        let mut g = facet_fixture("Sworn companion; the parting is a standing regret.", vec![]);
        let mut traces = Vec::new();
        let terms: Vec<String> =
            ["regret", "regrets", "jealousy", "faithful boy", "xury"].map(String::from).into();
        add_routing(&mut g, "people/xury", &terms, &mut traces);
        let r = &g.distillants["people/xury"].routing;
        assert!(r.contains(&"regret".to_string()) && r.contains(&"regrets".to_string()));
        assert!(r.contains(&"xury".to_string()), "topical vocabulary passes");
        assert!(r.contains(&"faithful boy".to_string()), "phrases are referring expressions");
        assert!(!r.contains(&"jealousy".to_string()), "ungated facet rejected");
        assert!(traces.iter().any(|t| t.contains('✋') && t.contains("jealousy")), "{traces:?}");
    }

    #[test]
    fn distillant_rewrite_prunes_orphaned_facet_routing() {
        let mut g = facet_fixture(
            "Sworn companion; the parting is a standing regret.",
            vec!["regret", "regrets", "xury", "faithful boy"],
        );
        let traces = apply_ops(
            &mut g,
            vec![Op::Distill {
                distillant: "people/xury".into(),
                line: "Proves loyal through every trial.".into(),
            }],
            2_000,
        )
        .traces;
        let r = &g.distillants["people/xury"].routing;
        assert!(!r.contains(&"regret".to_string()) && !r.contains(&"regrets".to_string()));
        assert!(r.contains(&"xury".to_string()), "topical vocabulary survives the rewrite");
        assert!(r.contains(&"faithful boy".to_string()), "phrases survive too");
        assert!(traces.iter().any(|t| t.contains('✂') && t.contains("regret")), "{traces:?}");
    }

    #[test]
    fn distillant_facets_mirror_into_routing_automatically() {
        let mut g = facet_fixture("Sworn companion; the parting is a standing regret.", vec![]);
        let mut traces = Vec::new();
        sync_facet_routing(&mut g, "people/xury", &mut traces);
        let r = &g.distillants["people/xury"].routing;
        assert!(r.contains(&"regret".to_string()) && r.contains(&"regrets".to_string()));
        assert!(r.contains(&"regretted".to_string()), "the whole family lands: {r:?}");
        assert!(!r.contains(&"fear".to_string()), "uncarried families stay out");
        assert!(traces.iter().any(|t| t.contains('⇐')), "{traces:?}");

        // A rewrite swaps the family: regret out, fear in — synonyms included.
        g.distillants.get_mut("people/xury").unwrap().line =
            "Still afraid of the open sea.".into();
        let mut traces = Vec::new();
        sync_facet_routing(&mut g, "people/xury", &mut traces);
        let r = &g.distillants["people/xury"].routing;
        assert!(!r.contains(&"regret".to_string()), "dropped family pruned: {r:?}");
        assert!(
            r.contains(&"afraid".to_string()) && r.contains(&"fear".to_string()),
            "family breadth: an 'afraid' line routes 'fear' too: {r:?}"
        );
    }

    #[test]
    fn new_distillant_gets_derived_facet_routing_on_creation() {
        let mut g = Graph::seed();
        apply_ops(
            &mut g,
            vec![Op::Distillant {
                id: "people/widow".into(),
                tree: Tree::Registry,
                label: "The widow".into(),
                line: "The one soul he would trust with everything.".into(),
                routing: vec!["widow".into()],
                parents: vec!["people".into()],
            }],
            1_000,
        );
        let r = &g.distillants["people/widow"].routing;
        assert!(r.contains(&"trust".to_string()) && r.contains(&"trusted".to_string()), "{r:?}");
        assert!(r.contains(&"widow".to_string()), "model-written topical terms still land");
    }

    #[test]
    fn distillant_upsert_with_new_distillant_prunes_orphans() {
        let mut g = facet_fixture(
            "Sworn companion; the parting is a standing regret.",
            vec!["regret", "regrets", "xury"],
        );
        apply_ops(
            &mut g,
            vec![Op::Distillant {
                id: "people/xury".into(),
                tree: Tree::Registry,
                label: "Xury".into(),
                line: "The Moorish boy he still trusts completely.".into(),
                routing: vec!["trust".into(), "trusts".into(), "fear".into()],
                parents: vec!["people".into()],
            }],
            2_000,
        );
        let r = &g.distillants["people/xury"].routing;
        assert!(!r.contains(&"regret".to_string()), "orphan of the old line pruned");
        assert!(r.contains(&"trust".to_string()) && r.contains(&"trusts".to_string()));
        assert!(!r.contains(&"fear".to_string()), "still gated by the new line");
    }

    #[test]
    fn supersede_moves_old_value_to_history() {
        let mut g = Graph::seed();
        apply_ops(&mut g, vec![state_op("rent-amount", "Rent is $2,200/mo.", Relation::Novel, "")], 1_000);
        apply_ops(
            &mut g,
            vec![state_op("rent-new", "Rent is $2,400/mo.", Relation::Supersedes, "rent-amount")],
            2_000,
        );
        let leaf = &g.leaves["rent-amount"];
        assert_eq!(leaf.text, "Rent is $2,400/mo.");
        assert_eq!(leaf.kind.history()[0].value, "Rent is $2,200/mo.");
    }

    #[test]
    fn novel_id_collision_with_new_text_supersedes() {
        let mut g = Graph::seed();
        apply_ops(&mut g, vec![state_op("rent-amount", "Rent is $2,200/mo.", Relation::Novel, "")], 1_000);
        apply_ops(&mut g, vec![state_op("rent-amount", "Rent is $2,500/mo.", Relation::Novel, "")], 2_000);
        let leaf = &g.leaves["rent-amount"];
        assert_eq!(leaf.text, "Rent is $2,500/mo.");
        assert_eq!(leaf.kind.history().len(), 1);
    }

    #[test]
    fn contradicts_weakens_without_changing_text() {
        let mut g = Graph::seed();
        apply_ops(&mut g, vec![state_op("rent-amount", "Rent is $2,200/mo.", Relation::Novel, "")], 1_000);
        apply_ops(
            &mut g,
            vec![state_op("rent-doubt", "ignored", Relation::Contradicts, "rent-amount")],
            2_000,
        );
        let leaf = &g.leaves["rent-amount"];
        assert_eq!(leaf.text, "Rent is $2,200/mo.");
        assert_eq!(leaf.kind.belief().unwrap().contradict, 1);
    }

    #[test]
    fn disposition_nudges_drift_and_flip_eventually() {
        let mut g = Graph::seed();
        let d = |dir: i8| Op::Disposition {
            id: "spend-discipline".into(),
            distillants: vec!["money-style".into()],
            text: "Keeps spending tightly controlled.".into(),
            dir,
            note: "obs".into(),
            importance: 0.5,
        };
        apply_ops(&mut g, vec![d(1)], 1_000);
        assert!(disposition_of(&g, "spend-discipline").axis > 0.0);
        for i in 0..8 {
            apply_ops(&mut g, vec![d(-1)], 2_000 + i);
        }
        let d = disposition_of(&g, "spend-discipline");
        assert!(d.axis < -0.3, "consistent counter-evidence drifts the axis across");
        assert_eq!(d.nudges.len(), 9, "trajectory preserved");
    }

    #[test]
    fn parse_rejects_garbage_and_accepts_sections() {
        assert!(parse_ops("not json").is_err());
        assert!(parse_ops("{\"ops\": []}").is_err(), "old tagged-union shape rejected");
        let empty = r#"{"distillants": [], "episodes": [], "states": [], "dispositions": [], "aliases": [], "distills": []}"#;
        assert!(parse_ops(empty).unwrap().is_empty());
        let one = r#"{"distillants": [], "episodes": [{"text": "paid rent", "tags": []}], "states": [], "dispositions": [], "aliases": [], "distills": []}"#;
        assert_eq!(parse_ops(one).unwrap().len(), 1);
        // Degenerate empty-text episodes (the structured-output failure mode) are dropped.
        let degenerate = r#"{"distillants": [], "episodes": [{"text": "", "tags": []}], "states": [], "dispositions": [], "aliases": [], "distills": []}"#;
        assert!(parse_ops(degenerate).unwrap().is_empty());
        // Sections may be omitted entirely (hand-composed ops files).
        assert!(parse_ops("{}").unwrap().is_empty());
        let sparse = r#"{"moves": [{"leaf": "island-goats", "parents": ["life/island/animals"]}]}"#;
        assert_eq!(parse_ops(sparse).unwrap().len(), 1);
        // occurred_at flows through both null and value forms.
        let timed = r#"{"episodes": [{"text": "washed ashore", "tags": [], "occurred_at": 123}], "states": []}"#;
        match &parse_ops(timed).unwrap()[0] {
            Op::Episode { occurred_at, .. } => assert_eq!(*occurred_at, Some(123)),
            other => panic!("unexpected op {other:?}"),
        }
    }

    #[test]
    fn occurred_at_renders_story_time_not_write_time() {
        let mut g = Graph::seed();
        let month = 30 * 86_400;
        let now = 100 * month;
        let mut op = state_op("rent-amount", "Rent is $2,200/mo.", Relation::Novel, "");
        if let Op::State { occurred_at, .. } = &mut op {
            *occurred_at = Some(now - 2 * month);
        }
        apply_ops(&mut g, vec![op], now);
        assert_eq!(g.leaves["rent-amount"].occurred_at, Some(now - 2 * month));

        let mut sup = state_op("rent-new", "Rent is $2,400/mo.", Relation::Supersedes, "rent-amount");
        if let Op::State { occurred_at, .. } = &mut sup {
            *occurred_at = Some(now - month);
        }
        apply_ops(&mut g, vec![sup], now);
        let leaf = &g.leaves["rent-amount"];
        assert_eq!(leaf.kind.history()[0].superseded_at, now - month, "change dated at event time");
        assert_eq!(leaf.updated_at, now, "write time still governs freshness");
    }

    #[test]
    fn reinforce_supports_without_dummy_text() {
        let mut g = Graph::seed();
        apply_ops(&mut g, vec![state_op("rent-amount", "Rent is $2,200/mo.", Relation::Novel, "")], 1_000);
        let ops = vec![
            Op::Episode { text: "Paid rent on time again.".into(), tags: vec![], occurred_at: None },
            Op::Reinforce { target: "rent-amount".into() },
        ];
        let traces = apply_ops(&mut g, ops, 2_000).traces;
        let leaf = &g.leaves["rent-amount"];
        assert_eq!(leaf.kind.belief().unwrap().support, 2);
        assert_eq!(leaf.evidence.len(), 1, "episode wired as evidence");
        assert_eq!(leaf.text, "Rent is $2,200/mo.", "text untouched");
        assert!(traces.iter().any(|t| t.contains("reinforced")));

        let traces = apply_ops(&mut g, vec![Op::Reinforce { target: "ghost".into() }], 3_000).traces;
        assert!(traces.iter().any(|t| t.contains("missing leaf")));
    }

    #[test]
    fn move_rehomes_stranded_leaves_under_new_distillant() {
        // The Crusoe case: life/island/animals created mid-run, earlier
        // leaves stuck on the parent.
        let mut g = Graph::seed();
        apply_ops(
            &mut g,
            vec![Op::State {
                id: "island-goats".into(),
                distillants: vec!["life".into()],
                text: "Keeps a herd of tame goats.".into(),
                relation: Relation::Novel,
                target: "".into(),
                importance: 0.5,
                aliases: vec![],
                occurred_at: None,
            }],
            1_000,
        );
        let ops = vec![
            Op::Distillant {
                id: "life/island/animals".into(),
                tree: Tree::Registry,
                label: "Island animals".into(),
                line: "The island menagerie.".into(),
                routing: vec!["goats".into()],
                parents: vec!["life".into()],
            },
            Op::Move { leaf: "island-goats".into(), parents: vec!["life/island/animals".into()] },
        ];
        let traces = apply_ops(&mut g, ops, 2_000).traces;
        assert_eq!(g.leaves["island-goats"].parents, vec!["life/island/animals"]);
        assert!(traces.iter().any(|t| t.contains("moved [island-goats]")));
        // And the leaf is now reachable through the new distillant's routing.
        let p = crate::projection::project(&g, "how are the goats?", 2_000);
        assert!(p.opened.iter().any(|o| o.distillant_id == "life/island/animals"));
    }

    #[test]
    fn reparent_refuses_cycles() {
        let mut g = Graph::seed();
        apply_ops(
            &mut g,
            vec![
                Op::Distillant {
                    id: "life/island".into(),
                    tree: Tree::Registry,
                    label: "Island".into(),
                    line: "island life".into(),
                    routing: vec![],
                    parents: vec!["life".into()],
                },
                Op::Distillant {
                    id: "life/island/animals".into(),
                    tree: Tree::Registry,
                    label: "Animals".into(),
                    line: "the menagerie".into(),
                    routing: vec![],
                    parents: vec!["life/island".into()],
                },
            ],
            1_000,
        );
        let traces = apply_ops(
            &mut g,
            vec![Op::Reparent { distillant: "life".into(), parents: vec!["life/island/animals".into()] }],
            2_000,
        )
        .traces;
        assert!(traces.iter().any(|t| t.contains("would cycle")));
        assert!(g.distillants["life"].parents.is_empty(), "cycle-creating parent dropped");
    }

    #[test]
    fn merge_leaves_combines_counts_evidence_and_parents() {
        let mut g = Graph::seed();
        apply_ops(
            &mut g,
            vec![
                Op::Episode { text: "ep one".into(), tags: vec![], occurred_at: None },
                state_op("rent-a", "Rent is $2,200/mo.", Relation::Novel, ""),
            ],
            1_000,
        );
        apply_ops(
            &mut g,
            vec![
                Op::Episode { text: "ep two".into(), tags: vec![], occurred_at: None },
                Op::State {
                    id: "rent-b".into(),
                    distillants: vec!["money".into()],
                    text: "Rent: $2,200 monthly.".into(),
                    relation: Relation::Novel,
                    target: "".into(),
                    importance: 0.9,
                    aliases: vec![],
                    occurred_at: None,
                },
            ],
            2_000,
        );
        let traces = apply_ops(
            &mut g,
            vec![Op::MergeLeaf { from: "rent-b".into(), into: "rent-a".into() }],
            3_000,
        )
        .traces;
        assert!(!g.leaves.contains_key("rent-b"));
        let leaf = &g.leaves["rent-a"];
        assert_eq!(leaf.kind.belief().unwrap().support, 2, "counts combine");
        assert_eq!(leaf.evidence.len(), 2, "evidence unions");
        assert_eq!(leaf.parents, vec!["money/rent"], "parents union, antichained: the crown is implied by money/rent");
        assert!((leaf.kind.salience().unwrap().importance - 0.9).abs() < 1e-6, "importance is max");
        assert!(traces.iter().any(|t| t.contains("merged [rent-b]")));
    }

    #[test]
    fn merge_disposition_replays_combined_trajectory() {
        let mut g = Graph::seed();
        let mut d = |id: &str, dir: i8, at: u64| {
            apply_ops(
                &mut g,
                vec![Op::Disposition {
                    id: id.into(),
                    distillants: vec!["money-style".into()],
                    text: "Keeps spending tightly controlled.".into(),
                    dir,
                    note: "obs".into(),
                    importance: 0.5,
                }],
                at,
            );
        };
        d("spend-a", 1, 1_000);
        d("spend-a", 1, 2_000);
        d("spend-b", 1, 1_500);
        apply_ops(&mut g, vec![Op::MergeLeaf { from: "spend-b".into(), into: "spend-a".into() }], 3_000);
        let d = disposition_of(&g, "spend-a");
        assert_eq!(d.nudges.len(), 3);
        let replayed = d.nudges.iter().fold(0.0, |a, n| belief::nudge_axis(a, n.dir));
        assert!((d.axis - replayed).abs() < 1e-6, "axis re-derived from merged trajectory");
    }

    #[test]
    fn merge_distillants_rehomes_everything_and_keeps_vocabulary() {
        let mut g = Graph::seed();
        apply_ops(
            &mut g,
            vec![
                Op::Distillant {
                    id: "people/lisa".into(),
                    tree: Tree::Registry,
                    label: "Lisa".into(),
                    line: "The sister.".into(),
                    routing: vec!["lisa.eth".into()],
                    parents: vec!["people".into()],
                },
                Op::Distillant {
                    id: "people/sister".into(),
                    tree: Tree::Registry,
                    label: "My sister".into(),
                    line: "Sister dealings.".into(),
                    routing: vec!["my sister".into()],
                    parents: vec!["people".into()],
                },
                Op::Episode {
                    text: "Sent sister the usual.".into(),
                    tags: vec!["people/sister".into()],
                    occurred_at: None,
                },
                Op::State {
                    id: "sister-loan".into(),
                    distillants: vec!["people/sister".into()],
                    text: "Owes Lisa $300.".into(),
                    relation: Relation::Novel,
                    target: "".into(),
                    importance: 0.6,
                    aliases: vec![],
                    occurred_at: None,
                },
            ],
            1_000,
        );
        apply_ops(
            &mut g,
            vec![Op::MergeDistillant { from: "people/sister".into(), into: "people/lisa".into() }],
            2_000,
        );
        assert!(!g.distillants.contains_key("people/sister"));
        assert_eq!(g.leaves["sister-loan"].parents, vec!["people/lisa"]);
        assert_eq!(g.episodes[0].tags, vec!["people/lisa"], "episode tags follow the merge");
        let lisa = &g.distillants["people/lisa"];
        assert!(lisa.routing.iter().any(|r| r == "my sister"), "absorbed vocabulary still routes");
        let p = crate::projection::project(&g, "pay my sister back", 2_000);
        assert!(p.opened.iter().any(|o| o.distillant_id == "people/lisa"));
    }
}
