//! Transcript replay: the harvester driven over recorded turns from a fixed
//! starting graph under one directory scope, recording what each absorb
//! cost and wrote. The same turns under each scope, from the same graph,
//! is the comparison the scope question needs (docs/harvest-scope-replay.md).
//!
//! The turn text takes the host's shape — one speaker-prefixed line per
//! row — and the request the host's: the keyless prompt as one user
//! message, no output schema, the host's response cap. Event time is the
//! turn's own, for the render and the apply alike.

use std::collections::BTreeSet;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::harvest::{self, DirectoryScope, Neighborhood, Op, Relation};
use crate::llm::{ChatRequest, Llm, Usage};
use crate::model::Graph;
use crate::routing::RoutingTable;
use crate::store;

/// The host's response cap for one harvest.
const RESPONSE_CAP_TOKENS: u32 = 32_768;

#[derive(Deserialize)]
pub struct Turns {
    pub turns: Vec<Turn>,
}

/// One recorded turn: the user's row and the assistant rows that answered
/// it, stamped with the row's own time.
#[derive(Deserialize)]
pub struct Turn {
    pub seq: u64,
    pub at: String,
    pub at_epoch: u64,
    pub user: String,
    pub assistant: String,
}

/// What one absorb emitted, by op section, states split by relation.
#[derive(Serialize, Default)]
pub struct OpCounts {
    pub distillants: usize,
    pub episodes: usize,
    pub states_novel: usize,
    pub states_duplicate: usize,
    pub states_supports: usize,
    pub states_contradicts: usize,
    pub states_supersedes: usize,
    pub dispositions: usize,
    pub reinforces: usize,
    pub aliases: usize,
    pub distills: usize,
    pub moves: usize,
    pub reparents: usize,
    pub merge_leaves: usize,
    pub merge_distillants: usize,
    pub forgets: usize,
    pub manual: usize,
}

#[derive(Serialize)]
pub struct TurnReport {
    pub seq: u64,
    pub at: String,
    pub prompt_chars: usize,
    /// The turn's [`Neighborhood`] — the distillants on the branches it is
    /// on, and the ones beside them — a property of the turn against the
    /// graph, reported under every scope.
    pub on_branch_distillants: usize,
    pub beside_distillants: usize,
    pub usage: Usage,
    pub stop_reason: String,
    pub elapsed_ms: u128,
    pub attempts: u8,
    pub parse_failed: bool,
    pub ops: OpCounts,
    pub new_distillants: Vec<String>,
    pub new_leaves: Vec<String>,
    pub traces: Vec<String>,
}

fn count(ops: &[Op]) -> OpCounts {
    let mut c = OpCounts::default();
    for op in ops {
        match op {
            Op::Distillant { .. } => c.distillants += 1,
            Op::Episode { .. } => c.episodes += 1,
            Op::State { relation, .. } => match relation {
                Relation::Novel => c.states_novel += 1,
                Relation::Duplicate => c.states_duplicate += 1,
                Relation::Supports => c.states_supports += 1,
                Relation::Contradicts => c.states_contradicts += 1,
                Relation::Supersedes => c.states_supersedes += 1,
            },
            Op::Disposition { .. } => c.dispositions += 1,
            Op::Reinforce { .. } => c.reinforces += 1,
            Op::Alias { .. } => c.aliases += 1,
            Op::Distill { .. } => c.distills += 1,
            Op::Move { .. } => c.moves += 1,
            Op::Reparent { .. } => c.reparents += 1,
            Op::MergeLeaf { .. } => c.merge_leaves += 1,
            Op::MergeDistillant { .. } => c.merge_distillants += 1,
            Op::Forget { .. } => c.forgets += 1,
            Op::ManualUpsert { .. } | Op::ManualRetire { .. } => c.manual += 1,
        }
    }
    c
}

/// The host's transcript line shape.
fn transcript(turn: &Turn) -> String {
    format!("User: {}\nAssistant: {}", turn.user.trim(), turn.assistant.trim())
}

/// The host's salvage: the outermost JSON object of a response that wrapped
/// its ops in prose or fences.
fn salvage_json(text: &str) -> &str {
    match (text.find('{'), text.rfind('}')) {
        (Some(open), Some(close)) if open < close => &text[open..=close],
        _ => text,
    }
}

pub fn run(
    llm: &dyn Llm,
    graph: &mut Graph,
    data_file: &Path,
    turns: &Turns,
    scope: DirectoryScope,
    report_path: &Path,
) -> Result<()> {
    let mut reports: Vec<TurnReport> = Vec::new();
    for turn in &turns.turns {
        let turn_text = transcript(turn);
        let neighborhood = Neighborhood::open(graph, &RoutingTable::build(graph), &turn_text);
        let prompt = harvest::render_harvest_prompt_scoped(graph, &turn_text, "", "", turn.at_epoch, scope);
        let prompt_chars = prompt.len();
        let request = ChatRequest {
            system: String::new(),
            messages: vec![json!({"role": "user", "content": prompt})],
            tools: Vec::new(),
            output_schema: None,
            max_tokens: RESPONSE_CAP_TOKENS,
            thinking: false,
        };
        let started = Instant::now();
        let mut usage = Usage::default();
        let mut stop_reason = String::new();
        let mut attempts = 0u8;
        let mut parsed: Option<Vec<Op>> = None;
        // One retry on an unparseable response, as harvest::run allows.
        while parsed.is_none() && attempts < 2 {
            attempts += 1;
            let response = llm.chat(&request).with_context(|| format!("turn {}", turn.seq))?;
            usage.add(response.usage);
            stop_reason = response.stop_reason.clone();
            let text = response.text();
            parsed = harvest::parse_ops(&text)
                .or_else(|_| harvest::parse_ops(salvage_json(&text)))
                .ok();
        }
        let parse_failed = parsed.is_none();
        let ops = parsed.unwrap_or_default();
        let ops_count = count(&ops);
        let distillants_before: BTreeSet<String> = graph.distillants.keys().cloned().collect();
        let leaves_before: BTreeSet<String> = graph.leaves.keys().cloned().collect();
        let traces = harvest::apply_ops(graph, ops, turn.at_epoch);
        store::save(data_file, graph)?;
        let new_distillants: Vec<String> =
            graph.distillants.keys().filter(|k| !distillants_before.contains(*k)).cloned().collect();
        let new_leaves: Vec<String> =
            graph.leaves.keys().filter(|k| !leaves_before.contains(*k)).cloned().collect();
        let parse_note = if parse_failed { " (unparseable, nothing applied)" } else { "" };
        eprintln!(
            "turn {} — {} in / {} out tokens, +{} distillants +{} leaves, {} traces{parse_note}",
            turn.seq,
            usage.input_tokens,
            usage.output_tokens,
            new_distillants.len(),
            new_leaves.len(),
            traces.len()
        );
        reports.push(TurnReport {
            seq: turn.seq,
            at: turn.at.clone(),
            prompt_chars,
            on_branch_distillants: neighborhood.on_branch.len(),
            beside_distillants: neighborhood.beside.len(),
            usage,
            stop_reason,
            elapsed_ms: started.elapsed().as_millis(),
            attempts,
            parse_failed,
            ops: ops_count,
            new_distillants,
            new_leaves,
            traces,
        });
        std::fs::write(report_path, serde_json::to_string_pretty(&reports)?)?;
    }
    Ok(())
}
