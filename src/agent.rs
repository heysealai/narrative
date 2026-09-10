//! The chat agent: system prompt assembly (pinned rules + pinned profile +
//! registry skeleton) and the per-turn tool loop (`open_memory` BFS descent
//! over the map).

use anyhow::Result;
use serde_json::{json, Value};

use crate::llm::{ChatRequest, Llm, ToolDef};
use crate::model::Graph;
use crate::projection;

pub const OPEN_MEMORY: &str = "open_memory";
const MAX_TOOL_ROUNDS: usize = 6;
/// Keep the sim's API history bounded; the host app owns real eviction.
const HISTORY_CAP: usize = 40;

pub fn open_memory_tool() -> ToolDef {
    ToolDef {
        name: OPEN_MEMORY.to_string(),
        description: "Open a distillant from your memory map and read the detailed facts stored \
                      under it. Call this when the map shows a distillant relevant to the user's \
                      message whose details you need and that are not already in a <recall> \
                      block. Prefer the deepest relevant distillant. The id \"rules\" opens the \
                      standing instructions with the wording each one replaced."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "distillant": {
                    "type": "string",
                    "description": "Distillant id exactly as shown in the map, e.g. \"money\" or \"people/lisa\", or \"rules\""
                }
            },
            "required": ["distillant"],
            "additionalProperties": false
        }),
    }
}

pub fn build_system(graph: &Graph) -> String {
    let rules = match projection::render_rules(graph) {
        Some(rules) => format!("# Standing instructions\n{rules}\n"),
        None => String::new(),
    };
    let profile = projection::render_profile(graph);
    let skeleton = projection::render_registry_skeleton(graph);
    format!(
        "You are Narrative, a personal assistant with a real long-term memory of your user. \
         You are concise, warm, and concrete. You never mention the memory system, the trees, \
         distillants, leaves, or <recall> blocks — you simply know things, the way a good friend \
         with a great memory does. If memory contradicts what the user says now, trust the user \
         and note the change matter-of-factly.\n\
         \n\
         {rules}\
         # Profile (who you are talking to)\n\
         Always-relevant distillation of the user. Let it shape tone and judgment.\n\
         {profile}\n\
         # Memory map (registry skeleton)\n\
         What you know about, without the details. Use the open_memory tool to read details \
         under a distillant when relevant; details may also arrive pre-fetched in a <recall> \
         block on the user's message, in which case do not re-open the same distillant.\n\
         {skeleton}"
    )
}

/// Run one conversational turn: inject recall, loop tool calls, return reply text.
/// `history` accumulates raw API messages and is trimmed at the front, never
/// splitting a tool_use/tool_result pair.
pub fn run_turn(
    llm: &dyn Llm,
    graph: &Graph,
    history: &mut Vec<Value>,
    user_text: &str,
    injection: Option<String>,
    now: u64,
) -> Result<String> {
    let mut blocks = Vec::new();
    if let Some(inj) = injection {
        blocks.push(json!({"type": "text", "text": inj}));
    }
    blocks.push(json!({"type": "text", "text": user_text}));
    history.push(json!({"role": "user", "content": blocks}));
    trim_history(history);

    for _ in 0..MAX_TOOL_ROUNDS {
        let req = ChatRequest {
            system: build_system(graph),
            messages: history.clone(),
            tools: vec![open_memory_tool()],
            output_schema: None,
            max_tokens: 4096,
            thinking: true,
        };
        let resp = llm.chat(&req)?;
        history.push(json!({"role": "assistant", "content": resp.content}));

        if resp.stop_reason == "tool_use" {
            let mut results = Vec::new();
            for (id, name, input) in resp.tool_uses() {
                let text = if name == OPEN_MEMORY {
                    let distillant = input["distillant"].as_str().unwrap_or("");
                    // Visible descent: which distillants the model opens (and in
                    // how many rounds) is the evidence the BFS tier rests on.
                    println!("\x1b[2m  ↳ open_memory({distillant})\x1b[0m");
                    projection::render_open(graph, distillant, now)
                } else {
                    format!("Unknown tool: {name}")
                };
                results.push(json!({"type": "tool_result", "tool_use_id": id, "content": text}));
            }
            history.push(json!({"role": "user", "content": results}));
            continue;
        }
        return Ok(resp.text());
    }
    Ok("(stopped: memory descent exceeded the round limit)".to_string())
}

/// Drop oldest messages once over cap, but only ever cut at a boundary where
/// the next message is a plain user message (not a tool_result follow-up).
fn trim_history(history: &mut Vec<Value>) {
    while history.len() > HISTORY_CAP {
        let mut cut = None;
        for (i, m) in history.iter().enumerate().skip(1) {
            let is_plain_user = m["role"] == "user"
                && !m["content"]
                    .as_array()
                    .map(|a| a.iter().any(|b| b["type"] == "tool_result"))
                    .unwrap_or(false);
            if is_plain_user {
                cut = Some(i);
                break;
            }
        }
        match cut {
            Some(i) if i > 0 => {
                history.drain(0..i);
            }
            _ => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::MockLlm;
    use crate::model::Graph;

    #[test]
    fn tool_loop_resolves_open_memory_then_answers() {
        let g = Graph::seed();
        let llm = MockLlm::scripted(vec![
            json!([{"type": "tool_use", "id": "tu_1", "name": "open_memory", "input": {"distillant": "money"}}]),
            json!([{"type": "text", "text": "final answer"}]),
        ]);
        let mut history = Vec::new();
        let reply = run_turn(&llm, &g, &mut history, "what do you know about my money?", None, 1_000).unwrap();
        assert_eq!(reply, "final answer");
        // user, assistant(tool_use), user(tool_result), assistant(text)
        assert_eq!(history.len(), 4);
        let tr = &history[2]["content"][0];
        assert_eq!(tr["type"], "tool_result");
        assert!(tr["content"].as_str().unwrap().contains("money"));
    }

    #[test]
    fn trim_never_splits_tool_pairs() {
        let mut h = Vec::new();
        for i in 0..30 {
            h.push(json!({"role": "user", "content": [{"type": "text", "text": format!("u{i}")}]}));
            h.push(json!({"role": "assistant", "content": [{"type": "tool_use", "id": "x", "name": "t", "input": {}}]}));
            h.push(json!({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "r"}]}));
            h.push(json!({"role": "assistant", "content": [{"type": "text", "text": "a"}]}));
        }
        trim_history(&mut h);
        assert!(h.len() <= HISTORY_CAP);
        assert_eq!(h[0]["role"], "user");
        let first_is_plain = !h[0]["content"]
            .as_array()
            .unwrap()
            .iter()
            .any(|b| b["type"] == "tool_result");
        assert!(first_is_plain, "history must start at a plain user message");
    }
}
