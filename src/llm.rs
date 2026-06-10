//! LLM transport. One trait, two implementations:
//! - `AnthropicClient`: raw Messages API over HTTP (no official Rust SDK).
//! - `MockLlm`: scripted/auto responses for tests and offline sim runs.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};

pub const DEFAULT_MODEL: &str = "claude-opus-4-8";
const API_VERSION: &str = "2023-06-01";

#[derive(Debug, Clone)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub system: String,
    /// Full API message objects ({"role": ..., "content": ...}).
    pub messages: Vec<Value>,
    pub tools: Vec<ToolDef>,
    /// When set, the request uses structured output (output_config.format)
    /// and the first text block is guaranteed to be valid JSON for the schema.
    pub output_schema: Option<Value>,
    pub max_tokens: u32,
    pub thinking: bool,
}

#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub content: Vec<Value>,
    pub stop_reason: String,
}

impl ChatResponse {
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter(|b| b["type"] == "text")
            .filter_map(|b| b["text"].as_str())
            .collect::<Vec<_>>()
            .join("")
    }

    /// (tool_use_id, name, input)
    pub fn tool_uses(&self) -> Vec<(String, String, Value)> {
        self.content
            .iter()
            .filter(|b| b["type"] == "tool_use")
            .map(|b| {
                (
                    b["id"].as_str().unwrap_or_default().to_string(),
                    b["name"].as_str().unwrap_or_default().to_string(),
                    b["input"].clone(),
                )
            })
            .collect()
    }
}

pub trait Llm {
    fn chat(&self, req: &ChatRequest) -> Result<ChatResponse>;
    fn label(&self) -> String;
}

// ---------------------------------------------------------------------------
// Anthropic Messages API
// ---------------------------------------------------------------------------

pub struct AnthropicClient {
    agent: ureq::Agent,
    api_key: String,
    pub model: String,
    base_url: String,
}

impl AnthropicClient {
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .context("ANTHROPIC_API_KEY is not set (run with --mock for offline mode)")?;
        let model = std::env::var("NARRATIVE_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
        let base_url = std::env::var("ANTHROPIC_BASE_URL")
            .unwrap_or_else(|_| "https://api.anthropic.com".to_string());
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(10))
            .timeout_read(Duration::from_secs(300))
            .timeout_write(Duration::from_secs(30))
            .build();
        Ok(AnthropicClient { agent, api_key, model, base_url })
    }

    fn supports_adaptive_thinking(&self) -> bool {
        ["claude-fable", "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8", "claude-sonnet-4-6"]
            .iter()
            .any(|p| self.model.starts_with(p))
    }

    fn body(&self, req: &ChatRequest) -> Value {
        let mut body = json!({
            "model": self.model,
            "max_tokens": req.max_tokens,
            "system": req.system,
            "messages": req.messages,
        });
        if req.thinking && self.supports_adaptive_thinking() {
            body["thinking"] = json!({"type": "adaptive"});
        }
        if !req.tools.is_empty() {
            body["tools"] = Value::Array(
                req.tools
                    .iter()
                    .map(|t| {
                        json!({
                            "name": t.name,
                            "description": t.description,
                            "input_schema": t.input_schema,
                        })
                    })
                    .collect(),
            );
        }
        if let Some(schema) = &req.output_schema {
            body["output_config"] = json!({"format": {"type": "json_schema", "schema": schema}});
        }
        body
    }

    fn send(&self, body: &Value) -> Result<Value> {
        let url = format!("{}/v1/messages", self.base_url);
        let resp = self
            .agent
            .post(&url)
            .set("x-api-key", &self.api_key)
            .set("anthropic-version", API_VERSION)
            .set("content-type", "application/json")
            .send_json(body);
        match resp {
            Ok(r) => Ok(r.into_json::<Value>().context("decoding API response")?),
            Err(ureq::Error::Status(code, r)) => {
                let text = r.into_string().unwrap_or_default();
                Err(anyhow!("API error {code}: {text}"))
            }
            Err(e) => Err(anyhow!("transport error: {e}")),
        }
    }
}

impl Llm for AnthropicClient {
    fn chat(&self, req: &ChatRequest) -> Result<ChatResponse> {
        let body = self.body(req);
        // One retry on transient overload; everything else surfaces immediately.
        let v = match self.send(&body) {
            Ok(v) => v,
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("API error 429") || msg.contains("API error 529") || msg.contains("API error 5") {
                    std::thread::sleep(Duration::from_secs(3));
                    self.send(&body)?
                } else {
                    return Err(e);
                }
            }
        };
        if v["type"] == "error" {
            return Err(anyhow!("API error: {}", v["error"]["message"]));
        }
        Ok(ChatResponse {
            content: v["content"].as_array().cloned().unwrap_or_default(),
            stop_reason: v["stop_reason"].as_str().unwrap_or_default().to_string(),
        })
    }

    fn label(&self) -> String {
        format!("anthropic:{}", self.model)
    }
}

// ---------------------------------------------------------------------------
// Mock — scripted for tests, auto-echo for offline REPL driving
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct MockLlm {
    /// Each entry is a full `content` array consumed in order.
    script: RefCell<VecDeque<Value>>,
}

impl MockLlm {
    pub fn scripted(contents: Vec<Value>) -> Self {
        MockLlm { script: RefCell::new(contents.into()) }
    }

    pub fn push(&self, content: Value) {
        self.script.borrow_mut().push_back(content);
    }

    fn last_user_text(req: &ChatRequest) -> String {
        for m in req.messages.iter().rev() {
            if m["role"] != "user" {
                continue;
            }
            if let Some(s) = m["content"].as_str() {
                return s.to_string();
            }
            if let Some(blocks) = m["content"].as_array() {
                let texts: Vec<&str> = blocks
                    .iter()
                    .filter(|b| b["type"] == "text")
                    .filter_map(|b| b["text"].as_str())
                    .collect();
                if !texts.is_empty() {
                    // Skip injected <recall> blocks; the raw message is last.
                    return texts.last().unwrap_or(&"").to_string();
                }
            }
        }
        String::new()
    }
}

impl Llm for MockLlm {
    fn chat(&self, req: &ChatRequest) -> Result<ChatResponse> {
        if let Some(content) = self.script.borrow_mut().pop_front() {
            let stop = if content
                .as_array()
                .map(|a| a.iter().any(|b| b["type"] == "tool_use"))
                .unwrap_or(false)
            {
                "tool_use"
            } else {
                "end_turn"
            };
            return Ok(ChatResponse {
                content: content.as_array().cloned().unwrap_or_default(),
                stop_reason: stop.to_string(),
            });
        }
        // Unscripted: auto-behavior good enough to drive the loop offline.
        if req.output_schema.is_some() {
            let raw = Self::last_user_text(req);
            // Redistill prompts carry the distillant header; echo the current
            // line back ("already fresh") so auto-consolidation is a
            // visible no-op offline instead of a parse error.
            if let Some(line) = raw.lines().find_map(|l| l.strip_prefix("Current line: ")) {
                let out = json!({
                    "line": line, "routing": [],
                    "distillants": [], "moves": [], "merge_leaves": []
                });
                return Ok(ChatResponse {
                    content: vec![json!({"type": "text", "text": out.to_string()})],
                    stop_reason: "end_turn".to_string(),
                });
            }
            // Digest prompts carry the stream headers; answer with a marked
            // mock summary (must not start with the mechanical "[digest of "
            // marker, or the polish trigger would re-fire forever).
            if raw.starts_with("# Stream period to digest")
                || raw.starts_with("# Mechanical digest to polish")
            {
                let first = raw.lines().nth(1).unwrap_or("").chars().take(120).collect::<String>();
                let out = json!({"text": format!("[mock digest] {first}")});
                return Ok(ChatResponse {
                    content: vec![json!({"type": "text", "text": out.to_string()})],
                    stop_reason: "end_turn".to_string(),
                });
            }
            // Harvest prompts end with "# Turn to harvest\nUser: ...".
            let text = raw
                .rsplit_once("# Turn to harvest")
                .and_then(|(_, turn)| {
                    turn.lines().find_map(|l| l.strip_prefix("User: ")).map(str::to_string)
                })
                .unwrap_or(raw);
            let snippet: String = text.chars().take(120).collect();
            let ops = json!({
                "distillants": [], "states": [], "dispositions": [], "aliases": [], "distills": [],
                "episodes": [{"text": format!("User said: {snippet}"), "tags": []}]
            });
            return Ok(ChatResponse {
                content: vec![json!({"type": "text", "text": ops.to_string()})],
                stop_reason: "end_turn".to_string(),
            });
        }
        Ok(ChatResponse {
            content: vec![json!({"type": "text", "text": "[mock] noted."})],
            stop_reason: "end_turn".to_string(),
        })
    }

    fn label(&self) -> String {
        "mock".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_text_concatenates_text_blocks() {
        let r = ChatResponse {
            content: vec![
                json!({"type": "thinking", "thinking": ""}),
                json!({"type": "text", "text": "a"}),
                json!({"type": "text", "text": "b"}),
            ],
            stop_reason: "end_turn".into(),
        };
        assert_eq!(r.text(), "ab");
    }

    #[test]
    fn mock_auto_harvest_emits_valid_ops_json() {
        let m = MockLlm::default();
        let req = ChatRequest {
            system: String::new(),
            messages: vec![json!({"role": "user", "content": "hello there"})],
            tools: vec![],
            output_schema: Some(json!({"type": "object"})),
            max_tokens: 100,
            thinking: false,
        };
        let r = m.chat(&req).unwrap();
        let v: Value = serde_json::from_str(&r.text()).unwrap();
        assert!(v["episodes"].is_array());
        assert!(v["episodes"][0]["text"].as_str().unwrap().contains("hello there"));
    }

    #[test]
    fn mock_auto_redistill_echoes_current_distillant() {
        let m = MockLlm::default();
        let body = "Distillant: money (label: Money)\nCurrent line: Accounts, obligations.\nCurrent routing: rent\n\nLeaves:\n(none)\n";
        let req = ChatRequest {
            system: String::new(),
            messages: vec![json!({"role": "user", "content": body})],
            tools: vec![],
            output_schema: Some(json!({"type": "object"})),
            max_tokens: 100,
            thinking: false,
        };
        let r = m.chat(&req).unwrap();
        let v: Value = serde_json::from_str(&r.text()).unwrap();
        assert_eq!(v["line"], "Accounts, obligations.");
        assert!(v["distillants"].as_array().unwrap().is_empty());
    }
}
