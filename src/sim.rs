//! The chat-box simulator: a REPL that drives the full loop —
//! project → inject → agent turn → harvest → persist — with visible traces,
//! plus debug commands to inspect every store.

use std::io::{BufRead, Write};
use std::path::PathBuf;

use anyhow::Result;
use serde_json::Value;

use crate::model::{age_str, Graph};
use crate::{agent, consolidate, harvest, llm::Llm, model, projection, store};

const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

pub struct Sim {
    pub graph: Graph,
    pub llm: Box<dyn Llm>,
    pub history: Vec<Value>,
    pub data_file: PathBuf,
}

impl Sim {
    pub fn new(llm: Box<dyn Llm>, data_file: PathBuf) -> Result<Self> {
        let graph = store::load(&data_file)?;
        Ok(Sim { graph, llm, history: Vec::new(), data_file })
    }

    /// One full conversational turn through the memory loop.
    pub fn turn(&mut self, text: &str) -> Result<String> {
        let now = model::now();

        // 1. Mechanical projection — read-time hot path, no LLM.
        let proj = projection::project(&self.graph, text, now);
        if !proj.is_empty() {
            trace(&format!("↳ recall: {}", proj.summary(&self.graph)));
        }
        let injection = projection::render_injection(&self.graph, &proj, now);
        projection::touch(&mut self.graph, &proj, now);

        // 2. The agent turn (pinned profile + skeleton in system; recall injected).
        let reply =
            agent::run_turn(self.llm.as_ref(), &self.graph, &mut self.history, text, injection, now)?;

        // 3. Ambient harvest of the finished turn. Failure here must not eat the reply.
        match harvest::run(self.llm.as_ref(), &mut self.graph, &harvest::HarvestInput::turn(text, &reply), now) {
            Ok(applied) => {
                for t in &applied.traces {
                    trace(&format!("✎ {t}"));
                }
            }
            Err(e) => trace(&format!("⚠ harvest failed: {e}")),
        }

        // 3b. Automatic consolidation: one descent step on the worst offender
        // when harvest pushed pressure over the trigger (the sim-side stand-in
        // for the host's eviction-watermark coupling).
        self.consolidate_step(now);

        // 4. Persist memory (chat history intentionally not persisted).
        store::save(&self.data_file, &self.graph)?;
        Ok(reply)
    }

    fn consolidate_step(&mut self, now: u64) {
        match consolidate::auto_step(self.llm.as_ref(), &mut self.graph, now) {
            Ok(traces) => {
                for t in traces {
                    trace(&format!("✎ {t}"));
                }
            }
            Err(e) => trace(&format!("⚠ consolidation failed: {e}")),
        }
    }

    /// Returns true when the REPL should exit.
    pub fn command(&mut self, line: &str) -> Result<bool> {
        let now = model::now();
        let (cmd, arg) = match line.split_once(char::is_whitespace) {
            Some((c, a)) => (c, a.trim()),
            None => (line, ""),
        };
        match cmd {
            "/quit" | "/q" | "/exit" => return Ok(true),
            "/help" => println!("{}", HELP),
            "/rules" => match projection::render_rules(&self.graph) {
                Some(rules) => print!("{rules}"),
                None => println!("(no standing instructions yet)"),
            },
            "/profile" => {
                if let Some(rules) = projection::render_rules(&self.graph) {
                    println!("{BOLD}# Standing instructions{RESET}");
                    print!("{rules}");
                    println!("{BOLD}# Profile{RESET}");
                }
                print!("{}", projection::render_profile(&self.graph));
            }
            "/map" => print!("{}", projection::render_registry_skeleton(&self.graph)),
            "/tree" => {
                println!("{BOLD}# Registry{RESET}");
                print!("{}", projection::render_registry_skeleton(&self.graph));
                for id in self.graph.distillants.keys().cloned().collect::<Vec<_>>() {
                    let leaves = self.graph.leaves_under(&id);
                    if !leaves.is_empty() {
                        print!("{}", projection::render_open(&self.graph, &id, now));
                    }
                }
                println!("{BOLD}# Profile{RESET}");
                print!("{}", projection::render_profile(&self.graph));
            }
            "/open" => print!("{}", projection::render_open(&self.graph, arg, now)),
            "/stream" => {
                let n: usize = arg.parse().unwrap_or(10);
                for e in self.graph.episodes.iter().rev().take(n).rev() {
                    println!("[{}] {} {}", e.id, age_str(now, e.event_at()), e.text);
                }
                if self.graph.episodes.is_empty() {
                    println!("(stream empty)");
                }
            }
            "/project" => {
                let p = projection::project(&self.graph, arg, now);
                match projection::render_injection(&self.graph, &p, now) {
                    Some(inj) => println!("{inj}"),
                    None => println!("(nothing would be recalled for that message)"),
                }
            }
            "/feed" => {
                // Ingestion: harvest user-voice text directly, no agent turn.
                match harvest::run(self.llm.as_ref(), &mut self.graph, &harvest::HarvestInput::turn(arg, ""), now) {
                    Ok(applied) => {
                        let traces = applied.traces;
                        for t in &traces {
                            trace(&format!("✎ {t}"));
                        }
                        if traces.is_empty() {
                            trace("(nothing durable harvested)");
                        }
                    }
                    Err(e) => trace(&format!("⚠ feed harvest failed: {e}")),
                }
                self.consolidate_step(now);
                store::save(&self.data_file, &self.graph)?;
            }
            "/distill" => match consolidate::redistill(self.llm.as_ref(), &mut self.graph, arg, now) {
                Ok(traces) => {
                    for t in traces {
                        trace(&format!("✎ {t}"));
                    }
                    store::save(&self.data_file, &self.graph)?;
                }
                Err(e) => trace(&format!("⚠ distill failed: {e}")),
            },
            "/digest" => match consolidate::distill_stream(self.llm.as_ref(), &mut self.graph, now) {
                Ok(traces) if traces.is_empty() => println!("(no stream digest due)"),
                Ok(traces) => {
                    for t in traces {
                        trace(&format!("✎ {t}"));
                    }
                    store::save(&self.data_file, &self.graph)?;
                }
                Err(e) => trace(&format!("⚠ digest failed: {e}")),
            },
            "/stats" => print!("{}", consolidate::stats(&self.graph)),
            "/forget" => match self.graph.forget(arg, model::now()) {
                Some(gone) => {
                    trace(&format!(
                        "✗ forgot {arg} — {} leaves, {} distillants, {} episodes left the graph",
                        gone.leaves.len(),
                        gone.distillants.len(),
                        gone.episodes.len()
                    ));
                    store::save(&self.data_file, &self.graph)?;
                }
                None => println!("nothing with id \"{arg}\""),
            },
            "/reset-chat" => {
                self.history.clear();
                trace("chat history cleared (memory untouched)");
            }
            "/save" => {
                store::save(&self.data_file, &self.graph)?;
                trace(&format!("saved to {}", self.data_file.display()));
            }
            _ => println!("unknown command {cmd} — try /help"),
        }
        Ok(false)
    }
}

const HELP: &str = "\
chat:        type anything — full loop: project → recall → reply → harvest → save
/rules       standing instructions (pinned first, above the profile)
/profile     pinned tier (what always rides inline: rules, then profile)
/map         registry skeleton (the model's in-context map)
/tree        both trees, with leaves
/open <id>   open one distillant (what the open_memory tool returns)
/stream [n]  last n episodes
/project <t> dry-run: what a message would recall, without sending it
/feed <text> ingest user-voice text (harvest only, no agent reply)
/distill <id> consolidation step: re-distill a distillant via the model
/digest      stream step: distill the due digest pass via the model
/stats       residual report: where consolidation pressure is building
/forget <id> forget a leaf or distillant (with its sole evidence)
/reset-chat  clear chat history (memory persists — the whole point)
/save        force save
/quit        exit";

fn trace(s: &str) {
    println!("{DIM}{s}{RESET}");
}

pub fn run_repl(sim: &mut Sim) -> Result<()> {
    println!(
        "{BOLD}narrative{RESET} · model {} · memory {} ({} leaves, {} episodes)",
        sim.llm.label(),
        sim.data_file.display(),
        sim.graph.leaves.len(),
        sim.graph.episodes.len()
    );
    println!("{DIM}type to chat, /help for commands{RESET}");

    let stdin = std::io::stdin();
    loop {
        print!("\n{BOLD}you ❯{RESET} ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break; // EOF
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('/') {
            if sim.command(line)? {
                break;
            }
            continue;
        }
        match sim.turn(line) {
            Ok(reply) => println!("\n{BOLD}narrative ❯{RESET} {reply}"),
            Err(e) => println!("⚠ turn failed: {e}"),
        }
    }
    store::save(&sim.data_file, &sim.graph)?;
    Ok(())
}
