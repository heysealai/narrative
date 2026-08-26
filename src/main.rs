use std::io::Read;
use std::path::PathBuf;

use anyhow::{bail, Result};

use narrative::llm::{AnthropicClient, Llm, MockLlm};
use narrative::sim::{run_repl, Sim};
use narrative::{consolidate, harvest, model, projection, store};

const USAGE: &str = "\
narrative — structured long-term memory engine

  narrative                       chat REPL (needs ANTHROPIC_API_KEY, or --mock)
  narrative project <text|@file>  mechanical recall for a message (touches salience)
  narrative harvest-prompt <user text|@file> [assistant text|@file]
                                  print the harvester contract for a finished
                                  turn (system + output schema + input)
  narrative apply <file|->        apply harvester ops JSON (sections shape)
  narrative digest-prompt         print the input for a due stream digest pass
  narrative digest <text|@file|-> [--take n]
                                  apply distilled digest text to the due pass
                                  (take = episodes the digest covers, fold only)
  narrative redistill-prompt <distillant-id>
                                  print the redistill input for one distillant
  narrative redistill <distillant-id> <json|@file|->
                                  apply a redistill response to that distillant
  narrative open <distillant-id>    read leaves under one distillant
  narrative profile               pinned tier (always-inline profile)
  narrative map                   registry skeleton
  narrative stream [n]            last n episodes
  narrative stats                 residual / consolidation-pressure report
  narrative forget <id>           forget a leaf or distillant (with its sole evidence)

The project/harvest-prompt/apply subcommands externalize the model role:
whatever intelligence drives the CLI plays agent and harvester. Memory lives
in $NARRATIVE_DATA/memory.json (default ./data).";

/// The CLI is keyless — the driving intelligence is the model. Tell it when
/// a pass is due so it can run one (redistill-prompt / digest-prompt, then
/// apply the response).
fn print_due_hints(graph: &narrative::model::Graph) {
    if let Some(id) = consolidate::due(graph) {
        println!("⚙ consolidation due: {id} (pressure {})", consolidate::pressure(graph, &id));
    }
    if let Some(step) = consolidate::stream_due(graph) {
        println!("⚙ {}", step.describe(graph));
    }
}

/// Literal text, or @path to read a file, or "-" for stdin.
fn text_arg(arg: &str) -> Result<String> {
    if arg == "-" {
        let mut s = String::new();
        std::io::stdin().read_to_string(&mut s)?;
        Ok(s)
    } else if let Some(path) = arg.strip_prefix('@') {
        Ok(std::fs::read_to_string(path)?)
    } else {
        Ok(arg.to_string())
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mock = args.iter().any(|a| a == "--mock");
    let args: Vec<String> = args.into_iter().filter(|a| a != "--mock").collect();

    let data_dir = std::env::var("NARRATIVE_DATA").unwrap_or_else(|_| "data".to_string());
    let data_file = PathBuf::from(data_dir).join("memory.json");
    let now = model::now();

    let Some(cmd) = args.first().map(String::as_str) else {
        // No subcommand: interactive REPL with a real (or mock) model.
        let llm: Box<dyn Llm> = if mock {
            Box::new(MockLlm::default())
        } else {
            Box::new(AnthropicClient::from_env()?)
        };
        let mut sim = Sim::new(llm, data_file)?;
        return run_repl(&mut sim);
    };

    let mut graph = store::load(&data_file)?;
    match cmd {
        "help" | "--help" | "-h" => println!("{USAGE}"),
        "project" => {
            let text = text_arg(args.get(1).map(String::as_str).unwrap_or(""))?;
            let p = projection::project(&graph, &text, now);
            match projection::render_injection(&graph, &p, now) {
                Some(inj) => {
                    println!("{inj}");
                    projection::touch(&mut graph, &p, now);
                    store::save(&data_file, &graph)?;
                }
                None => println!("(nothing recalled)"),
            }
        }
        "harvest-prompt" => {
            let user = text_arg(args.get(1).map(String::as_str).unwrap_or(""))?;
            let assistant = match args.get(2) {
                Some(a) => text_arg(a)?,
                None => String::new(),
            };
            println!("{}", harvest::render_harvest_prompt(&graph, &user, &assistant, now));
        }
        "apply" => {
            let src = match args.get(1).map(String::as_str) {
                Some(p) if p != "-" => format!("@{p}"),
                _ => "-".to_string(),
            };
            let raw = text_arg(&src)?;
            let ops = harvest::parse_ops(&raw)?;
            for t in harvest::apply_ops(&mut graph, ops, now) {
                println!("✎ {t}");
            }
            print_due_hints(&graph);
            store::save(&data_file, &graph)?;
        }
        "digest-prompt" => match consolidate::stream_due(&graph) {
            Some(step) => print!("{}", consolidate::render_digest_prompt(&graph, &step, now)),
            None => println!("(no stream digest due)"),
        },
        "digest" => {
            let mut take: Option<usize> = None;
            let mut rest: Vec<&str> = Vec::new();
            let mut it = args.iter().skip(1).map(String::as_str);
            while let Some(a) = it.next() {
                if a == "--take" {
                    match it.next().and_then(|v| v.parse().ok()) {
                        Some(n) => take = Some(n),
                        None => bail!("--take needs a number"),
                    }
                } else {
                    rest.push(a);
                }
            }
            let text = text_arg(rest.first().copied().unwrap_or("-"))?;
            for t in consolidate::apply_digest(&mut graph, text, take, now)? {
                println!("✎ {t}");
            }
            print_due_hints(&graph);
            store::save(&data_file, &graph)?;
        }
        "redistill-prompt" => {
            let id = args.get(1).map(String::as_str).unwrap_or("");
            match consolidate::render_redistill_prompt(&graph, id) {
                Some(p) => print!("{p}"),
                None => bail!("no distillant \"{id}\""),
            }
        }
        "redistill" => {
            let Some(id) = args.get(1).map(String::as_str) else {
                bail!("redistill needs a distillant id");
            };
            let raw = text_arg(args.get(2).map(String::as_str).unwrap_or("-"))?;
            for t in consolidate::apply_redistilled(&mut graph, id, &raw, now)? {
                println!("✎ {t}");
            }
            print_due_hints(&graph);
            store::save(&data_file, &graph)?;
        }
        "open" => {
            print!("{}", projection::render_open(&graph, args.get(1).map(String::as_str).unwrap_or(""), now))
        }
        "profile" => print!("{}", projection::render_profile(&graph, now)),
        "map" => print!("{}", projection::render_registry_skeleton(&graph, now)),
        "stream" => {
            let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10);
            for e in graph.episodes.iter().rev().take(n).rev() {
                println!("[{}] {} {}", e.id, model::age_str(now, e.event_at()), e.text);
            }
        }
        "stats" => print!("{}", consolidate::stats(&graph)),
        "forget" => {
            let id = args.get(1).map(String::as_str).unwrap_or("");
            let Some(gone) = graph.forget(id, now) else {
                bail!("nothing with id \"{id}\"");
            };
            println!(
                "✗ forgot {id} — {} leaves, {} distillants, {} episodes left the graph",
                gone.leaves.len(),
                gone.distillants.len(),
                gone.episodes.len()
            );
            store::save(&data_file, &graph)?;
        }
        other => bail!("unknown subcommand {other}\n\n{USAGE}"),
    }
    Ok(())
}
