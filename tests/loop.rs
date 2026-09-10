//! End-to-end loop test with a scripted mock model:
//! turn 1 stores a fact through the full project → agent → harvest path;
//! turn 2 recalls it via mechanical projection (alias compiled at write time).

use serde_json::json;

use narrative::llm::MockLlm;
use narrative::model::{now, Graph};
use narrative::{agent, harvest, projection};

#[test]
fn fact_stored_on_turn_one_is_recalled_on_turn_two() {
    let t = now();
    let mut graph = Graph::seed();
    let mut history = Vec::new();

    // Scripted responses: [agent reply][harvester ops].
    let harvest_ops = json!({
        "distillants": [{
            "id": "money/rent",
            "tree": "registry",
            "label": "Rent",
            "line": "Rent and landlord dealings.",
            "routing": ["rent", "landlord", "lease"],
            "parents": ["money"]
        }],
        "episodes": [{"text": "User told me their rent terms.", "tags": ["money/rent"]}],
        "states": [{
            "id": "rent-amount",
            "distillants": ["money/rent"],
            "text": "Rent is $2,200/mo, due on the 1st.",
            "relation": "novel",
            "target": "",
            "importance": 0.9,
            "aliases": ["my landlord"]
        }],
        "dispositions": [],
        "aliases": [],
        "distills": []
    });
    let llm = MockLlm::scripted(vec![
        json!([{"type": "text", "text": "Got it — $2,200 on the 1st."}]),
        json!([{"type": "text", "text": harvest_ops.to_string()}]),
    ]);

    // ---- Turn 1: nothing to recall, fact gets harvested.
    let user1 = "btw my rent is $2,200 a month, due on the 1st";
    let proj1 = projection::project(&graph, user1, t);
    let inj1 = projection::render_injection(&graph, &proj1, t);
    let reply1 = agent::run_turn(&llm, &graph, &mut history, user1, inj1, t).unwrap();
    assert_eq!(reply1, "Got it — $2,200 on the 1st.");

    let traces = harvest::run(&llm, &mut graph, &harvest::HarvestInput::turn(user1, &reply1), t).unwrap().traces;
    assert!(traces.iter().any(|tr| tr.contains("rent-amount")), "trace mentions the new leaf: {traces:?}");
    assert!(graph.distillants.contains_key("money/rent"));
    let leaf = &graph.leaves["rent-amount"];
    assert_eq!(leaf.parents, vec!["money/rent"]);
    assert_eq!(leaf.evidence.len(), 1, "episode wired as evidence");

    // ---- Turn 2: mechanical projection recalls it — including via the alias
    // compiled at write time ("my landlord" never appeared in any routing seed).
    let user2 = "remind me, when do I owe my landlord?";
    let proj2 = projection::project(&graph, user2, t + 60);
    let inj2 = projection::render_injection(&graph, &proj2, t + 60)
        .expect("turn 2 must recall the rent fact");
    assert!(inj2.contains("Rent is $2,200/mo"), "recall block carries the fact: {inj2}");
    assert!(inj2.contains("money/rent"));
    // The tagged episode rides along through the same routing match.
    assert!(inj2.contains("## related events"), "episode section present: {inj2}");
    assert!(inj2.contains("User told me their rent terms."));

    // Retrieval reinforces salience.
    projection::touch(&mut graph, &proj2, t + 60);
    assert_eq!(graph.leaves["rent-amount"].kind.salience().unwrap().retrieval_count, 1);
}

#[test]
fn unscripted_mock_drives_the_loop_offline() {
    // The --mock REPL path: auto agent reply + auto episode harvest.
    let t = now();
    let mut graph = Graph::seed();
    let mut history = Vec::new();
    let llm = MockLlm::default();

    let reply = agent::run_turn(&llm, &graph, &mut history, "hello there", None, t).unwrap();
    assert!(!reply.is_empty());
    let traces = harvest::run(&llm, &mut graph, &harvest::HarvestInput::turn("hello there", &reply), t).unwrap().traces;
    assert!(traces.iter().any(|tr| tr.contains("episode")), "{traces:?}");
    assert_eq!(graph.episodes.len(), 1);
}

#[test]
fn harvest_pressure_triggers_automatic_consolidation_in_the_sim() {
    // Three facts land directly on the money crown root (the harvester
    // skipped distillant creation): residual pressure 3 hits the trigger, and
    // the same turn ends with an automatic redistill of the worst offender.
    let t = now();
    let harvest_ops = json!({
        "states": [
            {"id": "fact-a", "distillants": ["money"], "text": "Pays for a storage unit.",
             "relation": "novel", "target": "", "importance": 0.5, "aliases": [], "occurred_at": null},
            {"id": "fact-b", "distillants": ["money"], "text": "Has a Vanguard brokerage.",
             "relation": "novel", "target": "", "importance": 0.5, "aliases": [], "occurred_at": null},
            {"id": "fact-c", "distillants": ["money"], "text": "Splits utilities with a roommate.",
             "relation": "novel", "target": "", "importance": 0.5, "aliases": [], "occurred_at": null}
        ]
    });
    let redistilled = json!({
        "line": "Money is mostly recurring obligations.",
        "routing": ["storage", "vanguard", "utilities"],
        "distillants": [], "moves": [], "merge_leaves": []
    });
    // Scripted responses: [agent reply][harvester ops][auto redistill].
    let llm = MockLlm::scripted(vec![
        json!([{"type": "text", "text": "Noted."}]),
        json!([{"type": "text", "text": harvest_ops.to_string()}]),
        json!([{"type": "text", "text": redistilled.to_string()}]),
    ]);

    let data = std::env::temp_dir().join(format!("narrative-loop-test-{}.json", std::process::id()));
    let _ = std::fs::remove_file(&data);
    let mut sim = narrative::sim::Sim {
        graph: Graph::seed(),
        llm: Box::new(llm),
        history: Vec::new(),
        data_file: data.clone(),
    };
    sim.turn("a few money things to remember").unwrap();

    let m = &sim.graph.distillants["money"];
    assert_eq!(m.line, "Money is mostly recurring obligations.", "auto pass ran in the same turn");
    assert_eq!(m.misc_count, 0, "residual cleared");
    assert!(m.consolidated_at >= t, "pass stamped");
    let _ = std::fs::remove_file(&data);
}

#[test]
fn an_instruction_becomes_a_pinned_rule_the_next_turn_reads() {
    // Turn 1: the user gives a standing instruction the assistant acknowledges;
    // the host hands it to the harvester as an instruction to resolve.
    // Turn 2: the rule rides pinned in the system prompt, above the profile,
    // and a later correction supersedes it by id.
    let t = now();
    let mut graph = Graph::seed();
    let mut history = Vec::new();
    let pending = vec![harvest::Instruction {
        id: "i-1".into(),
        utterance: "from now on keep replies to five lines".into(),
        paraphrase: "reply in at most five lines".into(),
    }];
    let kept = json!({
        "rules": [{
            "id": "five-lines", "text": "keep replies to five lines", "relation": "novel",
            "target": "", "instruction": "i-1", "occurred_at": null
        }]
    });
    let superseded = json!({
        "rules": [{
            "id": "five-lines", "text": "keep replies to three lines", "relation": "supersedes",
            "target": "five-lines", "instruction": "", "occurred_at": null
        }]
    });
    let llm = MockLlm::scripted(vec![
        json!([{"type": "text", "text": "Five lines it is."}]),
        json!([{"type": "text", "text": kept.to_string()}]),
        json!([{"type": "text", "text": "Three, then."}]),
        json!([{"type": "text", "text": superseded.to_string()}]),
    ]);

    let user1 = "from now on keep replies to five lines";
    let reply1 = agent::run_turn(&llm, &graph, &mut history, user1, None, t).unwrap();
    let input = harvest::HarvestInput { instructions: &pending, ..harvest::HarvestInput::turn(user1, &reply1) };
    let applied = harvest::run(&llm, &mut graph, &input, t).unwrap();
    let resolved = harvest::resolve(&pending, &applied);
    assert_eq!(
        resolved,
        vec![("i-1".to_string(), harvest::Resolution::Rule(harvest::RuleEffect::Kept { id: "five-lines".into(), text: "keep replies to five lines".into() }))]
    );

    let system = agent::build_system(&graph);
    let rules_at = system.find("# Standing instructions").expect("rules ride pinned");
    let profile_at = system.find("# Profile").expect("profile rides pinned");
    assert!(rules_at < profile_at, "rules above the profile: {system}");
    assert!(system.contains("- [five-lines] keep replies to five lines"), "{system}");
    assert_eq!(agent::build_system(&graph), system, "the pinned block is a pure function of the graph");

    let user2 = "make that three lines";
    let reply2 = agent::run_turn(&llm, &graph, &mut history, user2, None, t + 60).unwrap();
    let applied = harvest::run(&llm, &mut graph, &harvest::HarvestInput::turn(user2, &reply2), t + 60).unwrap();
    assert!(matches!(applied.rules[0].effect, harvest::RuleEffect::Superseded { .. }), "{:?}", applied.rules);
    let system = agent::build_system(&graph);
    assert!(system.contains("- [five-lines] keep replies to three lines (was: keep replies to five lines)"), "{system}");
    assert_eq!(graph.rules().len(), 1, "one rule, reworded");
    assert!(projection::project(&graph, "five lines?", t + 120).opened.is_empty(), "no routing reaches a rule");
}
