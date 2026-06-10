//! Local-first persistence: the whole graph as one pretty JSON file.
//! Memory survives across sim sessions; chat history deliberately does not.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

use crate::model::Graph;

pub fn load(path: &Path) -> Result<Graph> {
    if !path.exists() {
        return Ok(Graph::seed());
    }
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let g = serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
    Ok(g)
}

pub fn save(path: &Path, graph: &Graph) -> Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let raw = serde_json::to_string_pretty(graph)?;
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, raw).with_context(|| format!("writing {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| format!("replacing {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Leaf, Species};

    #[test]
    fn roundtrip() {
        let dir = std::env::temp_dir().join(format!("narrative-test-{}", std::process::id()));
        let path = dir.join("memory.json");
        let mut g = Graph::seed();
        let l = Leaf::new("x".into(), Species::State, "fact".into(), vec!["money".into()], 1);
        g.leaves.insert(l.id.clone(), l);
        save(&path, &g).unwrap();
        let g2 = load(&path).unwrap();
        assert_eq!(g2.leaves.len(), 1);
        assert_eq!(g2.leaves["x"].text, "fact");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_file_seeds() {
        let g = load(Path::new("/nonexistent/never/memory.json")).unwrap();
        assert!(g.distillants.contains_key("money"));
    }
}
