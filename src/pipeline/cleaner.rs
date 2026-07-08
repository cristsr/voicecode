//! Limpieza de texto (== `pipeline/cleaner.py`).

use regex::{Regex, RegexBuilder};
use tokio::sync::mpsc;

use crate::domain::models::{CleanText, TranscribedText};

/// Compila los patrones de muletillas como regex case-insensitive.
pub fn compile_patterns(patterns: &[String]) -> anyhow::Result<Vec<Regex>> {
    patterns
        .iter()
        .map(|p| {
            RegexBuilder::new(p)
                .case_insensitive(true)
                .build()
                .map_err(Into::into)
        })
        .collect()
}

/// Remueve muletillas, colapsa espacios, limpia puntuación colgante inicial y
/// capitaliza la primera letra. Función pura (== `clean_text` de Python).
pub fn clean_text(raw: &str, filler_patterns: &[Regex]) -> String {
    let mut result = raw.to_string();
    for pattern in filler_patterns {
        result = pattern.replace_all(&result, "").into_owned();
    }
    // Colapsar espacios en blanco y recortar.
    result = collapse_whitespace(&result);
    // Quitar una muletilla suele dejar una coma/;/: colgante al inicio
    // (p. ej. "eh, quiero" -> ", quiero"); se elimina antes de capitalizar.
    result = strip_leading_punctuation(&result);
    capitalize_first(&result)
}

fn collapse_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn strip_leading_punctuation(s: &str) -> String {
    s.trim_start_matches([',', ';', ':', ' ']).to_string()
}

fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// Etapa del pipeline: consume `TranscribedText` y produce `CleanText`.
pub struct RegexCleaner {
    patterns: Vec<Regex>,
}

impl RegexCleaner {
    pub fn new(patterns: Vec<Regex>) -> Self {
        Self { patterns }
    }

    pub async fn run(
        &self,
        mut text_rx: mpsc::Receiver<TranscribedText>,
        clean_tx: mpsc::Sender<CleanText>,
    ) {
        while let Some(item) = text_rx.recv().await {
            let text = clean_text(&item.raw, &self.patterns);
            if clean_tx
                .send(CleanText {
                    seq: item.seq,
                    text,
                })
                .await
                .is_err()
            {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn patterns() -> Vec<Regex> {
        let cfg = crate::config::Cleaner::default();
        compile_patterns(&cfg.filler_patterns).unwrap()
    }

    #[test]
    fn removes_filler_and_capitalizes() {
        assert_eq!(clean_text("eh, quiero esto", &patterns()), "Quiero esto");
    }

    #[test]
    fn removes_multiple_fillers() {
        assert_eq!(
            clean_text("pues o sea quiero digamos algo", &patterns()),
            "Quiero algo"
        );
    }

    #[test]
    fn collapses_internal_whitespace() {
        assert_eq!(clean_text("hola    mundo", &patterns()), "Hola mundo");
    }

    #[test]
    fn empty_input_stays_empty() {
        assert_eq!(clean_text("", &patterns()), "");
    }

    #[test]
    fn only_filler_becomes_empty() {
        assert_eq!(clean_text("eh mmm", &patterns()), "");
    }

    #[test]
    fn case_insensitive_filler() {
        assert_eq!(clean_text("PUES claro", &patterns()), "Claro");
    }

    #[test]
    fn accented_first_letter_is_capitalized() {
        assert_eq!(clean_text("ábaco", &patterns()), "Ábaco");
    }

    #[test]
    fn strips_dangling_leading_comma() {
        assert_eq!(clean_text("o sea, listo", &patterns()), "Listo");
    }
}
