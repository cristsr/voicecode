//! Text cleaning stage: strips filler words and tidies the transcription.

use std::sync::OnceLock;

use regex::{Regex, RegexBuilder};
use tokio::sync::mpsc;

use crate::domain::models::{CleanText, TranscribedText};

/// Compiles the filler-word patterns as case-insensitive regexes.
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

/// Matches a run of two or more punctuation marks separated only by
/// whitespace (e.g. ", ," or ", , ,"), which is what's left behind when a
/// filler word sitting between two marks is removed (e.g. "hola, eh, quiero"
/// -> "hola, , quiero"). Captures the first mark so the run collapses to it.
fn repeated_punctuation() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"([,;:])(?:\s*[,;:])+").expect("valid regex"))
}

/// Pure text cleaner: removes filler words, collapses orphaned punctuation and
/// whitespace left behind, strips dangling punctuation at either end, and
/// capitalizes the first letter.
pub fn clean_text(raw: &str, filler_patterns: &[Regex]) -> String {
    let mut result = raw.to_string();
    for pattern in filler_patterns {
        result = pattern.replace_all(&result, "").into_owned();
    }
    // Do this before collapsing whitespace: removing a filler from between
    // two marks leaves a run like ", ," that whitespace-collapsing alone
    // would not fix.
    result = repeated_punctuation().replace_all(&result, "$1").into_owned();
    result = collapse_whitespace(&result);
    // Removing a filler often leaves dangling punctuation at either end
    // (e.g. "eh, quiero" -> ", quiero", "quiero, pues" -> "quiero,"); drop it
    // before capitalizing.
    result = strip_dangling_punctuation(&result);
    capitalize_first(&result)
}

fn collapse_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn strip_dangling_punctuation(s: &str) -> String {
    s.trim_matches([',', ';', ':', ' ']).to_string()
}

fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// Pipeline stage: consumes `TranscribedText` and produces `CleanText`.
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

    #[test]
    fn collapses_orphaned_comma_left_by_mid_sentence_filler() {
        assert_eq!(
            clean_text("Hola, eh, quiero decir algo", &patterns()),
            "Hola, quiero decir algo"
        );
    }

    #[test]
    fn collapses_orphaned_commas_from_consecutive_fillers() {
        assert_eq!(
            clean_text("Hola, eh, mmm, quiero decir algo", &patterns()),
            "Hola, quiero decir algo"
        );
    }

    #[test]
    fn strips_dangling_trailing_comma() {
        assert_eq!(
            clean_text("Quiero decir algo, pues", &patterns()),
            "Quiero decir algo"
        );
    }
}
