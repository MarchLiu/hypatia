//! Chinese text segmentation for FTS (Full-Text Search).
//!
//! Provides pre-segmentation of Chinese text using jieba before it enters
//! SQLite FTS5. The existing `porter unicode61` tokenizer handles English
//! stemming and space-based tokenization; we insert spaces between Chinese
//! words so `unicode61` can tokenize them correctly.

use std::sync::LazyLock;

use jieba_rs::Jieba;

/// Thread-safe singleton Jieba instance. Dictionary loaded once on first access.
static JIEBA: LazyLock<Jieba> = LazyLock::new(Jieba::new);

/// Trait for text segmentation/tokenization.
/// Future implementations could use LLM-based tokenizers.
pub trait TextSegmenter: Send + Sync {
    fn segment(&self, text: &str) -> String;
}

/// Jieba-based Chinese text segmenter using `cut_for_search` mode.
/// Produces sub-words for better recall (e.g., "南京市" also yields "南京").
pub struct JiebaSegmenter;

impl TextSegmenter for JiebaSegmenter {
    fn segment(&self, text: &str) -> String {
        if !contains_chinese(text) {
            return text.to_string();
        }
        let words = JIEBA.cut_for_search(text, true);
        words.join(" ")
    }
}

/// Check whether the text contains any CJK Unified Ideographs (U+4E00..U+9FFF).
pub fn contains_chinese(text: &str) -> bool {
    text.chars().any(|c| ('\u{4E00}'..='\u{9FFF}').contains(&c))
}

/// Segment text for FTS indexing or querying.
///
/// For text without Chinese characters, returns the input unchanged.
/// For mixed Chinese/English text, Chinese portions are segmented into
/// space-separated words; English words pass through as-is.
pub fn segment_for_fts(text: &str) -> String {
    JiebaSegmenter.segment(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_chinese_passes_through() {
        assert_eq!(segment_for_fts("hello world"), "hello world");
    }

    #[test]
    fn empty_string_passes_through() {
        assert_eq!(segment_for_fts(""), "");
    }

    #[test]
    fn chinese_segmented() {
        let result = segment_for_fts("南京市长江大桥");
        // cut_for_search produces sub-words for better recall
        assert!(result.contains("南京"));
        assert!(result.contains("长江"));
        assert!(result.contains("大桥"));
        assert!(result.contains(' '));
    }

    #[test]
    fn mixed_chinese_english() {
        let result = segment_for_fts("Rust编程语言很好");
        assert!(result.contains("Rust"));
        assert!(result.contains("编程"));
        assert!(result.contains("语言"));
    }

    #[test]
    fn contains_chinese_detection() {
        assert!(contains_chinese("中文"));
        assert!(contains_chinese("hello 世界"));
        assert!(!contains_chinese("hello world"));
        assert!(!contains_chinese(""));
        assert!(!contains_chinese("123 !@#"));
    }
}
