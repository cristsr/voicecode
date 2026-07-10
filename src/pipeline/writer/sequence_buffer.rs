//! Pure reordering logic for the writer stage.
//!
//! Transcriptions complete out of order (each chunk runs on its own task), so
//! this buffer holds early arrivals until the contiguous `seq` before them has
//! passed through, releasing them strictly in order. Kept free of any
//! clipboard/keyboard I/O so the part with the most edge cases is unit-tested
//! in isolation.

use std::collections::HashMap;

use crate::domain::models::CleanText;

/// Reorders `CleanText` arriving out of order and releases them by `seq`.
#[derive(Default)]
pub struct SequenceBuffer {
    expected_seq: u64,
    pending: HashMap<u64, CleanText>,
}

impl SequenceBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Processes an item and returns those now ready, in contiguous order.
    pub fn process(&mut self, item: CleanText) -> Vec<CleanText> {
        let mut ready = Vec::new();
        if item.seq == self.expected_seq {
            ready.push(item);
            self.expected_seq += 1;
            while let Some(next) = self.pending.remove(&self.expected_seq) {
                ready.push(next);
                self.expected_seq += 1;
            }
        } else {
            self.pending.insert(item.seq, item);
        }
        ready
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clean(seq: u64, text: &str) -> CleanText {
        CleanText {
            seq,
            text: text.to_string(),
        }
    }

    #[test]
    fn in_order_passes_through() {
        let mut buf = SequenceBuffer::new();
        assert_eq!(buf.process(clean(0, "a")), vec![clean(0, "a")]);
        assert_eq!(buf.process(clean(1, "b")), vec![clean(1, "b")]);
    }

    #[test]
    fn out_of_order_buffers_until_gap_fills() {
        let mut buf = SequenceBuffer::new();
        assert_eq!(buf.process(clean(1, "b")), vec![]);
        assert_eq!(buf.process(clean(2, "c")), vec![]);
        // Once 0 arrives, 0,1,2 drain in order.
        assert_eq!(
            buf.process(clean(0, "a")),
            vec![clean(0, "a"), clean(1, "b"), clean(2, "c")]
        );
    }

    #[test]
    fn stops_draining_at_first_gap() {
        let mut buf = SequenceBuffer::new();
        buf.process(clean(1, "b"));
        buf.process(clean(3, "d"));
        // 0 arrives -> drains 0,1 but stops (2 is missing).
        assert_eq!(
            buf.process(clean(0, "a")),
            vec![clean(0, "a"), clean(1, "b")]
        );
    }
}
