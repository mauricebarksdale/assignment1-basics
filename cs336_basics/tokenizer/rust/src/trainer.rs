use std::collections::BinaryHeap;
use std::{collections::HashMap};
use std::fmt;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use itertools::Itertools;

use crate::types::{BpeModel, Merge, PretokenCounts, TokenId, Vocab};

/// Training configuration, passed to `Tokenizer.train`. Holds no special tokens: those belong to
/// the tokenizer, which hands them to `train` so there is only one copy.
#[derive(Debug)]
#[pyclass(name = "BpeTrainer", frozen)]
pub struct BpeTrainer {
  vocab_size: u32,
}

#[pymethods]
impl BpeTrainer {
  #[new]
  pub fn new(vocab_size: u32) -> Self {
    Self { vocab_size }
  }

  #[getter]
  pub fn vocab_size(&self) -> u32 {
    self.vocab_size
  }
}

const NUM_BYTE_VALUES: u32 = 256;

impl BpeTrainer {
  /// Phase 2 of training: the merge loop. Counts in, (vocab, merges) out -- no files, no chunks,
  /// no pretokenizer. `special_tokens` are only needed to seed the vocabulary; they never appear in
  /// `pretoken_counts`.
  ///
  /// Invariants the result must satisfy (see TOKENIZER_DRIVEN_TRAINING.md):
  /// - all 256 single bytes have an id
  /// - `merges` is in creation order (it is the encoder's rank function)
  /// - ties are broken on pair bytes only, never on hash-map iteration order
  pub fn train(&self, pretoken_counts: PretokenCounts, special_tokens: &[String]) -> Result<BpeModel, VocabSizeTooSmall> {
    let num_special_tokens: u32 = special_tokens.len() as u32;
    let num_seed_tokens: u32 = num_special_tokens + NUM_BYTE_VALUES;
    let num_merges: u32 = self.vocab_size
      .checked_sub(num_seed_tokens)
      .ok_or(VocabSizeTooSmall { vocab_size: self.vocab_size, num_seed_tokens })?;

    let mut vocab: Vocab = (0..num_seed_tokens)
      .map(|i| {
        if i < num_special_tokens {
          (i, special_tokens[i as usize].as_bytes().to_owned())
        } else {
          (i, vec![(i - num_special_tokens) as u8])
        }
      })
      .collect();
    let mut merges: Vec<Merge> = Vec::new();

    let mut pretokens: Vec<(Vec<TokenId>, u64)> = Vec::with_capacity(pretoken_counts.len());
    let mut pair_counts: HashMap<(TokenId, TokenId), u64> = HashMap::new();
    // Indices into `pretokens` of every pretoken containing the pair, each listed once.
    let mut pair_index: HashMap<(TokenId, TokenId), Vec<usize>> = HashMap::new();
    let mut deltas: HashMap<(TokenId, TokenId), i64> = HashMap::new();
    for (pretoken_bytes, count) in pretoken_counts {
      let ids: Vec<TokenId> = pretoken_bytes
        .into_iter()
        .map(|byte| u32::from(byte) + num_special_tokens)
        .collect();
      let pretoken_index: usize = pretokens.len();
      for pair in ids.iter().copied().tuple_windows() {
        *pair_counts.entry(pair).or_insert(0) += count as u64;
        let indices: &mut Vec<usize> = pair_index.entry(pair).or_default();
        // Pretokens are walked one at a time, so a repeat within this one is always the last entry.
        if indices.last() != Some(&pretoken_index) {
          indices.push(pretoken_index);
        }
      }
      pretokens.push((ids, count as u64));
    }

    let mut max_heap: BinaryHeap<(u64, (TokenId, TokenId))> = BinaryHeap::from(
      pair_counts
        .iter()
        .map(|((left_id, right_id), count)| (*count, (*left_id, *right_id)))
        .collect::<Vec<_>>());

    for _ in 0..num_merges {
      let Some(winning_pair) = take_best_pair(&mut max_heap, &pair_counts, &vocab) else {
        break;
      };

      // Update vocab + merges
      let winning_pair_bytes: Merge = (vocab.get(&winning_pair.0).unwrap().clone(), vocab.get(&winning_pair.1).unwrap().clone());
      let winning_pair_id: TokenId = num_seed_tokens + merges.len() as u32;
      vocab.insert(winning_pair_id, [winning_pair_bytes.0.as_slice(), winning_pair_bytes.1.as_slice()].concat());
      merges.push(winning_pair_bytes);

      let pretokens_indices_with_winning_pair: Vec<usize> = pair_index
        .remove(&winning_pair)
        .expect("winning pair must be indexed");
      for pretoken_index in pretokens_indices_with_winning_pair {
        let (pretoken_ids, pretoken_count): &mut (Vec<TokenId>, u64) = pretokens.get_mut(pretoken_index).unwrap();
        let pretoken_count: i64 = *pretoken_count as i64;

        // Merge every occurrence left to right
        let mut merged_ids: Vec<TokenId> = Vec::with_capacity(pretoken_ids.len());
        let mut i: usize = 0;
        while i < pretoken_ids.len() {
          if i + 1 < pretoken_ids.len() && (pretoken_ids[i], pretoken_ids[i + 1]) == winning_pair {
            merged_ids.push(winning_pair_id);
            i += 2;
          } else {
            merged_ids.push(pretoken_ids[i]);
            i += 1;
          }
        }

        // Index entries aren't removed until a count hits zero, so this pretoken may have lost
        // the pair in an earlier merge.
        if merged_ids.len() == pretoken_ids.len() {
          continue;
        }

        for pair in pretoken_ids.iter().copied().tuple_windows() {
          *deltas.entry(pair).or_insert(0) -= pretoken_count;
        }
        for pair in merged_ids.iter().copied().tuple_windows::<(TokenId, TokenId)>() {
          *deltas.entry(pair).or_insert(0) += pretoken_count;
          if pair.0 == winning_pair_id || pair.1 == winning_pair_id {
            let indices: &mut Vec<usize> = pair_index.entry(pair).or_default();
            if indices.last() != Some(&pretoken_index) {
              indices.push(pretoken_index);
            }
          }
        }

        *pretoken_ids = merged_ids;
      }

      for (pair, delta) in deltas.drain() {
        if delta == 0 {
          continue;
        }
        let new_count: i64 = pair_counts.get(&pair).copied().unwrap_or(0) as i64 + delta;
        if new_count == 0 {
          pair_counts.remove(&pair);
          pair_index.remove(&pair);
        } else {
          pair_counts.insert(pair, new_count as u64);
          max_heap.push((new_count as u64, pair));
        }
      }
    }

    Ok(BpeModel { vocab, merges })
  }
}

fn take_best_pair(
  max_heap: &mut BinaryHeap<(u64, (TokenId, TokenId))>,
  pair_counts: &HashMap<(TokenId, TokenId), u64>,
  vocab: &Vocab,
) -> Option<(TokenId, TokenId)> {
  // Discard stale entries from the top, including pairs removed from `pair_counts` at zero.
  while let Some(&(count, (left_id, right_id))) = max_heap.peek() && pair_counts.get(&(left_id, right_id)) != Some(&count) {
    max_heap.pop();
  }

  if max_heap.is_empty() {
    return None;
  }

  let highest_count: u64 = max_heap.peek().unwrap().0;
  let mut merge_candidates: Vec<(u64, (TokenId, TokenId))> = Vec::new();

  // Stale entries can also sit at `highest_count`; only live ones become candidates.
  while let Some(&(count, (left_id, right_id))) = max_heap.peek() && count == highest_count {
    let candidate = max_heap.pop().unwrap();
    match pair_counts.get(&(left_id, right_id)) {
      Some(&current) if current == count => merge_candidates.push(candidate),
      current => debug_assert!(current.map_or(true, |current| count >= *current), "missed push"),
    }
  }

  let winning_pair: (TokenId, TokenId) = merge_candidates
    .iter()
    .max_by_key(|(_, (left_id, right_id))| (&vocab[left_id], &vocab[right_id]))
    .map(|(_, pair)| *pair)
    .unwrap();

  // The losers are still live, so they go back in.
  max_heap.extend(merge_candidates.into_iter().filter(|(_, pair)| *pair != winning_pair));

  Some(winning_pair)
}

/// `vocab_size` cannot hold the 256 byte tokens plus the special tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VocabSizeTooSmall {
  vocab_size: u32,
  num_seed_tokens: u32,
}

impl VocabSizeTooSmall {
  pub fn vocab_size(&self) -> u32 {
    self.vocab_size
  }

  pub fn num_seed_tokens(&self) -> u32 {
    self.num_seed_tokens
  }
}

impl fmt::Display for VocabSizeTooSmall {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(
      f,
      "vocab_size {} is smaller than the {} seed tokens (256 bytes plus special tokens)",
      self.vocab_size, self.num_seed_tokens,
    )
  }
}

impl std::error::Error for VocabSizeTooSmall {}

// `vocab_size` is a caller argument, so this is a `ValueError` on the Python side.
impl From<VocabSizeTooSmall> for PyErr {
  fn from(err: VocabSizeTooSmall) -> Self {
    PyValueError::new_err(err.to_string())
  }
}