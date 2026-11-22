use rust_tokenizer::trainer::BpeTrainer;
use rust_tokenizer::types::{Merge, PretokenCounts, TokenId};

const NUM_BYTE_VALUES: u32 = 256;

fn counts(pretokens: &[(&str, u32)]) -> PretokenCounts {
  pretokens.iter().map(|&(pretoken, count)| (pretoken.as_bytes().to_vec(), count)).collect()
}

fn merge(left: &str, right: &str) -> Merge {
  (left.as_bytes().to_vec(), right.as_bytes().to_vec())
}

#[test]
fn seeds_special_tokens_then_every_byte() {
  let special_tokens: Vec<String> = vec!["<|endoftext|>".into(), "<|pad|>".into()];
  let model = BpeTrainer::new(258).train(PretokenCounts::new(), &special_tokens).unwrap();

  assert!(model.merges.is_empty());
  assert_eq!(model.vocab.len(), 258);
  assert_eq!(model.vocab[&0], b"<|endoftext|>");
  assert_eq!(model.vocab[&1], b"<|pad|>");
  for byte in 0..=u8::MAX {
    assert_eq!(model.vocab[&(2 + TokenId::from(byte))], vec![byte], "byte {byte}");
  }
}

#[test]
fn merges_most_frequent_pair_weighted_by_count() {
  // (c, d) appears in one pretoken but that pretoken occurs 5 times; (a, b) totals 2.
  let model = BpeTrainer::new(NUM_BYTE_VALUES + 1)
    .train(counts(&[("ab", 1), ("xab", 1), ("cd", 5)]), &[])
    .unwrap();
  assert_eq!(model.merges, vec![merge("c", "d")]);
  assert_eq!(model.vocab[&NUM_BYTE_VALUES], b"cd");
}

#[test]
fn ties_break_on_lexicographically_greater_pair() {
  let model = BpeTrainer::new(NUM_BYTE_VALUES + 1)
    .train(counts(&[("ab", 3), ("ba", 3)]), &[])
    .unwrap();
  assert_eq!(model.merges, vec![merge("b", "a")]);
}

#[test]
fn merges_build_on_earlier_merges_in_creation_order() {
  let model = BpeTrainer::new(NUM_BYTE_VALUES + 2).train(counts(&[("abc", 4)]), &[]).unwrap();
  // (a, b) and (b, c) tie at 4; (b, c) is greater, then (a, bc) is the only pair left.
  assert_eq!(model.merges, vec![merge("b", "c"), merge("a", "bc")]);
  assert_eq!(model.vocab[&NUM_BYTE_VALUES], b"bc");
  assert_eq!(model.vocab[&(NUM_BYTE_VALUES + 1)], b"abc");
}

#[test]
fn overlapping_occurrences_merge_left_to_right() {
  // "aaa" -> [aa, a], so the second merge is (aa, a), not (a, aa).
  let model = BpeTrainer::new(NUM_BYTE_VALUES + 2).train(counts(&[("aaa", 1)]), &[]).unwrap();
  assert_eq!(model.merges, vec![merge("a", "a"), merge("aa", "a")]);
}

#[test]
fn merged_ids_follow_special_tokens() {
  let special_tokens: Vec<String> = vec!["<|endoftext|>".into()];
  let model = BpeTrainer::new(NUM_BYTE_VALUES + 2).train(counts(&[("ab", 1)]), &special_tokens).unwrap();
  assert_eq!(model.vocab[&(NUM_BYTE_VALUES + 1)], b"ab");
}

#[test]
fn stops_early_when_no_pairs_remain() {
  let model = BpeTrainer::new(1000).train(counts(&[("ab", 1), ("c", 9)]), &[]).unwrap();
  assert_eq!(model.merges, vec![merge("a", "b")]);
  assert_eq!(model.vocab.len(), NUM_BYTE_VALUES as usize + 1);
}

#[test]
fn vocab_size_below_seed_tokens_is_an_error() {
  let special_tokens: Vec<String> = vec!["<|endoftext|>".into()];
  let err = BpeTrainer::new(NUM_BYTE_VALUES).train(counts(&[("ab", 1)]), &special_tokens).unwrap_err();
  assert_eq!((err.vocab_size(), err.num_seed_tokens()), (NUM_BYTE_VALUES, NUM_BYTE_VALUES + 1));
}
