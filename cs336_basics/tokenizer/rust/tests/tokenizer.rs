use std::path::{Path, PathBuf};

use rust_tokenizer::pretokenizer::Pretokenizer;
use rust_tokenizer::tokenizer::Tokenizer;
use rust_tokenizer::trainer::BpeTrainer;
use rust_tokenizer::types::Merge;

const EOT: &str = "<|endoftext|>";
const NUM_BYTE_VALUES: u32 = 256;

fn write_fixture(name: &str, contents: &str) -> PathBuf {
  let path = std::env::temp_dir().join(format!("rust_tokenizer_{}_{name}", std::process::id()));
  std::fs::write(&path, contents).unwrap();
  path
}

fn train(input_path: &Path, vocab_size: u32, special_tokens: &[&str]) -> Tokenizer {
  let special_tokens: Vec<String> = special_tokens.iter().map(|s| s.to_string()).collect();
  let mut tokenizer = Tokenizer::untrained(Pretokenizer::new(special_tokens));
  tokenizer.train(input_path, &BpeTrainer::new(vocab_size)).unwrap();
  tokenizer
}

fn merge(left: &str, right: &str) -> Merge {
  (left.as_bytes().to_vec(), right.as_bytes().to_vec())
}

#[test]
fn documents_split_on_special_tokens_train_as_whole_pretokens() {
  // If chunking cut a document mid-word, "aaaa" would be counted as smaller pieces and the
  // (xx, xx) merges would never appear.
  let path = write_fixture("boundaries", &format!("aaaa{EOT}bbbb{EOT}cccc{EOT}dddd"));
  let tokenizer = train(&path, 1000, &[EOT]);
  assert_eq!(
    tokenizer.merges().unwrap(),
    &[
      merge("d", "d"), merge("c", "c"), merge("b", "b"), merge("a", "a"),
      merge("dd", "dd"), merge("cc", "cc"), merge("bb", "bb"), merge("aa", "aa"),
    ],
  );
}

#[test]
fn file_without_special_token_trains_like_whole_file() {
  let path = write_fixture("no_special", "no special tokens anywhere in here");
  let with_special = train(&path, 1000, &[EOT]);
  let without_special = train(&path, 1000, &[]);
  assert!(!with_special.merges().unwrap().is_empty());
  assert_eq!(with_special.merges().unwrap(), without_special.merges().unwrap());
}

#[test]
fn files_smaller_than_chunk_count_still_train() {
  let tiny = write_fixture("tiny", "ab");
  assert_eq!(train(&tiny, 1000, &[EOT]).merges().unwrap(), &[merge("a", "b")]);

  let empty = write_fixture("empty", "");
  let tokenizer = train(&empty, 1000, &[EOT]);
  assert!(tokenizer.merges().unwrap().is_empty());
  assert_eq!(tokenizer.vocab().unwrap().len(), NUM_BYTE_VALUES as usize + 1);
}

#[test]
fn special_tokens_are_not_counted_as_training_text() {
  let path = write_fixture("counts", &format!("the cat{EOT}the cat{EOT} the"));
  let tokenizer = train(&path, 1000, &[EOT]);
  let leaked: Vec<&Merge> = tokenizer
    .merges()
    .unwrap()
    .iter()
    .filter(|(left, right)| [left, right].iter().any(|token| token.iter().any(|b| b"<|>".contains(b))))
    .collect();
  assert!(leaked.is_empty(), "special token bytes were merged: {leaked:?}");
  assert_eq!(
    tokenizer.vocab().unwrap().values().filter(|token| token.as_slice() == EOT.as_bytes()).count(),
    1,
  );
}

#[test]
fn training_on_fixture_is_deterministic() {
  let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../tests/fixtures/tinystories_sample.txt");
  let reference = train(&path, 500, &[EOT]);
  assert!(!reference.merges().unwrap().is_empty());
  for _ in 0..3 {
    let tokenizer = train(&path, 500, &[EOT]);
    assert_eq!(tokenizer.merges().unwrap(), reference.merges().unwrap());
    assert_eq!(tokenizer.vocab().unwrap(), reference.vocab().unwrap());
  }
}
