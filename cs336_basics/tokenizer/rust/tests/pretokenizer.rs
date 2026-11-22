use rust_tokenizer::pretokenizer::{Pretoken, Pretokenizer};

#[test]
fn pretokenize_without_special_tokens() {
  let pretokenizer = Pretokenizer::new(vec![]);
  assert_eq!(
    pretokenizer.pretokenize("Hello, world! It's 2024.<|endoftext|>"),
    vec![
      Pretoken::Normal("Hello"),
      Pretoken::Normal(","),
      Pretoken::Normal(" world"),
      Pretoken::Normal("!"),
      Pretoken::Normal(" It"),
      Pretoken::Normal("'s"),
      Pretoken::Normal(" 2024"),
      Pretoken::Normal(".<|"),
      Pretoken::Normal("endoftext"),
      Pretoken::Normal("|>"),
    ],
  );
}

#[test]
fn pretokenize_with_special_tokens() {
  let pretokenizer = Pretokenizer::new(vec!["<|endoftext|>".into(), "<|endoftext|><|endoftext|>".into()]);
  assert_eq!(
    pretokenizer.pretokenize("a b<|endoftext|><|endoftext|>c  <|endoftext|>"),
    vec![
      Pretoken::Normal("a"),
      Pretoken::Normal(" b"),
      Pretoken::Special("<|endoftext|><|endoftext|>"),
      Pretoken::Normal("c"),
      Pretoken::Normal("  "),
      Pretoken::Special("<|endoftext|>"),
    ],
  );
}
