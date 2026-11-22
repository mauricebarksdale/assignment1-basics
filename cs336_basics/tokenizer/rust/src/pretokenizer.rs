use std::sync::Arc;

use fancy_regex::Regex;
use itertools::Itertools;
use pyo3::prelude::*;

const PAT: &str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pretoken<'a> {
  Special(&'a str),
  Normal(&'a str),
}

/// Plain Rust pretokenizer. Owns the special tokens so training and encoding read them from
/// one place.
#[derive(Debug)]
pub struct Pretokenizer {
  special_tokens: Vec<String>,
  special_tokens_regex: Option<Regex>,
  pat: Regex,
}

impl Pretokenizer {
  pub fn new(special_tokens: Vec<String>) -> Self {
    Self {
      special_tokens_regex: Self::create_special_tokens_regex(&special_tokens),
      special_tokens,
      pat: Regex::new(PAT).expect("PAT is a valid regex"),
    }
  }

  pub fn special_tokens(&self) -> &[String] {
    &self.special_tokens
  }

  pub fn pretokenize<'a>(&self, text: &'a str) -> Vec<Pretoken<'a>> {
    let text_split_on_special_tokens: Vec<Pretoken<'a>> = match &self.special_tokens_regex {
      Some(special_tokens_regex) => Self::split_text_on_special_tokens(special_tokens_regex, text),
      None => vec![Pretoken::Normal(text)],
    };
    text_split_on_special_tokens
      .into_iter()
      .flat_map(|pretoken| match pretoken {
        Pretoken::Special(_) => vec![pretoken],
        Pretoken::Normal(s) => self.pat
          .find_iter(s)
          .map(|m| Pretoken::Normal(m.expect("Must be able to unpack match").as_str()))
          .collect(),
      })
      .collect()
  }

  fn create_special_tokens_regex(special_tokens: &[String]) -> Option<Regex> {
    if special_tokens.is_empty() {
      None
    } else {
      // Escape so tokens like "<|endoftext|>" match literally instead of as alternation,
      // and try longer tokens first since regex alternation is leftmost-first, not longest.
      let special_tokens_pat = special_tokens
        .iter()
        .sorted_by_key(|token| std::cmp::Reverse(token.len()))
        .map(|token| fancy_regex::escape(token))
        .join("|");
      // Every alternative is escaped, so this is a literal alternation and cannot fail to compile.
      Some(Regex::new(&special_tokens_pat).expect("escaped special tokens form a valid regex"))
    }
  }

  // `split` discards the matched text, but special tokens need to survive as their
  // own pretoken (in position), so walk matches and keep the gaps *and* the matches.
  // Gaps come back as unsplit `Normal`s -- not real pretokens until `pretokenize`
  // runs PAT over them.
  fn split_text_on_special_tokens<'a>(regex: &Regex, text: &'a str) -> Vec<Pretoken<'a>> {
    let mut pretokens: Vec<Pretoken<'a>> = Vec::new();
    let mut last_end: usize = 0;
    for m in regex.find_iter(text) {
      let m: fancy_regex::Match<'_> = m.expect("Must be able to unpack match");
      if m.start() > last_end {
        pretokens.push(Pretoken::Normal(&text[last_end..m.start()]));
      }
      pretokens.push(Pretoken::Special(m.as_str()));
      last_end = m.end();
    }
    if last_end < text.len() {
      pretokens.push(Pretoken::Normal(&text[last_end..]));
    }
    pretokens
  }
}

/// Python handle for a `Pretokenizer`.
#[pyclass(name = "Pretokenizer", frozen)]
pub(crate) struct PyPretokenizer {
  // `pub(crate)` so a tokenizer can share this instance with `Arc::clone(&handle.inner)`; Python
  // never sees it, since only `#[pyo3(get)]` or a `#[getter]` would expose it.
  pub(crate) inner: Arc<Pretokenizer>,
}

#[pymethods]
impl PyPretokenizer {
  #[new]
  #[pyo3(signature = (special_tokens=None))]
  fn new(special_tokens: Option<Vec<String>>) -> Self {
    Self { inner: Arc::new(Pretokenizer::new(special_tokens.unwrap_or_default())) }
  }

  #[getter]
  fn special_tokens(&self) -> Vec<String> {
    self.inner.special_tokens().to_vec()
  }

  /// Returns `(piece, is_special)` pairs.
  fn pretokenize(&self, text: &str) -> Vec<(String, bool)> {
    self.inner
      .pretokenize(text)
      .into_iter()
      .map(|pretoken| match pretoken {
        Pretoken::Special(s) => (s.to_owned(), true),
        Pretoken::Normal(s) => (s.to_owned(), false),
      })
      .collect()
  }
}

// Rayon shares `&Pretokenizer` across worker threads during training.
const _: () = {
  const fn assert_send_sync<T: Send + Sync>() {}
  assert_send_sync::<Pretokenizer>();
};
