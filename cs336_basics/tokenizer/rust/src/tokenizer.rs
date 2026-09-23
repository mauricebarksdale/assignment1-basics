use pyo3::exceptions::{PyValueError,PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyIterator, PyType};
use std::collections::HashMap;
use std::iter::{successors, from_fn};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use itertools::Itertools;

use crate::pretokenize::{Pretokenizer, Pretoken};
use crate::types::{Merge, TokenBytes, TokenId, Vocab};

#[pyclass]
pub struct Tokenizer {
  vocab: Vocab,
  #[allow(dead_code)]
  merges: Vec<Merge>,
  token_to_id: HashMap<TokenBytes, TokenId>,
  merge_ranks: HashMap<Merge, usize>,
  pretokenizer: Pretokenizer,
}

#[pymethods]
impl Tokenizer {

  #[new]
  fn new(vocab: Vocab, merges: Vec<Merge>, special_tokens: Option<Vec<String>>) -> PyResult<Self> {
    let token_to_id: HashMap<TokenBytes, TokenId> = Self::create_token_to_id(&vocab);
    let merge_ranks: HashMap<Merge, usize> = Self::create_merge_ranks(&merges);
    let pretokenizer: Pretokenizer = Pretokenizer::new(special_tokens.as_deref().unwrap_or(&[]));

    Ok(Self {
      vocab,
      merges,
      token_to_id,
      merge_ranks,
      pretokenizer
    })
  }

  #[classmethod]
  fn from_files(_cls: &Bound<'_, PyType>, vocab_filepath: PathBuf, merges_filepath: PathBuf, special_tokens: Option<Vec<String>>) -> PyResult<Self> {
    let vocab: Vocab = Self::read_vocab_file(vocab_filepath)?;
    let merges: Vec<Merge> = Self::read_merges_file(merges_filepath)?;
    Self::new(vocab, merges, special_tokens)
  }

  fn encode(&self, text: &str) -> Vec<TokenId> {
    let pretokenized_text: Vec<Pretoken> = self.pretokenizer.pretokenize(text);
    self.tokenize_text(pretokenized_text)
  }

  fn encode_iterable(slf: Bound<'_, Self>, iterable: Bound<'_, PyAny>) -> PyResult<TokenIterator> {
    let tokenizer: Py<Tokenizer> = slf.unbind();
    match iterable.try_iter() {
      Ok(iterator) => {
        // A `Bound` is tied to this call's GIL lifetime and isn't `Send`, so unbind it
        // and re-bind inside `Python::attach` on each pull.
        let iterator: Py<PyIterator> = iterator.unbind();
        let tokens = from_fn(move || {
            Python::attach(|py| {
              let text = iterator.bind(py).clone().next()?;
              let tokenizer_ref = tokenizer.bind(py).borrow();
              Some(text.and_then(|text|
                Ok(tokenizer_ref.encode(text.extract()?))))
            })
          })
          .flat_map(|chunk|
            match chunk {
              Ok(ids) => ids.into_iter().map(Ok).collect::<Vec<PyResult<TokenId>>>(),
              Err(err) => vec![Err(err)],
            }
          );
        Ok(TokenIterator { iter: Mutex::new(Box::new(tokens)) })
      }
      Err(_) => Err(PyTypeError::new_err("Expected an iterable object"))
    }
  }

  fn decode(&self, ids: Vec<TokenId>) -> String {
    let tokens: Vec<TokenBytes> = ids
      .into_iter()
      .map(|token| self.vocab.get(&token).unwrap().clone())
      .collect();
    let bytes: Vec<u8> = tokens.into_iter().flatten().collect();
    String::from_utf8_lossy(&bytes).into_owned()
  }
}

impl Tokenizer {
  fn create_token_to_id(vocab: &Vocab) -> HashMap<TokenBytes, TokenId> {
    vocab
      .clone()
      .into_iter()
      .map(|(id, bytes)| (bytes, id))
      .collect()
  }

  fn create_merge_ranks(merges: &Vec<Merge>) -> HashMap<Merge, usize> {
    merges
        .clone()
        .into_iter()
        .enumerate()
        .map(|(rank, merge)| (merge, rank))
        .collect()
  }

  fn read_file(filepath: &Path) -> PyResult<String> {
    // Wrap the io::Error to put the path in the message; keeping its kind lets pyo3
    // raise the matching Python subclass (e.g. FileNotFoundError).
    fs::read_to_string(filepath)
      .map_err(|err| io::Error::new(err.kind(), format!("{}: {err}", filepath.display())).into())
  }

  fn read_vocab_file(vocab_filepath: PathBuf) -> PyResult<Vocab> {
    let vocab_str: String = Self::read_file(&vocab_filepath)?;
    serde_json::from_str(&vocab_str)
      .map_err(|err| PyValueError::new_err(format!("{}: invalid vocab JSON: {err}", vocab_filepath.display())))
  }

  fn read_merges_file(merges_filepath: PathBuf) -> PyResult<Vec<Merge>> {
    let merges_str: String = Self::read_file(&merges_filepath)?;
    merges_str
      .lines()
      .enumerate()
      .map(|(index, line)| {
        line.split_whitespace()
          .map(|s| s.as_bytes().to_vec())
          .collect_tuple()
          .ok_or_else(|| PyValueError::new_err(format!(
            "{}:{}: expected two space-separated tokens, got {line:?}",
            merges_filepath.display(),
            index + 1,
          )))
      })
      .collect()
  }

  fn tokenize_text(&self, pretokenized_text: Vec<Pretoken>) -> Vec<TokenId> {
    pretokenized_text
      .into_iter()
      .flat_map(|pretoken| match pretoken {
        Pretoken::Special(s) => vec![*self.token_to_id.get(s.as_bytes()).unwrap()],
        Pretoken::Normal(s) => self.merge_pretoken(s),
      })
      .collect()
  }

  fn merge_pretoken(&self, pretoken: &str) -> Vec<TokenId> {
    let mut current_seq: Vec<TokenBytes> = pretoken
      .as_bytes()
      .iter()
      .map(|byte| vec![*byte])
      .collect();
    current_seq = successors(Some(current_seq), |current_seq| self.merge_step(current_seq))
      .last()
      .unwrap();
    current_seq
      .into_iter()
      .map(|token_bytes| *self.token_to_id.get(&token_bytes).unwrap())
      .collect()
  }

  fn merge_step(&self, current_seq: &Vec<TokenBytes>) -> Option<Vec<TokenBytes>> {
    let index_of_lowest_rank: Option<(usize, usize)> = current_seq
      .windows(2)
      .enumerate()
      .filter_map(|(index, pair)| {
        let left: TokenBytes = pair[0].to_vec();
        let right: TokenBytes = pair[1].to_vec();
        self.merge_ranks.get(&(left, right)).map(|rank| (index, *rank))
      })
      .min_by_key(|(_index, rank)| { *rank });
    match index_of_lowest_rank {
      Some((index, _rank)) => {
        let mut new_seq: Vec<TokenBytes> = Vec::from(&current_seq[0..index]);
        let merged: TokenBytes = current_seq[index..index+2].concat();
        new_seq.push(merged);
        let rest = current_seq[index+2..].iter().cloned();
        new_seq.extend(rest);
        Some(new_seq)
      }
      None => None
    }
  }
}

#[pyclass]
pub struct TokenIterator {
  iter: Mutex<Box<dyn Iterator<Item = PyResult<TokenId>> + Send>>,
}

#[pymethods]
impl TokenIterator {
  fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
      slf
  }
  fn __next__(slf: PyRefMut<'_, Self>) -> PyResult<Option<TokenId>> {
      slf.iter.lock().unwrap().next().transpose()
  }
}