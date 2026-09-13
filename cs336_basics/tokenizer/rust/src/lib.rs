#[pyo3::pymodule(name = "rust_tokenizer")]
mod tokenizer {
  use pyo3::exceptions::{PyValueError,PyTypeError};
  use pyo3::prelude::*;
  use pyo3::types::{PyIterator, PyType};
  use std::collections::{HashMap, HashSet};
  use std::iter::{successors, from_fn};
  use fancy_regex::Regex;
  use std::fs;
  use std::path::PathBuf;
  use std::sync::Mutex;
  use serde_json;
  use itertools::Itertools;

  #[pyclass]
  struct Tokenizer {
    vocab: HashMap<u32, Vec<u8>>,
    merges: Vec<(Vec<u8>, Vec<u8>)>,
    special_tokens: Option<Vec<String>>,
    special_token_set: HashSet<String>,
    token_to_id: HashMap<Vec<u8>, u32>,
    merge_ranks: HashMap<(Vec<u8>, Vec<u8>), usize>,
    special_tokens_regex: Option<Regex>,
    pretokenization_regex: Regex
  }

  #[pymethods]
  impl Tokenizer {
    
    const PAT: &str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
    
    #[new]
    fn new(vocab: HashMap<u32, Vec<u8>>, merges: Vec<(Vec<u8>, Vec<u8>)>, special_tokens: Option<Vec<String>>) -> PyResult<Self> {
      let token_to_id: HashMap<Vec<u8>, u32> = Self::create_token_to_id(&vocab);
      let merge_ranks: HashMap<(Vec<u8>, Vec<u8>), usize> = Self::create_merge_ranks(&merges);
      let special_token_set: HashSet<String> = Self::create_special_token_set(&special_tokens);
      let special_tokens_regex: Option<Regex> = Self::create_special_tokens_regex(&special_tokens);
      let pretokenization_regex: Regex = Self::create_regex(Self::PAT);
      
      Ok(Self {
        vocab, 
        merges, 
        special_tokens,
        special_token_set,
        token_to_id,
        merge_ranks,
        special_tokens_regex,
        pretokenization_regex
      })
    }

    #[classmethod]
    fn from_files(_cls: &Bound<'_, PyType>, vocab_filepath: PathBuf, merges_filepath: PathBuf, special_tokens: Option<Vec<String>>) -> PyResult<Self> {
      let vocab: HashMap<u32, Vec<u8>> = Self::read_vocab_file(vocab_filepath);
      let merges: Vec<(Vec<u8>, Vec<u8>)> = Self::read_merges_file(merges_filepath);
      let token_to_id: HashMap<Vec<u8>, u32> = Self::create_token_to_id(&vocab);
      let merge_ranks: HashMap<(Vec<u8>, Vec<u8>), usize> = Self::create_merge_ranks(&merges);
      let special_token_set: HashSet<String> = Self::create_special_token_set(&special_tokens);
      let special_tokens_regex: Option<Regex> = Self::create_special_tokens_regex(&special_tokens);
      let pretokenization_regex: Regex = Self::create_regex(Self::PAT);

      Ok(Self {
        vocab,
        merges,
        special_tokens,
        special_token_set,
        token_to_id,
        merge_ranks,
        special_tokens_regex,
        pretokenization_regex
      })
    }
    
    fn encode(&self, text: &str) -> Vec<u32> {
      let pretokenized_text: Vec<&str> = self.pretokenize_text(text);
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
                Ok(ids) => ids.into_iter().map(Ok).collect::<Vec<PyResult<u32>>>(),
                Err(err) => vec![Err(err)],
              }
            );
          Ok(TokenIterator { iter: Mutex::new(Box::new(tokens)) })
        }
        Err(_) => Err(PyTypeError::new_err("Expected an iterable object"))
      }
    }

    fn decode(&self, ids: Vec<u32>) -> String {
      let tokens: Vec<Vec<u8>> = ids
        .into_iter()
        .map(|token| self.vocab.get(&token).unwrap().clone())
        .collect();
      let bytes: Vec<u8> = tokens.into_iter().flatten().collect();
      String::from_utf8_lossy(&bytes).into_owned()
    }
  }

  impl Tokenizer {
    fn create_token_to_id(vocab: &HashMap<u32, Vec<u8>>) -> HashMap<Vec<u8>, u32> {
      vocab
        .clone()
        .into_iter()
        .map(|(id, bytes)| (bytes, id))
        .collect()
    }

    fn create_merge_ranks(merges: &Vec<(Vec<u8>, Vec<u8>)>) -> HashMap<(Vec<u8>, Vec<u8>), usize> {
      merges
          .clone()
          .into_iter()
          .enumerate()
          .map(|(rank, merge)| (merge, rank))
          .collect()
    }

    fn create_regex(expr: &str) -> Regex {
      Regex::new(expr).unwrap()
    }

    fn create_special_token_set(special_tokens: &Option<Vec<String>>) -> HashSet<String> {
      special_tokens.as_deref().unwrap_or(&[]).iter().cloned().collect()
    }

    fn create_special_tokens_regex(special_tokens: &Option<Vec<String>>) -> Option<Regex> {
      // Escape so tokens like "<|endoftext|>" match literally instead of as alternation,
      // and try longer tokens first since regex alternation is leftmost-first, not longest.
      let special_tokens_pat = special_tokens
        .as_deref()
        .filter(|tokens| !tokens.is_empty())?
        .iter()
        .sorted_by_key(|token| std::cmp::Reverse(token.len()))
        .map(|token| fancy_regex::escape(token))
        .join("|");
      Some(Self::create_regex(&special_tokens_pat))
    }

    fn read_vocab_file(vocab_filepath: PathBuf) -> HashMap<u32, Vec<u8>> {
      let vocab_str: String = fs::read_to_string(vocab_filepath)
        .expect("{vocab_filepath} should be able to be read");
      let vocab: HashMap<u32, Vec<u8>> = serde_json::from_str(&vocab_str)
        .map_err(|err| PyValueError::new_err(err.to_string()))
        .expect("{vocab_filepath} should be able to be parsed into a HashMap<u32, u8>");
      vocab
    }

    fn read_merges_file(merges_filepath: PathBuf) -> Vec<(Vec<u8>, Vec<u8>)> {
      let merges_str: String = fs::read_to_string(merges_filepath)
        .expect("{merges_filepath} cannot be read");
      
      merges_str
        .lines()
        .map(|line| {
          line.split_whitespace()
            .map(|s| s.as_bytes().to_vec())
            .collect_tuple()
            .expect("{merges_filepath} cannot be parsed into a tuple")
        })
        .collect()
    }

    fn pretokenize_text<'a>(&self, text: &'a str) -> Vec<&'a str> {
      // `split` discards the matched text, but special tokens need to survive as their
      // own piece (in position), so walk matches and keep the gaps *and* the matches.
      let text_split_on_special_tokens: Vec<&str> = match &self.special_tokens_regex {
        Some(special_tokens_regex) => Self::split_text_on_special_tokens(special_tokens_regex, text),
        None => vec![text],
      };
      text_split_on_special_tokens
        .into_iter()
        .flat_map(|s| self.pretokenize_non_special_tokens(s))
        .collect()
    }

    fn split_text_on_special_tokens<'a>(regex: &Regex, text: &'a str) -> Vec<&'a str> {
      let mut pieces: Vec<&str> = Vec::new();
      let mut last_end: usize = 0;
      for m in regex.find_iter(text) {
        let m: fancy_regex::Match<'_> = m.expect("Must be able to unpack match");
        if m.start() > last_end {
          pieces.push(&text[last_end..m.start()]);
        }
        pieces.push(m.as_str());
        last_end = m.end();
      }
      if last_end < text.len() {
        pieces.push(&text[last_end..]);
      }
      pieces
    }

    fn pretokenize_non_special_tokens<'a>(&self, s: &'a str) -> Vec<&'a str> {
      if self.special_token_set.contains(s) {
        vec![s]
      } else {
        self.pretokenization_regex
          .find_iter(s)
          .map(|m| m.expect("Must be able to unpack match").as_str())
          .collect()
      }
    }

    fn tokenize_text(&self, pretokenized_text: Vec<&str>) -> Vec<u32> {
      pretokenized_text
        .iter()
        .flat_map(|s|{
          if self.special_token_set.contains(*s) {
            vec![*self.token_to_id.get(s.as_bytes()).unwrap()]
          } else {
            self.merge_pretokens(s)
          }
        })
        .collect()
    }

    fn merge_pretokens(&self, pretokens: &str) -> Vec<u32> {
      let mut current_seq: Vec<Vec<u8>> = pretokens
        .as_bytes()
        .iter()
        .map(|byte| vec![*byte])
        .collect();
      current_seq = successors(Some(current_seq), |current_seq| self.merge_step(current_seq))
        .last()
        .unwrap();
      current_seq
        .into_iter()
        .map(|pretoken| *self.token_to_id.get(&pretoken).unwrap())
        .collect()
    }

    fn merge_step(&self, current_seq: &Vec<Vec<u8>>) -> Option<Vec<Vec<u8>>> {
      let index_of_lowest_rank: Option<(usize, usize)> = current_seq
        .windows(2)
        .enumerate()
        .filter_map(|(index, pair)| { 
          let left_pretoken: Vec<u8> = pair[0].to_vec();
          let right_pretoken: Vec<u8> = pair[1].to_vec();
          self.merge_ranks.get(&(left_pretoken, right_pretoken))
            .map(|rank| (index, *rank))
        })
        .min_by_key(|(_index, rank)| { *rank });
      match index_of_lowest_rank {
        Some((index, _rank)) => {
          let mut new_seq: Vec<Vec<u8>> = Vec::from(&current_seq[0..index]);
          let merged: Vec<u8> = current_seq[index..index+2].concat();
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
  struct TokenIterator {
    iter: Mutex<Box<dyn Iterator<Item = PyResult<u32>> + Send>>,
  }

  #[pymethods]
  impl TokenIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(slf: PyRefMut<'_, Self>) -> PyResult<Option<u32>> {
        slf.iter.lock().unwrap().next().transpose()
    }
  }
}