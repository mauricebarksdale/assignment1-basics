use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyIterator, PyType};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, ErrorKind, Read, Seek, SeekFrom, Result};
use std::iter::{successors, from_fn};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread::available_parallelism;
use itertools::Itertools;

use crate::pretokenizer::{Pretoken, Pretokenizer, PyPretokenizer};
use crate::trainer::{BpeTrainer, VocabSizeTooSmall};
use crate::types::{Merge, PretokenCounts, TokenBytes, TokenId, Vocab, BpeModel};

const MINI_CHUNK_SIZE: usize = 4096;

/// Everything produced by training, kept together so it is either absent or complete. The
/// derived maps are only ever built from `vocab` and `merges` here, so they cannot drift.
#[derive(Debug)]
struct Model {
  vocab: Vocab,
  merges: Vec<Merge>,
  token_to_id: HashMap<TokenBytes, TokenId>,
  merge_ranks: HashMap<Merge, usize>,
}

#[derive(Debug)]
#[pyclass]
pub struct Tokenizer {
  model: Option<Model>,
  pretokenizer: Arc<Pretokenizer>,
}

/// The tokenizer has no model yet; call `train` first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Untrained;

impl fmt::Display for Untrained {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "Tokenizer is untrained; call train() first")
  }
}

impl std::error::Error for Untrained {}

// Calling a method before `train` is a state error, not a bad argument.
impl From<Untrained> for PyErr {
  fn from(err: Untrained) -> Self {
    PyRuntimeError::new_err(err.to_string())
  }
}

/// Why `Tokenizer::encode` failed.
#[derive(Debug, PartialEq, Eq)]
pub enum EncodeError {
  Untrained(Untrained),
  /// The text contains a special token that has no id in the vocabulary.
  UnknownSpecialToken(String),
}

impl fmt::Display for EncodeError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Untrained(err) => err.fmt(f),
      Self::UnknownSpecialToken(token) => write!(f, "special token {token:?} is not in the vocabulary"),
    }
  }
}

impl std::error::Error for EncodeError {
  fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
    match self {
      Self::Untrained(err) => Some(err),
      Self::UnknownSpecialToken(_) => None,
    }
  }
}

impl From<Untrained> for EncodeError {
  fn from(err: Untrained) -> Self {
    Self::Untrained(err)
  }
}

impl From<EncodeError> for PyErr {
  fn from(err: EncodeError) -> Self {
    match err {
      EncodeError::Untrained(err) => err.into(),
      EncodeError::UnknownSpecialToken(_) => PyValueError::new_err(err.to_string()),
    }
  }
}

/// Why `Tokenizer::decode` failed.
#[derive(Debug, PartialEq, Eq)]
pub enum DecodeError {
  Untrained(Untrained),
  /// An id that is not in the vocabulary.
  UnknownTokenId(TokenId),
}

impl fmt::Display for DecodeError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Untrained(err) => err.fmt(f),
      Self::UnknownTokenId(id) => write!(f, "token id {id} is not in the vocabulary"),
    }
  }
}

impl std::error::Error for DecodeError {
  fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
    match self {
      Self::Untrained(err) => Some(err),
      Self::UnknownTokenId(_) => None,
    }
  }
}

impl From<Untrained> for DecodeError {
  fn from(err: Untrained) -> Self {
    Self::Untrained(err)
  }
}

impl From<DecodeError> for PyErr {
  fn from(err: DecodeError) -> Self {
    match err {
      DecodeError::Untrained(err) => err.into(),
      DecodeError::UnknownTokenId(_) => PyValueError::new_err(err.to_string()),
    }
  }
}

/// Why `Tokenizer::train` failed.
#[derive(Debug)]
pub enum TrainError {
  /// Reading the training file failed.
  Io(io::Error),
  /// The trainer's `vocab_size` cannot hold the seed tokens.
  VocabSizeTooSmall(VocabSizeTooSmall),
}

impl fmt::Display for TrainError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Io(err) => write!(f, "failed to read training data: {err}"),
      Self::VocabSizeTooSmall(err) => err.fmt(f),
    }
  }
}

impl std::error::Error for TrainError {
  fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
    match self {
      Self::Io(err) => Some(err),
      Self::VocabSizeTooSmall(err) => Some(err),
    }
  }
}

impl From<io::Error> for TrainError {
  fn from(err: io::Error) -> Self {
    Self::Io(err)
  }
}

impl From<VocabSizeTooSmall> for TrainError {
  fn from(err: VocabSizeTooSmall) -> Self {
    Self::VocabSizeTooSmall(err)
  }
}

impl From<TrainError> for PyErr {
  fn from(err: TrainError) -> Self {
    match err {
      // Keeps the io kind, so pyo3 raises the matching subclass (e.g. FileNotFoundError).
      TrainError::Io(err) => err.into(),
      TrainError::VocabSizeTooSmall(err) => err.into(),
    }
  }
}

impl Tokenizer {
  pub fn new(vocab: Vocab, merges: Vec<Merge>, special_tokens: Vec<String>) -> Self {
    Self {
      model: Some(Model::new(vocab, merges)),
      pretokenizer: Arc::new(Pretokenizer::new(special_tokens)),
    }
  }

  /// A tokenizer with no model yet; `train` gives it one.
  pub fn untrained(pretokenizer: Pretokenizer) -> Self {
    Self { model: None, pretokenizer: Arc::new(pretokenizer) }
  }

  /// Trains on `input_path`, replacing any existing model. Phase 1 (pretoken counting) runs here so
  /// it uses the same `Pretokenizer` as encoding; the merge loop is `BpeTrainer::train`.
  pub fn train(&mut self, input_path: &Path, trainer: &BpeTrainer) -> std::result::Result<(), TrainError> {
    self.model = Some(self.train_model(input_path, trainer)?);
    Ok(())
  }

  pub fn vocab(&self) -> std::result::Result<&Vocab, Untrained> {
    Ok(&self.model()?.vocab)
  }

  /// Merges in creation order.
  pub fn merges(&self) -> std::result::Result<&[Merge], Untrained> {
    Ok(&self.model()?.merges)
  }

  pub fn special_tokens(&self) -> &[String] {
    self.pretokenizer.special_tokens()
  }

  pub fn encode(&self, text: &str) -> std::result::Result<Vec<TokenId>, EncodeError> {
    let model: &Model = self.model()?;
    let mut ids: Vec<TokenId> = Vec::new();
    for pretoken in self.pretokenizer.pretokenize(text) {
      match pretoken {
        // Resolved here and never merged: the model only ever sees ordinary pretokens.
        Pretoken::Special(s) => ids.push(model.token_to_id(s.as_bytes())
          .ok_or_else(|| EncodeError::UnknownSpecialToken(s.to_owned()))?),
        Pretoken::Normal(s) => ids.extend(model.merge_pretoken(s)),
      }
    }
    Ok(ids)
  }

  /// Invalid UTF-8 in the decoded bytes becomes U+FFFD.
  pub fn decode(&self, ids: &[TokenId]) -> std::result::Result<String, DecodeError> {
    let model: &Model = self.model()?;
    let mut bytes: Vec<u8> = Vec::new();
    for &id in ids {
      let token: &[u8] = model.id_to_token(id).ok_or(DecodeError::UnknownTokenId(id))?;
      bytes.extend_from_slice(token);
    }
    Ok(String::from_utf8_lossy(&bytes).into_owned())
  }
}

#[pymethods]
impl Tokenizer {
  #[new]
  #[pyo3(signature = (vocab, merges, special_tokens=None))]
  fn py_new(vocab: Vocab, merges: Vec<Merge>, special_tokens: Option<Vec<String>>) -> Self {
    Self::new(vocab, merges, special_tokens.unwrap_or_default())
  }

  #[classmethod]
  #[pyo3(name = "untrained")]
  fn py_untrained(_cls: &Bound<'_, PyType>, pretokenizer: PyRef<'_, PyPretokenizer>) -> Self {
    // Shares the Python pretokenizer's instance rather than copying it.
    Self { model: None, pretokenizer: Arc::clone(&pretokenizer.inner) }
  }

  #[classmethod]
  #[pyo3(signature = (vocab_filepath, merges_filepath, special_tokens=None))]
  fn from_files(_cls: &Bound<'_, PyType>, vocab_filepath: PathBuf, merges_filepath: PathBuf, special_tokens: Option<Vec<String>>) -> PyResult<Self> {
    let vocab: Vocab = Self::read_vocab_file(vocab_filepath)?;
    let merges: Vec<Merge> = Self::read_merges_file(merges_filepath)?;
    Ok(Self::new(vocab, merges, special_tokens.unwrap_or_default()))
  }

  #[pyo3(name = "train")]
  fn py_train(&mut self, py: Python<'_>, input_path: PathBuf, trainer: PyRef<'_, BpeTrainer>) -> PyResult<()> {
    let trainer: &BpeTrainer = &trainer;
    let this: &Self = self;
    let model: Model = py.detach(|| this.train_model(&input_path, trainer))?;
    self.model = Some(model);
    Ok(())
  }

  #[getter(vocab)]
  fn py_vocab(&self) -> PyResult<Vocab> {
    Ok(self.vocab()?.clone())
  }

  #[getter(merges)]
  fn py_merges(&self) -> PyResult<Vec<Merge>> {
    Ok(self.merges()?.to_vec())
  }

  #[getter(special_tokens)]
  fn py_special_tokens(&self) -> Vec<String> {
    self.special_tokens().to_vec()
  }

  #[pyo3(name = "encode")]
  fn py_encode(&self, text: &str) -> PyResult<Vec<TokenId>> {
    Ok(self.encode(text)?)
  }

  fn encode_iterable(slf: Bound<'_, Self>, iterable: Bound<'_, PyAny>) -> PyResult<TokenIterator> {
    // Fail at the call rather than on the first pull.
    slf.borrow().model()?;
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
                Ok(tokenizer_ref.encode(text.extract()?)?)))
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

  #[pyo3(name = "decode")]
  fn py_decode(&self, ids: Vec<TokenId>) -> PyResult<String> {
    Ok(self.decode(&ids)?)
  }
}

impl Tokenizer {
  fn model(&self) -> std::result::Result<&Model, Untrained> {
    self.model.as_ref().ok_or(Untrained)
  }

  fn train_model(&self, input_path: &Path, trainer: &BpeTrainer) -> std::result::Result<Model, TrainError> {
    let desired_num_chunks: NonZeroUsize = available_parallelism().unwrap_or(NonZeroUsize::MIN);
    let pretoken_counts: PretokenCounts = Self::count_pretokens(&self.pretokenizer, input_path, desired_num_chunks)?;
    let BpeModel { vocab, merges } = trainer.train(pretoken_counts, self.pretokenizer.special_tokens())?;
    Ok(Model::new(vocab, merges))
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
}

// Phase 1 of training: split the file into chunks on the first special token, pretokenize each
// chunk in parallel, and reduce to one pretoken -> count map.
impl Tokenizer {
  fn count_pretokens(pretokenizer: &Pretokenizer, input_path: &Path, desired_num_chunks: NonZeroUsize) -> io::Result<PretokenCounts> {
    let chunk_boundaries: Vec<u64> = match pretokenizer.special_tokens().first() {
      Some(split_special_token) => Self::find_chunk_boundaries(input_path, desired_num_chunks, split_special_token.as_bytes())?,
      None => Self::get_whole_file_boundaries(fs::metadata(input_path)?.len()),
    };
    Self::get_pretoken_frequencies(pretokenizer, input_path, chunk_boundaries)
  }

  /// Boundaries for a single chunk spanning the file: [0, file_size], or [0] if the file is empty.
  fn get_whole_file_boundaries(file_size: u64) -> Vec<u64> {
    if file_size == 0 { vec![0] } else { vec![0, file_size] }
  }

  /// Returns strictly increasing offsets from 0 to file_size; every interior offset is the start
  /// of a split_special_token. May return fewer than desired_num_chunks + 1 offsets, and returns
  /// [0] (zero chunks) for an empty file.
  fn find_chunk_boundaries(input_path: &Path, desired_num_chunks: NonZeroUsize, split_special_token: &[u8]) -> io::Result<Vec<u64>> {
    let mut file: File = File::open(input_path)?;
    let file_size: u64 = file.metadata()?.len();
    let chunk_size: u64 = file_size / desired_num_chunks.get() as u64;
    // fewer bytes than chunks
    if chunk_size == 0 {
      return Ok(Self::get_whole_file_boundaries(file_size));
    }

    let mut chunk_boundaries: Vec<u64> = vec![0];
    for chunk_index in 1..desired_num_chunks.get() as u64 {
      let target_offset: u64 = chunk_index * chunk_size;
      if chunk_boundaries.last().is_some_and(|&last_boundary| last_boundary >= target_offset) {
        continue;
      }
      match Self::find_next_token_offset(&mut file, target_offset, split_special_token)? {
        Some(token_offset) => chunk_boundaries.push(token_offset),
        None => break,
      }
    }
    chunk_boundaries.push(file_size);

    Ok(chunk_boundaries)
  }

  /// Offset of the first occurrence of token that starts at or after search_start, or None if
  /// there is none before EOF. Scans to EOF if necessary.
  fn find_next_token_offset<R: Read + Seek>(reader: &mut R, search_start: u64, token: &[u8]) -> io::Result<Option<u64>> {
    let token_len: usize = token.len();
    if !(1..=MINI_CHUNK_SIZE).contains(&token_len) {
      return Err(io::Error::new(ErrorKind::InvalidInput, "token length must be between 1 and MINI_CHUNK_SIZE"));
    }
    // longest prefix of a token that can sit at the end of the buffer without matching yet
    let max_partial_token_len: usize = token_len - 1;

    reader.seek(SeekFrom::Start(search_start))?;
    let mut buffer: Vec<u8> = vec![0; MINI_CHUNK_SIZE];
    let mut buffer_offset: u64 = search_start;
    let mut num_readable_bytes: usize = 0;

    loop {
      // leaves room to read into, so Ok(0) really means EOF
      debug_assert!(num_readable_bytes <= max_partial_token_len);
      match reader.read(&mut buffer[num_readable_bytes..]) {
        Ok(0) => return Ok(None),
        Ok(bytes_read) => num_readable_bytes += bytes_read,
        Err(e) if e.kind() == ErrorKind::Interrupted => continue,
        Err(e) => return Err(e),
      }

      let readable_bytes: &[u8] = &buffer[..num_readable_bytes];
      if let Some(match_index) = readable_bytes.windows(token_len).position(|window| window == token) {
        return Ok(Some(buffer_offset + match_index as u64));
      }

      // drop the bytes that can't start a token; keep the rest in case a token straddles reads
      let num_consumed_bytes: usize = num_readable_bytes.saturating_sub(max_partial_token_len);
      buffer.copy_within(num_consumed_bytes..num_readable_bytes, 0);
      buffer_offset += num_consumed_bytes as u64;
      num_readable_bytes -= num_consumed_bytes;
    }
  }

  fn get_pretoken_frequencies(pretokenizer: &Pretokenizer, input_path: &Path, chunk_boundaries: Vec<u64>) -> Result<PretokenCounts> {
    let chunk_ranges: Vec<(u64, u64)> = chunk_boundaries
      .into_iter()
      .tuple_windows() 
      .collect();
    chunk_ranges
      .par_iter()
      .try_fold(
        HashMap::new,
        |mut frequencies: PretokenCounts, &(start, end)| -> Result<PretokenCounts> {
          Self::count_chunk_pretokens(pretokenizer, input_path, &mut frequencies, start, end)?;
          Ok(frequencies)
        })
      .try_reduce(
        HashMap::new,
        |m1, m2| {
          let (mut larger, smaller) = if m1.len() >= m2.len() { (m1, m2) } else { (m2, m1) };
          for (pretoken, count) in smaller {
            *larger.entry(pretoken).or_insert(0) += count;
          }
          Ok(larger)
        }
      )
  }

  fn count_chunk_pretokens(pretokenizer: &Pretokenizer, input_path: &Path, frequencies: &mut PretokenCounts, start: u64, end: u64) -> Result<()> {
    let mut file: File = File::open(input_path)?;
    let mut buffer: Vec<u8> = vec![0; (end - start) as usize];
    file.seek(SeekFrom::Start(start))?;
    file.read_exact(&mut buffer)?;
    // Boundaries sit on special-token starts, so each chunk is valid UTF-8 and this borrows.
    let text = String::from_utf8_lossy(&buffer);

    for pretoken in pretokenizer.pretokenize(&text) {
      match pretoken {
        Pretoken::Normal(s) => {
          let bytes: &[u8] = s.as_bytes();
          match frequencies.get_mut(bytes) {
            Some(count) => *count += 1,
            None => { frequencies.insert(bytes.to_vec(), 1); }
          }
        }
        Pretoken::Special(_) => continue,
      }
    }

    Ok(())
  }
}

impl Model {
  fn new(vocab: Vocab, merges: Vec<Merge>) -> Self {
    let token_to_id: HashMap<TokenBytes, TokenId> = Self::create_token_to_id(&vocab);
    let merge_ranks: HashMap<Merge, usize> = Self::create_merge_ranks(&merges);
    Self { vocab, merges, token_to_id, merge_ranks }
  }

  fn create_token_to_id(vocab: &Vocab) -> HashMap<TokenBytes, TokenId> {
    vocab
      .iter()
      .map(|(id, bytes)| (bytes.clone(), *id))
      .collect()
  }

  fn create_merge_ranks(merges: &[Merge]) -> HashMap<Merge, usize> {
    merges
      .iter()
      .cloned()
      .enumerate()
      .map(|(rank, merge)| (merge, rank))
      .collect()
  }

  fn token_to_id(&self, token: &[u8]) -> Option<TokenId> {
    self.token_to_id.get(token).copied()
  }

  fn id_to_token(&self, id: TokenId) -> Option<&[u8]> {
    self.vocab.get(&id).map(Vec::as_slice)
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

  fn merge_step(&self, current_seq: &[TokenBytes]) -> Option<Vec<TokenBytes>> {
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

// Internals the public API can't reach: `train` picks the chunk count from the CPU count, so these
// pin the boundary rules and chunk-count independence directly.
#[cfg(test)]
mod tests {
  use super::*;
  use std::io::Write;

  const EOT: &str = "<|endoftext|>";

  fn write_fixture(name: &str, contents: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("rust_tokenizer_unit_{}_{name}", std::process::id()));
    File::create(&path).unwrap().write_all(contents.as_bytes()).unwrap();
    path
  }

  fn chunks(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
  }

  #[test]
  fn chunk_boundaries_land_on_special_tokens() {
    let contents = format!("aaaa{EOT}bbbb{EOT}cccc{EOT}dddd");
    let path = write_fixture("boundaries", &contents);
    let boundaries = Tokenizer::find_chunk_boundaries(&path, chunks(4), EOT.as_bytes()).unwrap();
    assert_eq!(boundaries.first(), Some(&0));
    assert_eq!(boundaries.last(), Some(&(contents.len() as u64)));
    assert!(boundaries.windows(2).all(|w| w[0] < w[1]));
    for &b in &boundaries[1..boundaries.len() - 1] {
      assert!(contents[b as usize..].starts_with(EOT), "boundary {b} is not on a special token");
    }
  }

  #[test]
  fn pretoken_counts_skip_special_tokens() {
    let path = write_fixture("counts", &format!("the cat{EOT}the cat{EOT} the"));
    let pretokenizer = Pretokenizer::new(vec![EOT.into()]);
    let counts = Tokenizer::count_pretokens(&pretokenizer, &path, chunks(1)).unwrap();
    let expected: PretokenCounts = [
      (b"the".to_vec(), 2),
      (b" cat".to_vec(), 2),
      (b" the".to_vec(), 1),
    ].into_iter().collect();
    assert_eq!(counts, expected);
  }

  #[test]
  fn pretoken_counts_independent_of_chunk_count() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../tests/fixtures/tinystories_sample.txt");
    let pretokenizer = Pretokenizer::new(vec![EOT.into()]);
    let reference = Tokenizer::count_pretokens(&pretokenizer, &path, chunks(1)).unwrap();
    assert!(!reference.is_empty());
    for n in [2, 3, 8, 64] {
      assert_eq!(Tokenizer::count_pretokens(&pretokenizer, &path, chunks(n)).unwrap(), reference, "{n} chunks");
    }
  }
}
