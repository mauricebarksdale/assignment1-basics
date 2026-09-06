#[pyo3::pymodule]
mod tokenizer {
  use pyo3::prelude::*;
  use pyo3::types::PyType;
  use std::collections::HashMap;
  use std::path::PathBuf;

  #[allow(dead_code)]
  #[pyclass]
  struct Tokenizer {
    vocab: HashMap<u32, Vec<u8>>,
    merges: Vec<(Vec<u8>, Vec<u8>)>,
    special_tokens: Option<Vec<String>>,
  }

  #[pymethods]
  impl Tokenizer {
    #[new]
    fn new(
      _vocab: HashMap<u32, Vec<u8>>,
      _merges: Vec<(Vec<u8>, Vec<u8>)>,
      _special_tokens: Option<Vec<String>>,
    ) -> PyResult<Self> {
      todo!()
    }

    #[classmethod]
    fn from_files(
      _cls: &Bound<'_, PyType>,
      _vocab_filepath: PathBuf,
      _merges_filepath: PathBuf,
      _special_tokens: Option<Vec<String>>,
    ) -> PyResult<Self> {
      todo!()
    }

    fn encode(&self, _text: &str) -> PyResult<Vec<u32>> {
      todo!()
    }

    fn encode_iterable(&self, _iterable;
      : Vec<String>) -> PyResult<Vec<u32>> {
      todo!()
    }

    fn decode(&self, _ids: Vec<u32>) -> PyResult<String> {
      todo!()
    }
  }
}
