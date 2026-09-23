mod pretokenize;
mod types;
mod tokenizer;

#[pyo3::pymodule(name = "rust_tokenizer")]
mod py_module {
  #[pymodule_export]
  use crate::tokenizer::Tokenizer;
  #[pymodule_export]
  use crate::tokenizer::TokenIterator;
}