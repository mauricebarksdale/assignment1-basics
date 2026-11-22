pub mod pretokenizer;
pub mod tokenizer;
pub mod trainer;
pub mod types;

#[pyo3::pymodule(name = "rust_tokenizer")]
mod py_module {
  #[pymodule_export]
  use crate::tokenizer::Tokenizer;
  #[pymodule_export]
  use crate::tokenizer::TokenIterator;
  #[pymodule_export]
  use crate::pretokenizer::PyPretokenizer;
  #[pymodule_export]
  use crate::trainer::BpeTrainer;
}
