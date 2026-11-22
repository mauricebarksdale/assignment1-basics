use std::collections::HashMap;

pub type TokenId = u32;
pub type PretokenBytes = Vec<u8>;
pub type TokenBytes = Vec<u8>;
pub type Merge = (TokenBytes, TokenBytes);
pub type Vocab = HashMap<TokenId, TokenBytes>;

#[derive(Debug)]
pub struct BpeModel {
  pub vocab: Vocab,
  pub merges: Vec<Merge>
}

pub type PretokenCounts = HashMap<PretokenBytes, u32>;