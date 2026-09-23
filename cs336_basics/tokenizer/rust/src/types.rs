use std::collections::HashMap;

pub(crate) type TokenId = u32;
pub(crate) type TokenBytes = Vec<u8>;
pub(crate) type Merge = (TokenBytes, TokenBytes);
pub(crate) type Vocab = HashMap<TokenId, TokenBytes>;
pub(crate) type TrainBpeResult = (Vocab, Vec<Merge>);