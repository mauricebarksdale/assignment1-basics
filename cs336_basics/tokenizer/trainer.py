import rust_tokenizer


class BpeTrainer:
    """Training configuration for `Tokenizer.train`. Special tokens come from the tokenizer."""

    def __init__(self, vocab_size: int):
        self._trainer = rust_tokenizer.BpeTrainer(vocab_size)

    @property
    def vocab_size(self) -> int:
        return self._trainer.vocab_size
