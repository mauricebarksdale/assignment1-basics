import rust_tokenizer


class Pretokenizer:
    def __init__(self, special_tokens: list[str] | None = None):
        self._pretokenizer = rust_tokenizer.Pretokenizer(special_tokens)

    @property
    def special_tokens(self) -> list[str]:
        return self._pretokenizer.special_tokens

    def pretokenize(self, text: str) -> list[tuple[str, bool]]:
        """Split text into (piece, is_special) pairs."""
        return self._pretokenizer.pretokenize(text)
