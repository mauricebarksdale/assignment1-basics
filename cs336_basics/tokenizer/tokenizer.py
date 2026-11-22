from os import PathLike
from typing import Iterable, Iterator
import rust_tokenizer

from cs336_basics.tokenizer.pretokenizer import Pretokenizer
from cs336_basics.tokenizer.trainer import BpeTrainer

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self._tokenizer = rust_tokenizer.Tokenizer(vocab, merges, special_tokens)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | PathLike[str],
        merges_filepath: str | PathLike[str],
        special_tokens: list[str] | None = None
    ) -> "Tokenizer":
        instance = cls.__new__(cls)
        instance._tokenizer = rust_tokenizer.Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
        return instance

    @classmethod
    def untrained(cls, pretokenizer: Pretokenizer) -> "Tokenizer":
        instance = cls.__new__(cls)
        instance._tokenizer = rust_tokenizer.Tokenizer.untrained(pretokenizer._pretokenizer)
        return instance

    def train(self, input_path: str | PathLike[str], trainer: BpeTrainer) -> None:
        self._tokenizer.train(input_path, trainer._trainer)

    @property
    def vocab(self) -> dict[int, bytes]:
        return self._tokenizer.vocab

    @property
    def merges(self) -> list[tuple[bytes, bytes]]:
        return self._tokenizer.merges

    @property
    def special_tokens(self) -> list[str]:
        return self._tokenizer.special_tokens

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        return self._tokenizer.encode_iterable(iterable)

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids)