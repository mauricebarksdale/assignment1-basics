from os import PathLike
from typing import Iterable, Iterator, Self
import rust_tokenizer

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> Self:
        raise NotImplementedError

    @classmethod
    def from_files(
        cls, 
        vocab_filepath: str | PathLike[str], 
        merges_filepath: str | PathLike[str], 
        special_tokens: list[str] | None = None
    ) -> None:
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError