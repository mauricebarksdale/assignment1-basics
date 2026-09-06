from typing import NamedTuple

class TrainBpeResult(NamedTuple):
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> TrainBpeResult:
    raise NotImplementedError