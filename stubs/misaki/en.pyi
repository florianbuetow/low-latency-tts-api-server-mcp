from collections.abc import Callable

from misaki.token import MToken

class G2P:
    def __init__(
        self,
        version: str | None = ...,
        trf: bool = ...,
        british: bool = ...,
        fallback: Callable[[MToken], tuple[str, int]] | None = ...,
        unk: str = ...,
    ) -> None: ...
    def __call__(self, text: str, preprocess: bool = ...) -> tuple[str, list[MToken]]: ...
