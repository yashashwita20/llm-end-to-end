from .tokenizer import Tokenizer

class ByteTokenizer(Tokenizer):
    """
    Represents a string as a sequence of Bytes.
    1. Converts text to it's UTF-8 encoding
    2. Converts each byte's UTF-8 hex codes to decimal
    """

    def encode(self, string: str) -> list[int]:
        utf_bytes = string.encode("UTF-8")
        indices = list(map(int, utf_bytes))
        return indices
    
    def decode(self, indices: list[int]) -> str:
        utf_bytes = bytes(indices)
        string = utf_bytes.decode("utf-8")
        return string