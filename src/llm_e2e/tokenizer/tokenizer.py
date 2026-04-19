from abc import ABC, abstractmethod

class Tokenizer(ABC):
    """
    Abstract Interface for a tokenizer
    """

    @abstractmethod
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError
    
    @abstractmethod
    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError