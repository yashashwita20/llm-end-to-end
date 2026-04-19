from .tokenizer import Tokenizer
import regex

class WordTokenizer(Tokenizer):
    """
    A simple tokenizer that builds a vocabulary from the given text, encodes and decodes text into indices
    """

    def __init__(self, text:str = None, pattern:str = None):
        """
        Initialize the tokenizer with initial text to build vocab

        The default regex used here splits the text based on 
        space and separates the punctuations and keeps 
        the alphanumeric chars together
        """
        self.vocab = set()
        self.stoi = {}
        self.itos = {}
        self.pattern= pattern or r"\s?\w+|[^\w\s]"

        if text:
            self.train(text)

    def train(self, text:str):
        """
        Build vocabulary from the text
        """

        tokens = regex.findall(self.pattern, text)
        self.vocab = set(tokens)
        self.vocab_size = len(self.vocab) + 2 #2 for <pad> and <unk>
        self.stoi = {word: i for i, word in enumerate(self.vocab, start = 2)}
        self.stoi['<pad>'] = 0
        self.stoi['<unk>'] = 1
        self.itos = {i:word for word, i in self.stoi.items()}

    def encode(self, text) -> list[int]:
        """
        Encode the text into a list of indices.
        """

        tokens = regex.findall(self.pattern, text)
        return [self.stoi.get(word, self.stoi['<unk>']) for word in tokens]
    
    def decode(self, indices: list[int]) -> str:
        """
        Decode the list of indices back into text.
        """
        
        return ''.join(self.itos.get(index, '<unk>') for index in indices)