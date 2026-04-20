import regex
import os
import json
from collections import Counter
from .tokenizer import Tokenizer

class BPEBasicTokenizer(Tokenizer):
    """
    BPE tokenizer class that builds a vocabulary from the given text and encodes/decodes text into indices.
    """

    def __init__(self,
             text: str | os.PathLike = None,
             lowercase: bool = False,
             remove_punctuation: bool = False,
             pattern: str = None,
             special_tokens: list[str] = None,
             num_merges: int = 100):
        """
        Args:
            text: Optional training corpus. If provided, `train()` is called immediately.
            lowercase: Whether to lowercase the text before tokenization.
            remove_punctuation: Whether to strip punctuation before tokenization.
            pattern: Regex pattern used to split text into pre-tokens. Defaults to
                splitting on words and individual punctuation characters, keeps the leading space.
            special_tokens: Tokens like "<|endoftext|>" that are handled verbatim and
                never split or merged. Assigned IDs starting at 256.
            num_merges: Number of BPE merge operations to learn during training.
        """

        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.pattern = pattern or r"\s?\w+|[^\w\s]|\s"
        self._compiled_pat = regex.compile(self.pattern)
        self.num_merges = num_merges
        self.bpe_merges = {}
        self._build_vocab(special_tokens)
        if self.special_tokens:
            # escape special chars in tokens like <|endoftext|>
            sp_pat = "|".join(regex.escape(t) for t in self.special_tokens)
            # regex.split with capture group keeps the delimiters
            self._split_pat = regex.compile(f"({sp_pat})")
        else:
            self._split_pat = None

        if text:
            self.train(text, self.num_merges)

    def _build_vocab(self, special_tokens:list[str] = None):
        """Initialize the base vocabulary and register special tokens.

        The base vocab contains one entry per byte value (0–255). Special tokens are
        stored as plain strings rather than byte sequences to avoid collisions: the
        byte encoding of a special token like ``<eot>`` is identical to regular text
        containing that character sequence, so keeping them separate is the only safe
        approach.
        """

        # base vocab: 256 byte tokens 
        self.vocab = {x: bytes([x]) for x in range(256)}

        # special tokens stored separately
        # storing special tokens as strings so that their byte forms dont collide with actual text
        # a special token like <eot> encoded as bytes is b'<eot>', which is the same byte sequence that would appear if the text itself contained <eot>
        self.special_tokens = {}
        if special_tokens:
            for i, token in enumerate(special_tokens):
                self.special_tokens[token] = 256 + i
        self.vocab_size = 256 + len(self.special_tokens)
        # reverse lookup for special tokens
        self.itos_special = {i: token for token, i in self.special_tokens.items()}

    def preprocess_text(self, text:str) -> str:
        """Preprocess text by lowering case and removing punctuation if configured."""
        if self.lowercase:
            text = text.lower()
        if self.remove_punctuation:
            text = regex.sub(r'[^\w\s]', '', text)
        return text
    
    def split_on_special_tokens(self, text: str) -> list[str]:
        """Split text around special tokens, preserving the tokens themselves.

        Returns a list of alternating plain-text chunks and special token strings.
        Empty strings are removed. Example::

            ["Hello", "<|endoftext|>", "World"]

        Special token strings can be identified downstream by checking membership
        in ``self.special_tokens``.
        """
        if not self.special_tokens:
            return [text]
        parts = self._split_pat.split(text)
        return [p for p in parts if p] # remove empty strings

    def pretokenize(self, text_parts:list[str]) -> list:
        """Split text on special tokens, then tokenize each chunk into byte lists."""
        compiled_pat = self._compiled_pat
        pretokenized = []
        for text in text_parts:
            if text in self.special_tokens:
                pretokenized.append(text)
            else:
                # tokens = regex.findall(self.pattern, text)
                tokens = compiled_pat.findall(text)
                for token in tokens:
                    pretokenized.append(tuple(token.encode("utf-8")))
        return pretokenized
    
    def get_byte_word_frequencies(self, pretokenized:list) -> dict[tuple[bytes, ...], int]:
        """Count how often each byte-tuple pre-token appears in the corpus.

        Special tokens are excluded because they are never merged and should not
        influence pair frequency statistics.
        """
        return Counter(token for token in pretokenized if token not in self.special_tokens)
    
    def get_pair_frequencies(self, byte_word_freq:dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, ...], int]:
        """Count adjacent byte-pair occurrences weighted by word frequency.

        Each pair within a word is counted once per occurrence of that word in the
        corpus, not once per character position. This means a word that appears 100
        times contributes 100 to the frequency of each of its adjacent pairs.
        """
        pair_freq = Counter()

        for byte_word_tuple, freq in byte_word_freq.items():
            for x, y in zip(byte_word_tuple, byte_word_tuple[1:]):
                pair_freq[(x,y)] += freq

        return pair_freq

    def merge_pair(self, byte_word_freq:dict[tuple[bytes, ...], int], pair_to_merge:tuple) -> dict[tuple[bytes, ...], int]:
        """Replace every occurrence of ``pair_to_merge`` in the corpus with a new token ID.

        The new token ID is ``self.vocab_size`` at the time of the call (the caller is
        responsible for incrementing it afterward). Returns a new frequency dict with
        the merged tokens; the original is not modified.
        """
        new_byte_word_freq = {}

        for byte_word_tuple, freq in byte_word_freq.items():
            new_word = []
            i = 0

            while i < len(byte_word_tuple):
                if i < len(byte_word_tuple) - 1 and (byte_word_tuple[i], byte_word_tuple[i + 1]) == pair_to_merge:
                    new_word.append(self.vocab_size)
                    i += 2
                else:
                    new_word.append(byte_word_tuple[i])
                    i += 1

            new_byte_word_freq[tuple(new_word)] = freq

        return new_byte_word_freq
    
    def train(self, text:str, num_merges:int):
        """Learn BPE merge rules from a training corpus.

        Pipeline:
            1. Preprocess (lowercase / strip punctuation).
            2. Split on special tokens so they are never merged.
            3. Pre-tokenize each chunk into tuples of UTF-8 bytes.
            4. Iteratively find the most frequent adjacent byte pair, merge it
               into a new token, and record the rule in ``self.bpe_merges``.

        Stops early if no pairs remain (fully merged corpus).
        """
        text = self.preprocess_text(text)

        parts = self.split_on_special_tokens(text)

        pretokenized = self.pretokenize(parts)

        byte_word_freq = self.get_byte_word_frequencies(pretokenized)

        for _ in range(num_merges):
            byte_pair_freq = self.get_pair_frequencies(byte_word_freq)

            if not byte_pair_freq:
                break

            # doesn't break ties
            # most_freq_pair = max(byte_pair_freq, key = byte_pair_freq.get)

            # Breaking ties to prefer the lexicographically greater pair
            # frequency tie: (104, 101) and (108, 111) both appear 5 times
            # (5, (104, 101)) vs (5, (108, 111))
            # same freq → compare pairs → (108, 111) wins because 108 > 104
            most_freq_pair = max(byte_pair_freq, key = lambda pair: (byte_pair_freq[pair], pair))

            byte_word_freq = self.merge_pair(byte_word_freq, most_freq_pair)
            self.vocab[self.vocab_size] = self.vocab[most_freq_pair[0]] + self.vocab[most_freq_pair[1]]
            self.bpe_merges[most_freq_pair] = self.vocab_size
            self.vocab_size += 1

        return self.vocab, list(self.bpe_merges.keys())
            

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs.

        Applies learned BPE merges in the order they were learned (insertion order of
        ``self.bpe_merges``). Applying them out of order would produce incorrect
        tokenizations because later merges depend on earlier ones having been applied.
        Special tokens bypass the BPE process and are mapped directly to their IDs.
        """
        text = self.preprocess_text(text)
        parts = self.split_on_special_tokens(text)
        pretokenized = self.pretokenize(parts)

        encoded = []

        for item in pretokenized:
            if item in self.special_tokens:
                encoded.append(self.special_tokens[item])
                continue

            byte_list = list(item)

            for merge_pair, token_id in self.bpe_merges.items():
                new_list = []
                i = 0
                while i < len(byte_list):
                    if i < len(byte_list) - 1 and (byte_list[i], byte_list[i + 1]) == merge_pair:
                        new_list.append(token_id)
                        i += 2
                    else:
                        new_list.append(byte_list[i])
                        i += 1
                byte_list = new_list

                if len(byte_list) == 1:
                    break

            encoded.extend(byte_list)

        return encoded
    
    def decode(self, indices: list[int]) -> str:
        """Decode a list of token IDs back into a string.

        Special token IDs are converted back to their string form. All other IDs are
        looked up in ``self.vocab`` and their byte sequences are concatenated before
        UTF-8 decoding. Replacement character (U+FFFD) is used for any undecodable
        byte sequences.
        """
        byte_list = []
        for idx in indices:
            if idx in self.itos_special:
                byte_list.extend(self.itos_special[idx].encode("utf-8"))
            elif idx in self.vocab:
                byte_list.extend(self.vocab[idx])
            else:
                raise ValueError(f"Unknown token id: {idx}")
            
        return bytes(byte_list).decode("utf-8", errors="replace")
    
    @classmethod
    def from_file(cls, filepath, **kwargs):
        """Construct a ``BPETokenizer`` trained on the full contents of a text file.

        All keyword arguments are forwarded to ``__init__`` (e.g. ``num_merges``,
        ``special_tokens``).
        """
        with open(filepath, "r") as f:
            text = f.read()
        return cls(text=text, **kwargs)

    def save(self, filepath: str):
        data = {
            "pattern": self.pattern,
            "special_tokens": self.special_tokens,
            "bpe_merges": {f"{k[0]},{k[1]}": v for k, v in self.bpe_merges.items()},
            "vocab_size": self.vocab_size,
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    @classmethod
    def from_pretrained(cls, filepath: str):
        with open(filepath, "r") as f:
            data = json.load(f)
        tokenizer = cls(pattern=data["pattern"], special_tokens=list(data["special_tokens"].keys()))
        tokenizer.bpe_merges = {tuple(map(int, k.split(","))): v for k, v in data["bpe_merges"].items()}
        tokenizer.vocab_size = data["vocab_size"]
        # rebuild vocab from merges
        for pair, new_id in tokenizer.bpe_merges.items():
            tokenizer.vocab[new_id] = tokenizer.vocab[pair[0]] + tokenizer.vocab[pair[1]]
        return tokenizer