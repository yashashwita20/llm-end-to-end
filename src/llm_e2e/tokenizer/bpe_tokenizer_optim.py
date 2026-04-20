import heapq
import regex
import os
import json
from multiprocessing import Pool, heap
from functools import partial
from collections import Counter
from .tokenizer import Tokenizer

class BPEOptimTokenizer(Tokenizer):
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
        self.heap = []
        self.bpe_merges = {}
        self._encode_cache = {}
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
    
    @staticmethod
    def _pretokenize_chunk(chunk, pattern, special_tokens):
        """Standalone function for multiprocessing."""
        compiled_pat = regex.compile(pattern)
        result = []
        for part in chunk:
            if part in special_tokens:
                result.append(part)
            else:
                #tokens = regex.findall(pattern, part)
                tokens = compiled_pat.findall(part)
                for token in tokens:
                    result.append(tuple(token.encode("utf-8")))
        return result
    
    def pretokenize_optim(self, text_parts: list[str], num_workers: int = None) -> list:
        if num_workers is None:
            num_workers = os.cpu_count()

        # # split text_parts into chunks for each worker
        # chunks = [[] for _ in range(num_workers)]
        # for i, part in enumerate(text_parts):
        #     chunks[i % num_workers].append(part)

        # Balancing chunks by character length to avoid stragglers
        chunk_sizes = [0] * num_workers
        chunks = [[] for _ in range(num_workers)]
        for part in text_parts:
            idx = chunk_sizes.index(min(chunk_sizes))
            chunks[idx].append(part)
            chunk_sizes[idx] += len(part)

        # remove empty chunks
        chunks = [c for c in chunks if c]

        if len(chunks) <= 1:
            return self.pretokenize(text_parts)

        func = partial(
            BPEOptimTokenizer._pretokenize_chunk, #_pretokenize_chunk is a method on the class, so you need to reference it through the class name when using it with partial
            pattern = self.pattern,
            special_tokens = self.special_tokens
        )

        with Pool(num_workers) as pool:
            results = pool.map(func, chunks)

        # flatten results
        return [token for result in results for token in result]
    
    def get_byte_word_frequencies(self, pretokenized:list) -> dict[tuple[bytes, ...], int]:
        """Count how often each byte-tuple pre-token appears in the corpus.

        Special tokens are excluded because they are never merged and should not
        influence pair frequency statistics.
        """
        return Counter(token for token in pretokenized if token not in self.special_tokens)
    
    def build_pair_index(self, byte_word_freq): # replaced get_byte_pair_frequencies with this
        byte_pair_freq = Counter()
        byte_pair_to_words = {}
        for word, freq in byte_word_freq.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                byte_pair_freq[pair] += freq
                byte_pair_to_words.setdefault(pair, set()).add(word)
        
        heap = [(-freq, (-p[0], -p[1])) for p, freq in byte_pair_freq.items()]
        heapq.heapify(heap)

        return byte_pair_freq, byte_pair_to_words, heap

    
    def merge_pair_incremental(self, byte_word_freq, byte_pair_freq, byte_pair_to_words, pair_to_merge, heap): # replaced merge_pair with this

        for byte_word_tuple in list(byte_pair_to_words.get(pair_to_merge, set())):
            freq = byte_word_freq[byte_word_tuple]

            # build new word
            new_byte_word = []
            i = 0
            while i < len(byte_word_tuple):
                if i < len(byte_word_tuple) - 1 and (byte_word_tuple[i], byte_word_tuple[i + 1]) == pair_to_merge:
                    new_byte_word.append(self.vocab_size)
                    i += 2
                else:
                    new_byte_word.append(byte_word_tuple[i])
                    i += 1
            new_byte_word_tuple = tuple(new_byte_word)

            # update byte_word_freq
            del byte_word_freq[byte_word_tuple]
            byte_word_freq[new_byte_word_tuple] = byte_word_freq.get(new_byte_word_tuple, 0) + freq

            # update byte_pair_to_words: remove old, add new
            for i in range(len(byte_word_tuple) - 1):
                byte_pair_to_words.setdefault((byte_word_tuple[i], byte_word_tuple[i+1]), set()).discard(byte_word_tuple)
            for i in range(len(new_byte_word) - 1):
                byte_pair_to_words.setdefault((new_byte_word[i], new_byte_word[i+1]), set()).add(new_byte_word_tuple)

            # update byte_pair_freq: diff old pairs vs new pairs for this word
            old_pairs = Counter(zip(byte_word_tuple, byte_word_tuple[1:]))
            new_pairs = Counter(zip(new_byte_word_tuple, new_byte_word_tuple[1:]))
            for p in set(old_pairs) | set(new_pairs):
                delta = (new_pairs.get(p, 0) - old_pairs.get(p, 0)) * freq
                if delta:
                    byte_pair_freq[p] += delta
                    if delta > 0:
                        heapq.heappush(heap, (-byte_pair_freq[p], (-p[0], -p[1])))
                    if byte_pair_freq[p] <= 0:
                        del byte_pair_freq[p]

        byte_pair_freq.pop(pair_to_merge, None)
        byte_pair_to_words.pop(pair_to_merge, None)

        return byte_word_freq, byte_pair_freq, byte_pair_to_words
    
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

        pretokenized = self.pretokenize_optim(parts)

        byte_word_freq = self.get_byte_word_frequencies(pretokenized)

        byte_pair_freq, byte_pair_to_words, heap = self.build_pair_index(byte_word_freq)

        for _ in range(num_merges):

            if not byte_pair_freq:
                break

            #most_freq_pair = max(byte_pair_freq, key = lambda pair: (byte_byte_pair_freq[pair], pair))
            while True:
                neg_freq, neg_pair = heapq.heappop(heap)
                most_freq_pair = (-neg_pair[0], -neg_pair[1])
                if byte_pair_freq.get(most_freq_pair, 0) == -neg_freq:
                    break
            byte_word_freq, byte_pair_freq, byte_pair_to_words = self.merge_pair_incremental(byte_word_freq, byte_pair_freq, byte_pair_to_words, most_freq_pair, heap)
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

            if item in self._encode_cache:
                encoded.extend(self._encode_cache[item])
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

            self._encode_cache[item] = byte_list

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