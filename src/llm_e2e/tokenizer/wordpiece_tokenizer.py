from collections import Counter
import json
import regex
from .tokenizer import Tokenizer

class WordPieceTokenizer(Tokenizer):
    def __init__(self,
                 text: str = None,
                 lowercase: bool = False,
                 vocab_size: int = 30000,
                 unk_token: str = "[UNK]",
                 pad_token: str = "[PAD]",
                 cls_token: str = "[CLS]",
                 sep_token: str = "[SEP]",
                 mask_token: str = "[MASK]",
                 pattern: str = r"\w+|[^\w\s]",
                 special_tokens: list[str] = None,):
        
        self.lowercase = lowercase
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token
        defaults = [unk_token, pad_token, cls_token, sep_token, mask_token]
        extras = [token for token in special_tokens if token not in defaults] if special_tokens is not None else [] #deduplicate special tokens
        self.special_tokens = defaults + extras
        self.token_to_id = {}
        self.id_to_token = {}
        self._encode_cache = {}
        self._compiled_pat = regex.compile(pattern)

        for i, token in enumerate(self.special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        
        if text is not None:
            self.train(text)

    def preprocess_text(self, text:str) -> str:
        """Preprocess text by lowering case and removing punctuation if configured."""
        if self.lowercase:
            text = text.lower()
        return text
    
    def pretokenize(self, text: str) -> Counter:
        #words = text.split() #simple whitespace tokenization
        #words = regex.findall(r"\w+|[^\w\s]", text) #regex-based tokenization to capture words and punctuation as separate tokens
        words = self._compiled_pat.findall(text)

        return Counter(words)
    
    def _build_vocab(self, word_freq: Counter) -> dict[str, list[str]]:

        start_id = len(self.special_tokens)

        word_tokens = {}
        for word in word_freq:
            tokens = []
            for i, char in enumerate(word):
                token = char if i == 0 else "##" + char
                tokens.append(token)
                if token not in self.token_to_id:
                    self.token_to_id[token] = start_id
                    self.id_to_token[start_id] = token
                    start_id += 1
            word_tokens[word] = tokens

        return word_tokens
    
    def _get_token_frequency(self, word_freq: Counter, word_tokens: dict[str, list[str]]) -> Counter:
        token_freq = Counter()
        for word, tokens in word_tokens.items():
            freq = word_freq[word]
            for token in tokens:
                token_freq[token] += freq
        return token_freq
    
    def _get_pair_freq(self, word_tokens: dict[str, list[str]], word_freq: Counter, token_freq: Counter) -> Counter:
        pair_freq = Counter()
        for word, tokens in word_tokens.items():
            freq = word_freq[word]
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_freq[pair] += freq
        
        return pair_freq
    
    def _get_pair_score(self, pair_freq: Counter, token_freq: Counter) -> dict:
        pair_scores = {}
        for pair in pair_freq:
            pair_scores[pair] = pair_freq[pair] / (token_freq[pair[0]] * token_freq[pair[1]])
        return pair_scores
    
    def _merge_pair(self, pair: tuple[str, str], word_tokens: dict[str, list[str]], word_freq: Counter, token_freq: Counter, pair_freq: Counter, pair_scores: dict) -> tuple[dict, Counter, Counter, dict]:
        new_token = pair[0] + pair[1].lstrip('##')
        new_id = len(self.token_to_id)
        self.token_to_id[new_token] = new_id
        self.id_to_token[new_id] = new_token
        new_pairs = set()
        del pair_freq[pair]
        del pair_scores[pair]

        for word, tokens in word_tokens.items():
            i = 0
            n_merges = 0
            new_tokens = []
            merge = False
            freq = word_freq[word]
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:

                    if i != 0:
                            old_pair = (tokens[i-1], tokens[i])
                            new_pair = (tokens[i-1], new_token)
                            pair_freq[old_pair] -= freq
                            if pair_freq[old_pair] <= 0:
                                del pair_freq[old_pair]
                                del pair_scores[old_pair]
                            else:
                                pair_scores[old_pair] = pair_freq[old_pair] / (token_freq[old_pair[0]] * token_freq[old_pair[1]])
                            pair_freq[new_pair] += freq
                            new_pairs.add(new_pair)
                    
                    if i < len(tokens) - 2:
                            
                            old_pair = (tokens[i+1], tokens[i+2])
                            new_pair = (new_token, tokens[i+2])
                            pair_freq[old_pair] -= freq
                            if pair_freq[old_pair] <= 0:
                                del pair_freq[old_pair]
                                del pair_scores[old_pair]
                            else:
                                pair_scores[old_pair] = pair_freq[old_pair] / (token_freq[old_pair[0]] * token_freq[old_pair[1]])
                            pair_freq[new_pair] += freq
                            new_pairs.add(new_pair)

                    new_tokens.append(new_token)
                    i += 2
                    merge = True
                    n_merges += 1

                else:
                    new_tokens.append(tokens[i])
                    i += 1
            if merge:
                word_tokens[word] = new_tokens
                token_freq[pair[0]] -= n_merges * freq
                token_freq[pair[1]] -= n_merges * freq
                token_freq[new_token] += n_merges * freq

        for p in new_pairs:
            pair_scores[p] = pair_freq[p] / (token_freq[p[0]] * token_freq[p[1]])

        return word_tokens, token_freq, pair_freq, pair_scores
    
    def train(self, text:str) -> None:
        text = self.preprocess_text(text)

        word_freq = self.pretokenize(text)
        word_tokens = self._build_vocab(word_freq)
        token_freq = self._get_token_frequency(word_freq, word_tokens)
        pair_freq = self._get_pair_freq(word_tokens, word_freq, token_freq)
        pair_scores = self._get_pair_score(pair_freq, token_freq)

        while pair_freq:

            best_pair = max(pair_scores, key=pair_scores.get)
            word_tokens, token_freq, pair_freq, pair_scores = self._merge_pair(best_pair, word_tokens, word_freq, token_freq, pair_freq, pair_scores)

            if len(self.token_to_id) >= self.vocab_size:
                break

    def _maxmatch(self, word:str) -> list[str]:

        if word in self._encode_cache:
            return self._encode_cache[word]
        
        start = 0
        tokens = []

        while start < len(word):
            end = len(word)
            found = False
            while end > start:
                substr = word[start:end]
                if start > 0:
                    substr = "##" + substr

                if substr in self.token_to_id:
                    tokens.append(substr)
                    start = end
                    found = True
                    break
                end -= 1

            if not found:
                self._encode_cache[word] = [self.unk_token]
                return [self.unk_token]
            
        self._encode_cache[word] = tokens
            
        return tokens
    
    def encode(self, text : str) -> list[int]:

        text = self.preprocess_text(text)

        words = self._compiled_pat.findall(text)

        result = []

        for word in words:
            tokens = self._maxmatch(word)
            token_ids = [self.token_to_id[token] for token in tokens]
            result.extend(token_ids)

        return result
    
    def decode(self, ids: list[int]) -> str:

        result = []
        for id in ids:
            if id not in self.id_to_token:
                raise ValueError(f"Unknown token id: {id}")
            token = self.id_to_token[id]
            if token.startswith('##'):
                result [-1] += token[2:] # continue last token
            else:
                result.append(token)
        
        return " ".join(result)
    
    def save(self, filepath: str):
        data = {
            "token_to_id": self.token_to_id,
            #"id_to_token": self.id_to_token, json doesnt support integer keys
            "special_tokens": self.special_tokens,
            "lowercase": self.lowercase
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    @classmethod
    def from_pretrained(cls, filepath: str, **kwargs):
        with open(filepath, "r") as f:
            data = json.load(f)

        tokenizer = cls(**kwargs)
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.id_to_token = {int(id): token for token, id in data["token_to_id"].items()}
        tokenizer.special_tokens = data["special_tokens"]
        tokenizer.lowercase = data["lowercase"]
        tokenizer.vocab_size = len(tokenizer.token_to_id)

        return tokenizer
    
    @classmethod
    def from_file(cls, filepath: str, **kwargs):
        with open(filepath, "r") as f:
            text = f.read()
        return cls(text=text, **kwargs)
        
