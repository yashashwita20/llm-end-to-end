from collections import Counter
import json
import regex
import math
from .tokenizer import Tokenizer
import unicodedata

SPIECE_UNDERLINE = "\u2581"

class UnigramTokenizer(Tokenizer):
    def __init__(self,
                 text:str = None,
                 lowercase:bool = False,
                 vocab_size:int = 8000,
                 shrink_factor: float = 0.75, #fraction of vocab to keep each round
                 pattern: str = r"\w+|[^\w\s]",
                 special_tokens: list[str] = None,
                 max_substring_len:int = 16,
                 convergence_threshold:float = 0.001,
                 normalize: bool = True,
                 unk_token:str = '<unk>',
                 verbose = True):
        self.lowercase = lowercase
        self.vocab_size = vocab_size
        self.shrink_factor = shrink_factor
        self.max_substring_len = max_substring_len
        self.convergence_threshold = convergence_threshold
        self.normalize = normalize
        self.token_to_id = {}
        self.id_to_token = {}
        self.token_probs = {}
        self.verbose = verbose
        self.unk_token = unk_token
        defaults = [unk_token,'<s>','</s>']
        extras = [token for token in special_tokens if token not in defaults] if special_tokens is not None else [] #deduplicate special tokens
        self.special_tokens = defaults + extras
        self.byte_tokens = {f"<0x{i:02X}>": i for i in range(256)}
        self._compiled_pat = regex.compile(pattern)

        for i, token in enumerate(self.special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        start_id = len(self.special_tokens)
        for token in self.byte_tokens:
            self.token_to_id[token] = start_id
            self.id_to_token[start_id] = token
            start_id += 1
        
        if text is not None:
            self.train(text)

    def _normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = " ".join(text.split()) # collapse whitespace
        return text.strip()

    def preprocess_text(self, text:str) -> str:
        """Preprocess text by lowering case and removing punctuation if configured."""
        if self.lowercase:
            text = text.lower()
        if self.normalize:
            text = self._normalize(text)
        return text
    
    def pretokenize(self, text: str) -> Counter:
        #words = text.split() #simple whitespace tokenization
        #words = regex.findall(r"\w+|[^\w\s]", text) #regex-based tokenization to capture words and punctuation as separate tokens
        # words = self._compiled_pat.findall(text)

        # return Counter(SPIECE_UNDERLINE + word for word in words)
        result = []
        for match in self._compiled_pat.finditer(text):
            word = match.group()
            prefix = SPIECE_UNDERLINE if (match.start() == 0 or text[match.start() - 1].isspace()) else ""
            result.append(prefix + word)
        return Counter(result)


    def _init_token_probs(self, word_freq: Counter):

        token_freq = Counter()
        single_chars = set()
        for word, freq in word_freq.items():

            for i in range(len(word)):
                for j in range(i + 1, min( i + self.max_substring_len + 1, len(word) + 1)):
                    token_freq[word[i:j]] += freq

                    #guaranteed set — all single characters (ensures full coverage)
                    if j - i == 1: single_chars.add(word[i:j])

        #top N substrings as candidates
        initial_vocab_size = self.vocab_size * 10 #initial vocab needs to be much larger than target so there's room to prune                
        top_tokens = {t for t, _ in token_freq.most_common(initial_vocab_size)}
        
        #union — single chars always wins regardless of rank
        all_tokens = single_chars | top_tokens

        total_counts = sum(token_freq[token] for token in all_tokens)
        for token in all_tokens:
            # prob of a word  = product of probabilities of each substring of the word
            # taking log of probabilities helps us sum them instead of multiplying them
            # it also helps avoid floating point underflow on long words
            self.token_probs[token] = math.log(token_freq[token] / total_counts) 
            
    def _viterbi(self, word:str) -> list[str]:

        best_score = {0 : 0.0} # position:best score at position
        best_piece = {} # position : (token_len, token)

        # Viterbi Forward Pass
        for i in range(1, len(word) + 1):
            for j in range(max(0, i - self.max_substring_len), i):
                token = word[j:i]
                if token in self.token_probs:
                    score = best_score.get(j, -math.inf) + self.token_probs[token]
                    if score > best_score.get(i, -math.inf):
                        best_score[i] = score
                        best_piece[i] = (j,token)

        if len(word) not in best_score:
            #return [self.unk_token] # if word not found in token, return unk_token

            # if token not found, fallback to byte tokens
            fallback = []
            
            for char in word:
                for byte in char.encode("utf-8"):
                    fallback.append(f"<0x{byte:02X}>")
            return fallback

        # Viterbi Backtrack        
        tokens = []
        i = len(word)
        while i > 0:
            j, token = best_piece[i]
            tokens.append(token)
            i = j
        tokens.reverse()

        return tokens     

    def _compute_loss(self, word_freq: Counter) -> float:
        loss = 0.0

        for word, freq in word_freq.items():
            tokens = self._viterbi(word)
            token_score = sum(self.token_probs.get(token, math.log(1e-10)) for token in tokens)
            loss += freq * token_score

        return -loss #negative as log probs are negative, we are minimizing loss so we want positive loss

    def _compute_token_counts(self, word_freq: Counter) -> Counter:
        # E Step
        token_count = Counter()
        for word, freq, in word_freq.items():
            tokens = self._viterbi(word)
            for token in tokens:
                token_count[token] += freq

        # M Step
        # total = sum(token_count.values()) # Byte tokens from fallback words accumulate in token_count, inflating the denominator and making real token probabilities slightly smaller than they should be
        total = sum(count for token, count in token_count.items() if token not in self.byte_tokens)
        for token in self.token_probs:
            if token in token_count:
                self.token_probs[token] = math.log(token_count[token] / total)
            else:
                self.token_probs[token] = math.log(1e-10) # near-zero for unused token

        return token_count

    def _prune_vocab(self, token_count: Counter) -> None:    

        removable = [token for token in self.token_probs if len(token) > 1 and token not in self.special_tokens and token not in self.byte_tokens]
        removable.sort(key = lambda token: token_count[token] * self.token_probs[token], reverse = True)
        
        keep_n = max(int(len(self.token_probs) * self.shrink_factor), self.vocab_size)
        n_remove = len(self.token_probs) - keep_n
        tokens_to_remove = removable[:n_remove]

        for token in tokens_to_remove:
            del self.token_probs[token]

        # Re-normalizing probs as they no more sum to 1
        total = math.log(sum(math.exp(lp) for lp in self.token_probs.values()))
        self.token_probs = {t: lp - total for t, lp in self.token_probs.items()}    

    def train(self, text:str) -> None:
        text = self.preprocess_text(text)

        word_freq = self.pretokenize(text)

        self._init_token_probs(word_freq)

        prev_loss = None

        while len(self.token_probs) > self.vocab_size:
            token_count = self._compute_token_counts(word_freq) # E+M Step
            new_loss = self._compute_loss(word_freq)
            self._prune_vocab(token_count)

            if self.verbose:
                print(f"vocab: {len(self.token_probs)}, loss: {new_loss:.4f}")

            if prev_loss is not None and abs(prev_loss - new_loss) < self.convergence_threshold:
                break

            prev_loss = new_loss

        start_id = len(self.token_to_id)
        for token in self.token_probs:
            if token not in self.token_to_id:
                self.token_to_id[token] = start_id
                self.id_to_token[start_id] = token
                start_id += 1

    def encode(self, text:str) -> list[int]:
        text = self.preprocess_text(text)

        # words = self._compiled_pat.findall(text)

        result = []

        # for word in words:
        #     tokens = self._viterbi(SPIECE_UNDERLINE + word)
        for match in self._compiled_pat.finditer(text):
            word = match.group()
            prefix = SPIECE_UNDERLINE if (match.start() == 0 or text[match.start() - 1].isspace()) else ""
            tokens = self._viterbi(prefix + word)

            token_ids = [self.token_to_id[token] for token in tokens]
            result.extend(token_ids)

        return result
    
    def decode(self, ids: list[int]) -> str:
        byte_buffer = []
        result = []

        def flush_bytes():
            if not byte_buffer:
                return
            decoded = bytes(byte_buffer).decode("utf-8", errors="replace")
            byte_buffer.clear()
            parts = decoded.split(SPIECE_UNDERLINE)
            for i, part in enumerate(parts):
                if not part:
                    continue
                if i == 0:
                    if result:
                        result[-1] += part
                    else:
                        result.append(part)
                else:
                    result.append(part)

        for id in ids:
            if id not in self.id_to_token:
                raise ValueError(f"Unknown token id: {id}")
            token = self.id_to_token[id]

            if token.startswith("<0x") and token.endswith(">"):
                byte_buffer.append(int(token[3:5], 16))
            else:
                flush_bytes()
                if token.startswith(SPIECE_UNDERLINE):
                    result.append(token[1:]) # strip SPIECE_UNDERLINE, start new word
                else:
                    if result:
                        result[-1] += token # continuation, add to last token
                    else:
                        result.append(token) # fallback for if very first token doesnt start with SPIECE_UNDERLINE, result would be empty and result[-1] will throw an index error

        flush_bytes()
        return " ".join(result)

    def save(self, filepath: str) -> None:
        data = {
            "token_to_id": self.token_to_id,
            "token_probs": self.token_probs,
            "special_tokens": self.special_tokens,
            "lowercase": self.lowercase,
            "vocab_size": self.vocab_size
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    @classmethod
    def from_pretrained(cls, filepath: str, **kwargs):
        with open(filepath, "r") as f:
            data = json.load(f)

        tokenizer = cls(**kwargs)
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.id_to_token = {int(k): v for k, v in data["token_to_id"].items()}  # invert
        tokenizer.token_probs = data["token_probs"]
        tokenizer.special_tokens = data["special_tokens"]
        tokenizer.lowercase = data["lowercase"]
        tokenizer.vocab_size = data["vocab_size"]
        return tokenizer

    @classmethod
    def from_file(cls, filepath: str, **kwargs) -> "UnigramTokenizer":
        with open(filepath, "r") as f:
            text = f.read()
        return cls(text=text, **kwargs)
