import json
import os
import tempfile

from llm_e2e.tokenizer import (
    ByteTokenizer,
    CharacterTokenizer,
    WordTokenizer,
    BPETokenizer,
)


# ---------------------------------------------------------------------------
# ByteTokenizer
# ---------------------------------------------------------------------------

class TestByteTokenizer:
    def setup_method(self):
        self.tok = ByteTokenizer()

    def test_encode_ascii(self):
        assert self.tok.encode("hi") == [104, 105]

    def test_decode_ascii(self):
        assert self.tok.decode([104, 105]) == "hi"

    def test_roundtrip_ascii(self):
        text = "Hello, World!"
        assert self.tok.decode(self.tok.encode(text)) == text

    def test_roundtrip_unicode(self):
        text = "café"
        assert self.tok.decode(self.tok.encode(text)) == text

    def test_encode_empty(self):
        assert self.tok.encode("") == []

    def test_decode_empty(self):
        assert self.tok.decode([]) == ""

    def test_encode_values_in_byte_range(self):
        ids = self.tok.encode("ABC")
        assert all(0 <= i <= 255 for i in ids)

    def test_multibyte_unicode_encodes_to_multiple_ids(self):
        # "é" is 2 bytes in UTF-8
        ids = self.tok.encode("é")
        assert len(ids) == 2


# ---------------------------------------------------------------------------
# CharacterTokenizer
# ---------------------------------------------------------------------------

class TestCharacterTokenizer:
    def setup_method(self):
        self.tok = CharacterTokenizer()

    def test_encode_ascii(self):
        assert self.tok.encode("AB") == [65, 66]

    def test_decode_ascii(self):
        assert self.tok.decode([65, 66]) == "AB"

    def test_roundtrip_ascii(self):
        text = "Hello, World!"
        assert self.tok.decode(self.tok.encode(text)) == text

    def test_roundtrip_unicode(self):
        text = "日本語"
        assert self.tok.decode(self.tok.encode(text)) == text

    def test_encode_empty(self):
        assert self.tok.encode("") == []

    def test_decode_empty(self):
        assert self.tok.decode([]) == ""

    def test_one_id_per_character(self):
        text = "hello"
        assert len(self.tok.encode(text)) == len(text)


# ---------------------------------------------------------------------------
# WordTokenizer
# ---------------------------------------------------------------------------

class TestWordTokenizer:
    CORPUS = "the cat sat on the mat . the cat is fat ."

    def setup_method(self):
        self.tok = WordTokenizer(text=self.CORPUS)

    def test_vocab_contains_words(self):
        for word in ["cat", "sat", "mat", "the"]:
            assert word in self.tok.stoi

    def test_special_tokens_present(self):
        assert self.tok.stoi["<pad>"] == 0
        assert self.tok.stoi["<unk>"] == 1

    def test_encode_returns_ints(self):
        ids = self.tok.encode("the cat")
        assert all(isinstance(i, int) for i in ids)

    def test_encode_known_word_not_unk(self):
        ids = self.tok.encode("cat")
        assert ids[0] != self.tok.stoi["<unk>"]

    def test_encode_unknown_word_returns_unk(self):
        ids = self.tok.encode("zebra")
        assert ids[0] == self.tok.stoi["<unk>"]

    def test_decode_returns_string(self):
        ids = self.tok.encode("the cat")
        assert isinstance(self.tok.decode(ids), str)

    def test_roundtrip_known_text(self):
        text = "the cat sat"
        decoded = self.tok.decode(self.tok.encode(text))
        # decode joins with spaces; split to compare token sets
        assert set(decoded.split()) == set(text.split())

    def test_train_updates_vocab(self):
        tok = WordTokenizer()
        assert len(tok.vocab) == 0
        tok.train("hello world")
        assert len(tok.vocab) > 0

    def test_vocab_size_accounts_for_special_tokens(self):
        assert self.tok.vocab_size == len(self.tok.vocab) + 2

    def test_custom_pattern(self):
        tok = WordTokenizer(text="hello world", pattern=r"\w+")
        assert "hello" in tok.stoi
        assert "world" in tok.stoi


# ---------------------------------------------------------------------------
# BPETokenizer
# ---------------------------------------------------------------------------

TRAIN_TEXT = (
    "low low low low low "
    "lower lower "
    "newest newest newest newest newest newest "
    "widest widest widest "
)


class TestBPETokenizerBase:
    """Tests that don't require training (base 256-byte vocab)."""

    def setup_method(self):
        self.tok = BPETokenizer()

    def test_base_vocab_size(self):
        assert self.tok.vocab_size == 256

    def test_encode_ascii_no_merges(self):
        ids = self.tok.encode("hi")
        assert ids == [ord("h"), ord("i")]

    def test_decode_ascii_no_merges(self):
        assert self.tok.decode([ord("h"), ord("i")]) == "hi"

    def test_roundtrip_no_merges(self):
        text = "hello"
        assert self.tok.decode(self.tok.encode(text)) == text

    def test_encode_empty(self):
        assert self.tok.encode("") == []

    def test_decode_empty(self):
        assert self.tok.decode([]) == ""


class TestBPETokenizerTrained:
    """Tests that require a trained tokenizer."""

    def setup_method(self):
        self.tok = BPETokenizer(text=TRAIN_TEXT, num_merges=10)

    def test_vocab_grows_after_training(self):
        assert self.tok.vocab_size > 256

    def test_merges_recorded(self):
        assert len(self.tok.bpe_merges) > 0

    def test_encode_returns_list_of_ints(self):
        ids = self.tok.encode("low")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_trained_text_fewer_tokens(self):
        # After merges, "low" should encode to fewer IDs than its raw bytes
        raw_len = len("low".encode("utf-8"))
        ids = self.tok.encode("low")
        assert len(ids) <= raw_len

    def test_roundtrip(self):
        text = "low lower newest"
        assert self.tok.decode(self.tok.encode(text)) == text

    def test_roundtrip_unseen_text(self):
        text = "hello world"
        assert self.tok.decode(self.tok.encode(text)) == text


class TestBPETokenizerSpecialTokens:
    def setup_method(self):
        self.tok = BPETokenizer(
            text=TRAIN_TEXT,
            num_merges=5,
            special_tokens=["<|endoftext|>", "<|pad|>"],
        )

    def test_special_tokens_assigned_ids_above_256(self):
        for _, idx in self.tok.special_tokens.items():
            assert idx >= 256

    def test_encode_special_token(self):
        ids = self.tok.encode("<|endoftext|>")
        assert ids == [self.tok.special_tokens["<|endoftext|>"]]

    def test_special_token_roundtrip(self):
        # The \s? prefix in BPE's pretokenize pattern absorbs the space
        # immediately before a special token, so that space is lost on decode.
        text = "hello <|endoftext|> world"
        assert self.tok.decode(self.tok.encode(text)) == "hello<|endoftext|> world"

    def test_special_token_not_split_by_bpe(self):
        ids = self.tok.encode("<|endoftext|>")
        assert len(ids) == 1


class TestBPETokenizerPreprocessing:
    def test_lowercase(self):
        tok = BPETokenizer(text="Hello World", num_merges=0, lowercase=True)
        ids_lower = tok.encode("Hello")
        ids_direct = tok.encode("hello")
        assert ids_lower == ids_direct

    def test_remove_punctuation(self):
        tok = BPETokenizer(
            text="hello world", num_merges=0, remove_punctuation=True
        )
        ids_with = tok.encode("hello!")
        ids_without = tok.encode("hello")
        assert ids_with == ids_without


class TestBPETokenizerSaveLoad:
    def test_save_and_from_pretrained_roundtrip(self):
        tok = BPETokenizer(
            text=TRAIN_TEXT,
            num_merges=10,
            special_tokens=["<|endoftext|>"],
        )
        text = "low lower newest <|endoftext|>"

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            filepath = f.name

        try:
            tok.save(filepath)
            loaded = BPETokenizer.from_pretrained(filepath)

            assert tok.encode(text) == loaded.encode(text)
            # space before special token is absorbed by BPE's \s? pattern
            assert loaded.decode(loaded.encode(text)) == "low lower newest<|endoftext|>"
        finally:
            os.unlink(filepath)

    def test_save_produces_valid_json(self):
        tok = BPETokenizer(text=TRAIN_TEXT, num_merges=5)
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            filepath = f.name
        try:
            tok.save(filepath)
            with open(filepath) as f:
                data = json.load(f)
            assert "bpe_merges" in data
            assert "vocab_size" in data
        finally:
            os.unlink(filepath)

    def test_from_file(self):
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False, mode="w"
        ) as f:
            f.write(TRAIN_TEXT)
            filepath = f.name
        try:
            tok = BPETokenizer.from_file(filepath, num_merges=5)
            assert tok.vocab_size > 256
        finally:
            os.unlink(filepath)
