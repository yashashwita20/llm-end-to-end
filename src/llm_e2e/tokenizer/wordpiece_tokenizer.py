from .tokenizer import Tokenizer

class WordPieceTokenizer(Tokenizer):
    def __init__(self, vocab_file, unk_token="[UNK]", sep_token="[SEP]", cls_token="[CLS]", pad_token="[PAD]", mask_token="[MASK]"):
        super().__init__(vocab_file, unk_token, sep_token, cls_token, pad_token, mask_token)
        