import re
from string import ascii_lowercase
import torch

class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert isinstance(item, int)
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.tensor(
                [self.char2ind[char] for char in text],
                dtype=torch.long
            )
        except KeyError:
            unknown_chars = [char for char in text if char not in self.char2ind]
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        return "".join(
            [self.ind2char[int(ind)] for ind in inds]
        ).strip()

    def ctc_decode(self, inds) -> str:
        decoded_chars = []
        prev_char = None
        for i in inds:
            char = self.ind2char[int(i)]
            if char == self.EMPTY_TOK:
                prev_char = None
                continue
            if char == prev_char:
                continue
            decoded_chars.append(char)
            prev_char = char

        return "".join(decoded_chars).strip()

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
