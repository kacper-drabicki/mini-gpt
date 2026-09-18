class CharacterTokenizer:

    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.str_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_to_str = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text):
        return [self.str_to_int[ch] for ch in text]

    def decode(self, ids):
        return ''.join([self.int_to_str[i] for i in ids])
