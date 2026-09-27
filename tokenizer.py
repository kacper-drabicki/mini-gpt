class CharacterTokenizer:

    def __init__(self, text=None, chars=None):
        assert text is not None or chars is not None, 'Either text or chars must be provided!'
        self.chars = sorted(list(set(text))) if not chars else chars
        self.str_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_to_str = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text):
        return [self.str_to_int[ch] for ch in text]

    def decode(self, ids):
        return ''.join([self.int_to_str[i] for i in ids])
