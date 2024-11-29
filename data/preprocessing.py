import unicodedata
import re
from underthesea import word_tokenize

class VietnameseTextProcessor:
    def __init__(self):
        """Vietnamese text processor for tokenization and normalization."""
        self.vocab = {}
        self.char_to_idx = {}
        self.idx_to_char = {}

        # Special tokens
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1
        }

    def normalize_text(self, text):
        """Normalize text by removing accents and special characters."""
        text = unicodedata.normalize('NFC', text.lower())
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def tokenize_text(self, text):
        """Tokenize Vietnamese text."""
        tokens = word_tokenize(text)
        return ' '.join(tokens)

    def process_text(self, text):
        """Full pipeline for text processing."""
        normalized = self.normalize_text(text)
        tokenized = self.tokenize_text(normalized)
        return tokenized

    def build_vocabulary(self, texts):
        """Build vocabulary from a list of texts."""
        unique_chars = set(''.join(texts))
        self.char_to_idx = {char: idx + 2 for idx, char in enumerate(sorted(unique_chars))}
        self.char_to_idx.update(self.special_tokens)
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

    def encode_text(self, text):
        """Encode text to indices."""
        return [self.char_to_idx.get(char, self.char_to_idx['<UNK>']) for char in text]

    def decode_text(self, indices):
        """Decode indices back to text."""
        return ''.join([self.idx_to_char.get(idx, '<UNK>') for idx in indices])
