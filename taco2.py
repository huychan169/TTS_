import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, x):
        return self.embedding(x)

class CBHG(nn.Module):
    def __init__(self, in_channels, K=16):
        super().__init__()
        self.convbanks = nn.ModuleList([
            nn.Conv1d(in_channels, 128, kernel_size=k) 
            for k in range(1, K+1)
        ])
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        self.projconv = nn.Conv1d(128 * K, 128, kernel_size=3, padding=1)
        self.highway = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.gru = nn.GRU(128, 128, bidirectional=True, batch_first=True)
    
    def forward(self, x):
        conv_outputs = []
        for conv in self.convbanks:
            out = F.relu(conv(x))
            out = self.pool(out)
            conv_outputs.append(out)
        
        conv_output = torch.cat(conv_outputs, dim=1)
        proj_output = self.projconv(conv_output)
        
        highway_output = self.highway(proj_output.transpose(1, 2)).transpose(1, 2)
        gru_output, _ = self.gru(highway_output.transpose(1, 2))
        
        return gru_output

class Tacotron2(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, n_mels=80):
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            CBHG(256)
        )
        
        # Decoder
        self.decoder = nn.GRU(256, 256, batch_first=True)
        self.mel_linear = nn.Linear(256, n_mels)
    
    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Tacotron2
        
        Args:
            text_tokens (torch.Tensor): Tokenized input text
        
        Returns:
            torch.Tensor: Predicted mel spectrogram
        """
        # Embedding
        embedded = self.embedding(text_tokens)
        
        # Encoder
        encoded = self.encoder(embedded)
        
        # Decoder
        decoder_out, _ = self.decoder(encoded)
        
        # Mel spectrogram prediction
        mel_pred = self.mel_linear(decoder_out)
        
        return mel_pred

def create_vocab(processed_dataset: List[Tuple[List[str], np.ndarray, np.ndarray]]) -> Dict[str, int]:
    """
    Create vocabulary from processed dataset
    
    Args:
        processed_dataset: List of (tokens, mel_spec, audio)
    
    Returns:
        Dictionary mapping tokens to indices
    """
    all_tokens = [token for tokens, _, _ in processed_dataset for token in tokens]
    unique_tokens = sorted(set(all_tokens))
    return {token: idx for idx, token in enumerate(unique_tokens)}

# Example usage
vocab = create_vocab(processed_dataset)
vocab_size = len(vocab)

model = Tacotron2(vocab_size=vocab_size)

# Convert text tokens to indices for model input
def text_to_tensor(tokens: List[str], vocab: Dict[str, int]) -> torch.Tensor:
    return torch.tensor([vocab.get(token, vocab.get('<UNK>', 0)) for token in tokens])

# Example inference
sample_tokens = processed_dataset[0][0]  # First dataset sample's tokens
input_tensor = text_to_tensor(sample_tokens, vocab)
mel_prediction = model(input_tensor.unsqueeze(0))