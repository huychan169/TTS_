# import torch
# import torch.nn as nn

# class AudioDecoder(nn.Module):
#     def __init__(self, hidden_dim=512, output_dim=80):  # 80 mel spectrogram bins
#         super().__init__()
#         self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
#         self.output_layer = nn.Linear(hidden_dim, output_dim)
        
#     def forward(self, hidden, cell):
#         output, (new_hidden, new_cell) = self.lstm((hidden, cell))
#         mel_output = self.output_layer(output)
#         return mel_output, new_hidden, new_cell

class Tacotron2Decoder(nn.Module):
    def __init__(self, hidden_dim, n_mels):
        super().__init__()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, n_mels)
        self.linear = nn.Linear(hidden_dim, n_mels)

    def forward(self, encoder_outputs):
        x, _ = self.lstm(encoder_outputs)
        attention_weights = self.attention(x)
        mel_outputs = self.linear(attention_weights)
        return mel_outputs
