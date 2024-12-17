import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from data.preprocessing import VietnameseTextProcessor
import librosa
import numpy as np

class TTSDataset(Dataset):
    def __init__(self, dataset_name, text_column, audio_column, processor, sample_rate=22050, max_length=200):
        """
        Dataset for TTS using Hugging Face datasets.

        Args:
            dataset_name (str): Name of the Hugging Face dataset.
            text_column (str): Column containing text.
            audio_column (str): Column containing audio.
            processor (VietnameseTextProcessor): Text processor.
            sample_rate (int): Target sample rate for audio.
            max_length (int): Maximum sequence length.
        """
        self.dataset = load_dataset(dataset_name)['train']
        self.text_column = text_column
        self.audio_column = audio_column
        self.processor = processor
        self.sample_rate = sample_rate
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item[self.text_column]
        audio= item[self.audio_column]["array"]

        # Process text
        text = self.processor.process_text(text)
        text_indices = self.processor.encode_text(text)
        # Truncate or pad text
        if len(text_indices) > self.max_length:
            text_indices = text_indices[:self.max_length]  # Cắt nếu dài hơn max_length
        else:
            pad_width = self.max_length - len(text_indices)
            text_indices = text_indices + [self.processor.char_to_idx['<PAD>']] * pad_width  # Padding nếu ngắn hơn max_length
        # Process audio
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate, n_mels=80)

        # Truncate or pad audio
        if mel_spec.shape[1] > self.max_length:
            mel_spec = mel_spec[:, :self.max_length]
        else:
            pad_width = self.max_length - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')

        return {
            'text': torch.tensor(text_indices, dtype=torch.long),
            'audio': torch.tensor(mel_spec, dtype=torch.float32)
        }

        # return text_indices, mel_spec
