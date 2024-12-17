import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from data.preprocessing import VietnameseTextProcessor
from data.dataset import TTSDataset
from models.encoder import TextEncoder
from models.decoder import Tacotron2Decoder
from models.vocoder import WaveGlowVocoder


def prepare_dataloader(dataset_name, text_column, audio_column, processor, batch_size=32, num_workers=4):
    """
    Prepare DataLoader for training.
    """
    dataset = TTSDataset(
        dataset_name=dataset_name,
        text_column=text_column,
        audio_column=audio_column,
        processor=processor
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataset, dataloader


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Configurations
    config = {
        "dataset_name": "trinhtuyen201/my-audio-dataset",
        "text_column": "transcription",
        "audio_column": "audio",
        "batch_size": 16,
        "learning_rate": 0.001,
        "epochs": 100,
        "embedding_dim": 256,
        "hidden_dim": 512,
        "n_mels": 80
    }

    # Initialize text processor
    processor = VietnameseTextProcessor()

    # Prepare dataset and dataloader
    dataset, dataloader = prepare_dataloader(
        dataset_name=config["dataset_name"],
        text_column=config["text_column"],
        audio_column=config["audio_column"],
        processor=processor,
        batch_size=config["batch_size"]
    )

    # Build vocabulary based on dataset
    processor.build_vocabulary([item["transcription"] for item in dataset.dataset])

    # Initialize models
    encoder = TextEncoder(
        vocab_size=len(processor.char_to_idx),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"]
    ).to(device)
    decoder = Tacotron2Decoder(
        hidden_dim=config["hidden_dim"],
        n_mels=config["n_mels"]
    ).to(device)
    vocoder = WaveGlowVocoder().to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        list(encoder.parameters()) +
        list(decoder.parameters()) +
        list(vocoder.parameters()),
        lr=config["learning_rate"]
    )

    # Training loop
    for epoch in range(config["epochs"]):
        encoder.train()
        decoder.train()
        vocoder.train()

        total_loss = 0
        for batch in dataloader:
            text = batch['text'].to(device)  # Shape: (batch_size, seq_len)
            audio = batch['audio'].to(device)  # Shape: (batch_size, n_mels, time_steps)

            # Forward pass
            optimizer.zero_grad()

            # Encode text
            encoder_outputs = encoder(text)

            # Decode mel spectrogram
            mel_outputs = decoder(encoder_outputs)

            # Generate waveform
            predicted_waveform = vocoder(mel_outputs)

            # Compute loss (compare generated waveform with ground truth audio)
            loss = criterion(predicted_waveform, audio)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print epoch loss
        print(f"Epoch {epoch + 1}/{config['epochs']}, Loss: {total_loss / len(dataloader)}")

    # Save models
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'vocoder': vocoder.state_dict()
    }, 'tts_model.pth')


if __name__ == "__main__":
    train()




# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from datasets import load_dataset
# from data.preprocessing import VietnameseTextProcessor
# from data.dataset import TTSDataset
# from models.encoder import TextEncoder
# from models.decoder import Tacotron2Decoder
# from models.vocoder import WaveGlowVocoder



# def prepare_dataloader(dataset_path, batch_size=32, num_workers=4):
#     """
#     Prepare DataLoader for training
    
#     Args:
#         dataset_path (str): Path to Hugging Face Parquet dataset
#         batch_size (int): Batch size for training
#         num_workers (int): Number of worker processes for data loading
#     """
#     dataset = TTSDataset(
#         dataset_path=dataset_path, 
#         text_column='text',  # Adjust based on your dataset
#         audio_column='audio'  # Adjust based on your dataset
#     )

#     dataloader = DataLoader(
#         dataset, 
#         batch_size=batch_size, 
#         shuffle=True, 
#         num_workers=num_workers,
#         pin_memory=True
#     )
    
#     return dataset, dataloader

# def train():
    
#     # Hugging Face dataset path

#     dataset_path = load_dataset("trinhtuyen201/my-audio-dataset")
#     # dataset_path = "hf://datasets/trinhtuyen201/my-audio-dataset/data/train-*-of-*.parquet"
#     # Prepare dataset and dataloader
#     dataset, dataloader = prepare_dataloader(dataset_path)

#     # Initialize Models
#     encoder = TextEncoder(vocab_size=len(dataset.char_to_idx))
#     decoder = Tacotron2Decoder()
#     vocoder = WaveGlowVocoder()

#     # Loss and Optimizer
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(list(encoder.parameters()) + 
#                            list(decoder.parameters()) + 
#                            list(vocoder.parameters()))
    

#     # Training Loop
#     num_epochs = 3  # Adjust as needed
#     for epoch in range(num_epochs):
#         total_loss = 0
        
#         for batch in dataloader:
#             text = batch['text']
#             audio = batch['audio']

#             # Forward pass
#             optimizer.zero_grad()
            
#             hidden, cell = encoder(text)
#             mel_output, _, _ = decoder(hidden, cell)
#             waveform = vocoder(mel_output)

#             # Compute loss
#             loss = criterion(waveform, audio)

#             # Backward pass and optimization
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
        
#         # Print epoch loss
#         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")
    
#     # Save models
#     torch.save({
#         'encoder': encoder.state_dict(),
#         'decoder': decoder.state_dict(),
#         'vocoder': vocoder.state_dict()
#     }, 'tts_model.pth')

# if __name__ == "__main__":
#     train()

