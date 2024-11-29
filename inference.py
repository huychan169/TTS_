import torch
import soundfile as sf
from models.encoder import TextEncoder
from models.decoder import AudioDecoder
from models.vocoder import WaveformGenerator
from data.dataset import TTSDataset

def synthesize_speech(text, models, dataset):
    # Convert text to sequence using dataset's method
    text_sequence = dataset._process_text(text)
    text_tensor = torch.tensor(text_sequence).unsqueeze(0)

    # Unpack models
    encoder, decoder, vocoder = models

    # Encode text
    hidden, cell = encoder(text_tensor)

    # Decode to mel spectrogram
    mel_output, _, _ = decoder(hidden, cell)

    # Convert to waveform
    waveform = vocoder(mel_output)

    return waveform.detach().numpy()

def main():
    # Hugging Face dataset path
    dataset_path = "hf://datasets/trinhtuyen201/my-audio-dataset/data/train-*-of-*.parquet"
    
    # Load dataset
    dataset = TTSDataset(
        dataset_path=dataset_path,
        text_column='text',
        audio_column='audio'
    )

    # Load trained models
    checkpoint = torch.load('tts_model.pth')
    
    # Reinitialize models
    encoder = TextEncoder(vocab_size=len(dataset.char_to_idx))
    decoder = AudioDecoder()
    vocoder = WaveformGenerator()
    
    # Load state dicts
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    vocoder.load_state_dict(checkpoint['vocoder'])

    # Example text synthesis
    text = "Example text to synthesize"
    waveform = synthesize_speech(text, (encoder, decoder, vocoder), dataset)
    
    # Save audio
    sf.write('output_speech.wav', waveform, 22050)

if __name__ == "__main__":
    main()