import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(dataset_items: list[dict]):
    spectrograms = [item["spectrogram"] for item in dataset_items]
    texts_encoded = [item["text_encoded"] for item in dataset_items]
    texts = [item["text"] for item in dataset_items]
    audio_paths = [item["audio_path"] for item in dataset_items]
    spectrograms_padded = pad_sequence(
        spectrograms, batch_first=True, padding_value=0.0
    )

    texts_padded = pad_sequence(
        texts_encoded, batch_first=True, padding_value=0
    )
    spectrogram_lengths = torch.tensor(
        [s.shape[0] for s in spectrograms], dtype=torch.long
    )
    text_lengths = torch.tensor(
        [t.shape[0] for t in texts_encoded], dtype=torch.long
    )
    batch = {
        "spectrogram": spectrograms_padded,
        "spectrogram_length": spectrogram_lengths,
        "text_encoded": texts_padded,
        "text_encoded_length": text_lengths,
        "text": texts,
        "audio_path": audio_paths, 
    }
    return batch
