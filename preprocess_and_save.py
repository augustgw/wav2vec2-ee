import torchaudio
from transformers import Wav2Vec2Processor
from train_utils import *
from tqdm import tqdm
from datasets import Dataset

splits = ["train-clean-100","train-clean-360","train-other-500"]

for split in splits:
    train_dataset = torchaudio.datasets.LIBRISPEECH(
        "/workspace", url=split, download=False)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    i = 0
    j = 0
    train_processed = list()
    for k in tqdm(range(len(train_dataset))):
        item_processed = preprocess_item(processor, train_dataset[k])
        train_processed.append(item_processed)
        j += 1
        if (j >= 5000 or k == (len(train_dataset) - 1)):
            hf_dataset = Dataset.from_list(train_processed)
            hf_dataset.save_to_disk("LibriSpeech_processed/" + split + "-" + str(i))
            del hf_dataset
            train_processed = list()
            i += 1
            j = 0
