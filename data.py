"""
@author : George Wright & Daniele Falavigna
@when : 2023-03-30
@homepage : https://github.com/augustgw/early-exit-transformer
"""
import torch
import torchaudio
from datasets import load_from_disk, concatenate_datasets

# train_dataset1 = torchaudio.datasets.LIBRISPEECH(
#     "/workspace", url="train-clean-100", download=False)
# train_dataset2 = torchaudio.datasets.LIBRISPEECH(
#     "/workspace", url="train-clean-360", download=False)
# train_dataset3 = torchaudio.datasets.LIBRISPEECH(
#     "/workspace", url="train-other-500", download=False)
# train_dataset = torch.utils.data.ConcatDataset(
#     [train_dataset1, train_dataset2, train_dataset3])

# Small train for testing
# train_dataset = torchaudio.datasets.LIBRISPEECH(
#     "/workspace", url="dev-clean", download=False)

train_dataset = load_from_disk("LibriSpeech_processed/train-clean-100-1")

splits = ["train-clean-100-2", "train-clean-100-3", "train-clean-360-1", "train-clean-360-2",
        "train-clean-360-3", "train-clean-360-4", "train-clean-360-5", "train-clean-360-6",
        "train-clean-360-7", "train-clean-360-8", "train-clean-360-9", "train-clean-360-10",
        "train-clean-360-11", "train-other-500-1", "train-other-500-2", "train-other-500-3",
        "train-other-500-4", "train-other-500-5", "train-other-500-6", "train-other-500-7",
        "train-other-500-8", "train-other-500-9", "train-other-500-10", "train-other-500-11",
        "train-other-500-12", "train-other-500-13", "train-other-500-14", "train-other-500-15"]
for split in splits:
    train_dataset = concatenate_datasets([
        train_dataset, 
        load_from_disk("LibriSpeech_processed/" + split)
    ])

eval_dataset = torchaudio.datasets.LIBRISPEECH(
    "/workspace", url="dev-clean", download=False)