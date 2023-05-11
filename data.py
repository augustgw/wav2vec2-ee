"""
@author : George Wright & Daniele Falavigna
@when : 2023-03-30
@homepage : https://github.com/augustgw/early-exit-transformer
"""

import torchaudio
from datasets import load_dataset, load_from_disk, concatenate_datasets

train_dataset1 = torchaudio.datasets.LIBRISPEECH(
    "/workspace/wav2vec2/", url="train-clean-100", download=False)
train_dataset2 = torchaudio.datasets.LIBRISPEECH(
    "/workspace/wav2vec2/", url="train-clean-360", download=False)
train_dataset3 = torchaudio.datasets.LIBRISPEECH(
    "/workspace/wav2vec2/", url="train-other-500", download=False)
train_dataset = torch.utils.data.ConcatDataset(
    [train_dataset1, train_dataset2, train_dataset3])

# train_dataset = torchaudio.datasets.LIBRISPEECH(
#     "/workspace/wav2vec2/", url="dev-clean", download=False)

eval_dataset = torchaudio.datasets.LIBRISPEECH(
    "/workspace/wav2vec2/", url="dev-clean", download=False)