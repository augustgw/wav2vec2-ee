"""
@author : George Wright & Daniele Falavigna
@when : 2023-03-30
@homepage : https://github.com/augustgw/early-exit-transformer
"""

from conf import *
# import torchaudio
# from util.data_loader import collate_fn
from datasets import load_dataset

# train_dataset1 = torchaudio.datasets.LIBRISPEECH(
#     "/corpora", url="train-clean-100", download=False)
# train_dataset2 = torchaudio.datasets.LIBRISPEECH(
#     "/corpora", url="train-clean-360", download=False)
# train_dataset3 = torchaudio.datasets.LIBRISPEECH(
#     "/corpora", url="train-other-500", download=False)
# train_dataset = torch.utils.data.ConcatDataset(
#     [train_dataset1, train_dataset2, train_dataset3])

# train_dataset = torchaudio.datasets.LIBRISPEECH(
#     "/mnt/c/Projects/wav2vec2/wav2vec2/", url="dev-clean", download=True)

train_dataset = load_dataset("librispeech_asr", split="validation.clean", streaming=True)
eval_dataset = load_dataset("librispeech_asr", split="validation.clean", streaming=True)

# train_dataloader = torch.utils.data.DataLoader(
#     train_dataset, pin_memory=False, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers)

# eval_dataset = torchaudio.datasets.LIBRISPEECH(
#     "/mnt/c/Projects/wav2vec2/wav2vec2/", url="dev-clean", download=True)

# eval_dataloader = torch.utils.data.DataLoader(
#     eval_dataset, pin_memory=False, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers)

# data_files = {"train100": "train-clean-100",
#               "train360": "train-clean-360", "train500": "train-other-500"}
# dataset = load_dataset("namespace/your_dataset_name", data_files=data_files)

# print(train_dataset[0])
