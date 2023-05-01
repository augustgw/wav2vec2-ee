from conf import *
from train_utils import *
from transformers import Wav2Vec2Processor
from datasets import load_dataset, load_from_disk, concatenate_datasets
from tqdm import tqdm

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

# train_dataset = load_dataset("librispeech_asr", split="train.clean.100")
# train_dataset = train_dataset.map(map_to_array, remove_columns=['file','audio','speaker_id','chapter_id','id'])
# # new_column = ["foo"] * len(train_dataset)
# # train_dataset = train_dataset.add_column("labels", new_column).add_column("input_values", new_column)
# train_dataset = train_dataset.map(preprocess, fn_kwargs={'processor':processor}, remove_columns=['text','array'], batched=True)
# # train_dataset = train_dataset.rename_column("text", "labels")
# # train_dataset = train_dataset.rename_column("array", "input_values")
# train_dataset.save_to_disk("preprocessed_data/preprocessed-train.clean.100")

# train_dataset_2 = load_dataset("librispeech_asr", split="train.clean.360")
# for i in tqdm(range(10)):
#     train_dataset_2.append(preprocess(processor, train_dataset_2.shard(num_shards=10, index=i, contiguous=True)))
# train_dataset_2.save_to_disk("preprocessed_data/preprocessed-train.clean.360")

# train_dataset_3 = load_dataset("librispeech_asr", split="train.other.500")
# for i in tqdm(range(10)):
#     train_dataset_3.append(preprocess(processor, train_dataset_3.shard(num_shards=10, index=i, contiguous=True)))
# train_dataset_3.save_to_disk("preprocessed_data/preprocessed-train.other.500")

# eval_dataset = load_from_disk("data/train_dataset")
# for i in tqdm(range(10)):
#     eval_dataset.append(preprocess(processor, eval_dataset.shard(num_shards=10, index=i, contiguous=True)))
# eval_dataset.save_to_disk("preprocessed_data/preprocessed-eval.clean")

# train_dataset_1 = load_dataset("librispeech_asr", split="train.clean.100")
# train_dataset_1 = train_dataset_1.map(map_to_array, remove_columns=['file','audio','speaker_id','chapter_id','id'])
# train_dataset_1 = train_dataset_1.map(preprocess, fn_kwargs={'processor':processor}, remove_columns=['array','text'], batched=True)
# train_dataset_1.save_to_disk("preprocessed_data/preprocessed-train_dataset_1.clean")

train_dataset_2 = load_dataset("librispeech_asr", split="train.clean.360")
train_dataset_2 = train_dataset_2.map(map_to_array, remove_columns=['file','audio','speaker_id','chapter_id','id'])
train_dataset_2 = train_dataset_2.map(preprocess, fn_kwargs={'processor':processor}, remove_columns=['array','text'], batched=True)
train_dataset_2.save_to_disk("preprocessed_data/preprocessed-train_dataset_2.clean")

# train_dataset_3 = load_dataset("librispeech_asr", split="train.other.500")
# train_dataset_3 = train_dataset_3.map(map_to_array, remove_columns=['file','audio','speaker_id','chapter_id','id'])
# train_dataset_3 = train_dataset_3.map(preprocess, fn_kwargs={'processor':processor}, remove_columns=['array','text'], batched=True)
# train_dataset_3.save_to_disk("preprocessed_data/preprocessed-train_dataset_3.clean")

# eval_dataset = load_dataset("librispeech_asr", split="validation.clean")
# eval_dataset = eval_dataset.map(map_to_array, remove_columns=['file','audio','speaker_id','chapter_id','id'])
# eval_dataset = eval_dataset.map(preprocess, fn_kwargs={'processor':processor}, remove_columns=['array','text'], batched=True)
# eval_dataset.save_to_disk("preprocessed_data/preprocessed-eval.clean")