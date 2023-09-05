from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments
from train_utils import *
from data import *
from tqdm import tqdm
import os 
import torch

# torch.set_num_threads(1)

training_args = TrainingArguments(
    output_dir='/workspace/ee_finetuning_models',
    evaluation_strategy='no',
    # eval_steps=50,
    # save_total_limit=5,
    save_strategy = 'epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=1,
    num_train_epochs=50,
    weight_decay=0.01,
    push_to_hub=False,
    report_to='wandb',
    logging_strategy='epoch',
    dataloader_num_workers=10,
)

# * Load model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = EEWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")

# * Train
model.freeze_feature_encoder() # Original Wav2Vec2 paper does not train feature encoder during fine-tuning
model.train()

# train_list = tuple(map(preprocess_dict_elem, (processor, train_list)))

# train_list = (preprocess_dict_elem(processor, x) for x in train_list)

# train_batch_list = [train_list[x:x+500] for x in range(0, len(train_list), 500)]

# train_dataset = []
# for batch in tqdm(batched(train_list, 5000)):
#     # print(batch[0])
#     train_dataset.append(preprocess_dict(processor, batch))
# # eval_dataset = preprocess(processor, eval_dataset)

trainer = get_trainer(
    model, processor, training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset)

try:
    trainer.train()
except KeyboardInterrupt:
    print('Training early stopped by KeyboardInterrupt.')

# * Save
trainer.save_model()
