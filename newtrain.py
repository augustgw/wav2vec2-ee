from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments
from train_utils import *
from data import *
from tqdm import tqdm
import os 

os.environ["WANDB_DISABLED"] = "true"

training_args = TrainingArguments(
    output_dir='/workspace/wav2vec2/models',
    evaluation_strategy='no',
    # eval_steps=50,
    # save_total_limit=5,
    save_strategy = 'epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=24,
    per_device_eval_batch_size=1,
    num_train_epochs=100,
    weight_decay=0.01,
    push_to_hub=False,
    metric_for_best_model='f1',
    # report_to='wandb',
    eval_accumulation_steps=8,
    logging_strategy='epoch',
)

# * Load model
model = EEWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base", output_hidden_states=True)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

# * Train
model.freeze_feature_encoder() # Original Wav2Vec2 paper does not train encoder during fine-tuning
model.train()

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
