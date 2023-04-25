from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments
from train_utils import get_trainer, preprocess
from data import *

training_args = TrainingArguments(
    f"training_with_callbacks",
    evaluation_strategy='steps',
    eval_steps=50,  # Evaluation and Save happens every 50 steps
    # Only last 5 models are saved. Older ones are deleted.
    save_total_limit=5,
    learning_rate=2e-5,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    num_train_epochs=10000,
    weight_decay=0.01,
    push_to_hub=False,
    metric_for_best_model='f1',
    load_best_model_at_end=True,
    max_steps=10000
)

# * Load model
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base", output_hidden_states=True)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

# * Train
model.freeze_feature_encoder() # Original Wav2Vec2 paper does not train encoder during fine-tuning
model.train()
trainer = get_trainer(
    model, processor, training_args,
    train_dataset=preprocess(processor, train_dataset),
    eval_dataset=preprocess(processor, eval_dataset))

try:
    trainer.train()
except KeyboardInterrupt:
    print('Training early stopped by KeyboardInterrupt.')

# * Save
model.save_pretrained()
processor.save_pretrained()