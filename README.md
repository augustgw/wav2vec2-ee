# wav2vec2-ee

Wav2Vec2 model training with early-exit and knowledge distillation loss mechanisms.

# Usage

- Fine-tuning with only EE loss: `finetune_tf.py`
- Fine-tuning with joint EE + KD loss: `finetune_kd_tf.py`
  - Change `ee_alpha` in `finetune_kd_tf.py` to change weights in joint loss: `loss = (ee_alpha * ee_loss) + ((1 - ee_alpha) * kd_loss)`
- Fine-tuning a model without early exits: `finetune_non-ee.py`
  - Change `model_config = Wav2Vec2Config(num_hidden_layers=X)` to set the number of layers in the encoder. E.g., for 4-layer encoder: `model_config = Wav2Vec2Config(num_hidden_layers=4)`

For all models, check the `TrainingArguments' block for training hyperparameters, output paths, starting training from a checkpoint, etc.

