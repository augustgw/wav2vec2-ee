# wav2vec2-ee

Wav2Vec2 model training with early-exit and knowledge distillation loss mechanisms.

# Usage

## Fine-tuning

**Note:** For all models, check the `TrainingArguments' block for training hyperparameters, output paths, starting training from a checkpoint, etc.

### Basic training

- Fine-tuning with only EE loss: `finetune_ee.py`
- Fine-tuning a model without early exits: `finetune_non-ee.py`
  - Change `model_config = Wav2Vec2Config(num_hidden_layers=X)` to set the number of layers in the encoder. E.g., for 4-layer encoder: `model_config = Wav2Vec2Config(num_hidden_layers=4)`

### Knowledge distillation
  
- Fine-tuning with joint EE + KD loss: `finetune_kd.py`
  - Change `ee_alpha` to change weights in joint loss: `loss = (ee_alpha * ee_loss) + ((1 - ee_alpha) * kd_loss)` (default: `0.3`).
- Fine-tuning with dynamically weighted joint EE + KD loss: `finetune_dkd.py`
  - Change `ee_alpha` to change weights in joint loss. `ee_alpha` is a list of weights corresponding to each exit of the model. The length of `ee_alpha` must be equal to the number of exits in the model (default: `[0.65, 0.70, 0.75, 0.80, 0.85, 1.00]`).

### Confidence
  
- Fine-tuning with confidence-based EE loss: `finetune_confidence.py`
  -  Change `inverse_confidence` to change application of confidence scores on CTC loss. `True` multiplies CTC loss by `1/confidence_score`; `False` multiplies CTC loss by `confidence_score`. (default: `True`).

### Manual exit activation (choosing the number of exits with which to calculate loss)
  
- Normal fine-tuning: `finetune_manual_activation.py`
  -  Change `num_exits` to change the number of exits used to calculate loss. For example, when `num_exits = 2`, the output of the first and second exits in the model (e.g., Layers 2 and 4) will be used to calculate loss.
- Fine-tuning with confidence: `finetune_manual_activation_confidence.py`
  -  Change `num_exits` to change the number of exits used to calculate loss. For example, when `num_exits = 2`, the output of the first and second exits in the model (e.g., Layers 2 and 4) will be used to calculate loss.
  -  Change `inverse_confidence` to change application of confidence scores on CTC loss. `True` multiplies CTC loss by `1/confidence_score`; `False` multiplies CTC loss by `confidence_score`. (default: `True`).


## Evaluation

The evaluation scripts create files in the indicated output directory. `wer_results.txt` contains the layerwise WERs on the test sets indicated in the evaluation script. The remaining files contain the layerwise transcriptions of each item in each test set.

### Basic evaluation

- Normal evaluation: `eval.py path/to/model/checkpoint path/to/output/directory`
  -   For safetensors checkpoints saved by newer versions of Hugging Face, see note in `eval.py`.s
- Evaluation for models without early exits (evaluates only output of final layer): `eval_non-ee.py path/to/model/checkpoint path/to/output/directory`
