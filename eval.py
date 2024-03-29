from transformers import AutoProcessor, Wav2Vec2ForCTC
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from evaluate import load
from train_utils import *
import sys
import os

if len(sys.argv) != 3:
     sys.exit('Usage: eval.py path/to/checkpoint path/to/results/directory')
else:
     checkpoint_dir = sys.argv[1]
     results_dir = os.getcwd() + '/' + sys.argv[2]
     if not os.path.exists(results_dir):
          os.mkdir(results_dir)

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
model = EEWav2Vec2ForCTC.from_pretrained(checkpoint_dir, output_hidden_states=True)

# Note: newer versions of HuggingFace save model checkpoints as 'safetensors' objects. Use this line to load these newer checkpoints.
# model = EEWav2Vec2ForCTC.from_pretrained(checkpoint_dir, use_safetensors=True, output_hidden_states=True)

wer = load('wer')

model.eval()

def inference(items, outfile, results):

     audio_arrays = [item["audio"]["array"] for item in items]
     files = [item['file'] for item in items]
     texts = [item['text'] for item in items]

     inputs = processor(audio_arrays, sampling_rate=sampling_rate, return_tensors="pt")
     outfile.write('file: ' + files + '\nexpected: ' + texts + '\n')
     results['expected'].append(texts.lower())

     with torch.no_grad():
          outputs = model(**inputs)
          logits = outputs.logits

     predicted_ids = torch.argmax(logits, dim=-1)

     transcription = processor.batch_decode(predicted_ids)

     hidden_states = outputs.hidden_states

     for i in range(6):
         # print(len(hidden_states[i]))
         predicted_ids = torch.argmax(model.decoders[i](hidden_states[(i+1)*2]), dim=-1)
         transcription = processor.batch_decode(predicted_ids)
         outfile.write('\tlayer_' + str((i+1)*2) + ': ' + transcription[0].lower() + '\n')
         results['layer_' + str((i+1)*2)].append(transcription[0].lower())
     outfile.write('\n')


werfile = open(results_dir + '/wer_results.txt', 'w')

for split in ['test.clean','test.other']: # ['validation.clean','validation.other','test.clean','test.other']:
     print(split)
     werfile.write('split: ' + split + '\n')

     dataset = load_dataset("librispeech_asr", split=split, streaming=True)
     sampling_rate = dataset.features["audio"].sampling_rate

     outfile = open(results_dir + '/' + split + '_results.txt', 'w')
     results = {'expected':[],'layer_2':[],'layer_4':[],'layer_6':[],'layer_8':[],'layer_10':[],'layer_12':[]}

     for batch in tqdm(dataset.iter(batch_size=16)):
          inference(batch, outfile, results)

     for layer in results.keys():
          if layer == 'expected':
               continue
          wer_score = wer.compute(predictions=results[layer], references=results['expected'])
          print(layer + ' WER: ' + str(wer_score))
          werfile.write(layer + ' WER: ' + str(wer_score) + '\n')
     werfile.write('\n')

     outfile.close()

werfile.close()
