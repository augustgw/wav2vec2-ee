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
model = Wav2Vec2ForCTC.from_pretrained(checkpoint_dir, output_hidden_states=True)
wer = load('wer')

def inference(items, outfile, results):

    inputs = processor((items)["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
    outfile.write('file: ' + items['file'] + '\nexpected: ' + items['text'].lower() + '\n')
    results['expected'].append(items['text'].lower())

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_ids = torch.argmax(logits, dim=-1)

    transcription = processor.batch_decode(predicted_ids)

    hidden_states = outputs.hidden_states

    predicted_ids = torch.argmax(model.lm_head(hidden_states[-1]), dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    outfile.write('\tpredicted: ' + transcription[0].lower() + '\n')
    results['predicted'].append(transcription[0].lower())

werfile = open(results_dir + '/wer_results.txt', 'w')

for split in ['test.clean','test.other']: # ['validation.clean','validation.other','test.clean','test.other']:
    print(split)
    werfile.write('split: ' + split + '\n')

    dataset = load_dataset("librispeech_asr", split=split, streaming=True)
    sampling_rate = dataset.features["audio"].sampling_rate

    outfile = open(results_dir + '/' + split + '_results.txt', 'w')
    results = {'expected':[],'predicted':[]}

    for batch in tqdm(dataset):
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