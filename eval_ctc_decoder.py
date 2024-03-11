import sys
import os
import torch
from transformers import AutoProcessor
from datasets import load_dataset
from tqdm.auto import tqdm
from evaluate import load
from pyctcdecode import build_ctcdecoder
from train_utils import *

if len(sys.argv) != 3:
    sys.exit('Usage: eval.py path/to/checkpoint path/to/results/directory')
else:
    checkpoint_dir = sys.argv[1]
    results_dir = os.getcwd() + '/' + sys.argv[2]
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
model = EEWav2Vec2ForCTC.from_pretrained(
    checkpoint_dir, output_hidden_states=True)
# model = NewKDEEWav2Vec2ForCTC.from_pretrained(checkpoint_dir, processor=processor, output_hidden_states=True)
wer = load('wer')

model.eval()
ctc_decoder = build_ctcdecoder(list(processor.tokenizer.get_vocab().keys()))


def inference(items, outfile, results):

    inputs = processor((items)["audio"]["array"],
                       sampling_rate=sampling_rate, return_tensors="pt")
    outfile.write('file: ' + items['file'] +
                  '\nexpected: ' + items['text'].lower() + '\n')
    results['expected'].append(items['text'].lower())

    with torch.no_grad():
        outputs = model(**inputs)
        # logits = outputs.logits
        # print(logits.shape)

    hidden_states = outputs.hidden_states
    # print(hidden_states[0].shape)

    for i in range(6):
        predicted_ids = model.decoders[i](hidden_states[(i+1)*2]).detach().numpy()[0]
        # print(predicted_ids.shape)
        transcription = ctc_decoder.decode(predicted_ids)
        outfile.write('\tlayer_' + str((i+1)*2) + ': ' +
                      transcription.lower() + '\n')
        results['layer_' + str((i+1)*2)].append(transcription.lower())
    outfile.write('\n')


werfile = open(results_dir + '/wer_results.txt', 'w')

# ['validation.clean','validation.other','test.clean','test.other']:
for split in ['test.clean']:
    print(split)
    werfile.write('split: ' + split + '\n')

    dataset = load_dataset("librispeech_asr", split=split, streaming=True)
    sampling_rate = dataset.features["audio"].sampling_rate

    outfile = open(results_dir + '/' + split + '_results.txt', 'w')
    results = {'expected': [], 'layer_2': [], 'layer_4': [],
               'layer_6': [], 'layer_8': [], 'layer_10': [], 'layer_12': []}

    for batch in tqdm(dataset):
        inference(batch, outfile, results)

    for layer in results.keys():
        if layer == 'expected':
            continue
        wer_score = wer.compute(
            predictions=results[layer], references=results['expected'])
        print(layer + ' WER: ' + str(wer_score))
        werfile.write(layer + ' WER: ' + str(wer_score) + '\n')
    werfile.write('\n')

    outfile.close()

werfile.close()
