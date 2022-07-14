from fairseq.models.roberta_custom import RobertaModelFaug
from fairseq.models.roberta import RobertaModel
label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
ncorrect, nsamples = 0, 0

path = 'outputs/2022-07-13/12-08-52/checkpoints/'
model = RobertaModel.from_pretrained(path, checkpoint_file='checkpoint_best.pt', data_name_or_path='MNLI-bin')
model.cuda()
model.eval()

with open('glue_data/MNLI/dev_matched.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[8], tokens[9], tokens[-1]
        tokens = model.encode(sent1, sent2)
        prediction = model.predict('sentence_classification_head', tokens).argmax().item()
        prediction_label = label_map[prediction]
        ncorrect += int(prediction_label == target)
        nsamples += 1
print('| Accuracy: ', float(ncorrect)/float(nsamples))
