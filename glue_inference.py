from fairseq.models.roberta_custom import RobertaModelFaug
from fairseq.models.roberta import RobertaModel
label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
ncorrect, nsamples = 0, 0

# dropblock 0.1 seed 1
path = 'outputs/2022-08-09/21-34-39/checkpoints/'

# adaptive 0.1 seed 1
path = 'outputs/2022-08-09/21-36-31/checkpoints/'
# channel 0.1 seed 0
path = "outputs/2022-08-09/22-18-24/checkpoints/"

paths = ['15-27-07', '15-32-37', '15-42-56']
paths = [f"outputs/2022-08-11/{p}/checkpoints/"]



model = RobertaModel.from_pretrained(path, checkpoint_file='checkpoint_best.pt', data_name_or_path='SST-2-bin')
model.cuda()
model.eval()

with open('glue_data/SST-2/processed/train.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, label = tokens[0], tokens[1]
        tokens = model.encode(sent1, sent1)
        prediction = model.predict('sentence_classification_head', tokens).argmax().item()
        target = int(label)
        ncorrect += int(prediction == target)
        print(prediction, target)
        nsamples += 1
print('| Accuracy: ', float(ncorrect)/float(nsamples))
