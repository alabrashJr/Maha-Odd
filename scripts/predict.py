import json

import torch
from torch.utils.data import DataLoader
import numpy as np
from configs.config import ExpConfig
from lib import score_functions
from lib.dataclass_utils import datacli
from lib.modules.transformer_classifier import TransformerModule, TransformerClassifier, DistilBertWrapper
from lib.datasets import datasets
from scripts._run_transformer import get_tokenizer_and_model, bert_collate
from scripts._run_transformer import  get_kwargs_for_mahalanobis_score

def predict(self, dataset, batch_size=3,):
    train_outputs = []
    self.eval()
    with torch.no_grad():
        data_loader = DataLoader(dataset, batch_size=batch_size,
                                 collate_fn=self.collate_fn, shuffle=False, num_workers=0)
        for idx, batch in enumerate(data_loader):
            seq, att_idxs, labels, is_ood = batch['seq'], batch['attention_mask'], batch['labels'], batch['is_ood']
            logits = self.transformer(seq.to(self.device), att_idxs.to(self.device))
            feats = self.transformer.feats
            res = {
                'labels': labels,
                'logits': logits,
                'feats': feats,
                'predict_cls': np.argmax(logits.cpu().numpy(),axis=1)
            }
            train_outputs.append(res)
    self.train(True)
    train_labels = torch.cat([x['labels'] for x in train_outputs]).cpu().numpy().astype(int)
    # train_feats = torch.cat([label_dict[x['predict_cls']] for x in train_outputs])
    predict_labels = [label_dict[str(q)] for x in train_outputs for q in x['predict_cls']]
    return train_labels, predict_labels

if __name__ == '__main__':
    with open('/home/abdurrahman.beyaz/Downloads/task/Maha_OOD/Abd_test/label_dict.json') as f:
        label_dict = json.load(f)
    config = datacli(ExpConfig)
    tokenizer, transformer = get_tokenizer_and_model(config.bert_type)
    transformer_classifier = TransformerClassifier(
        transformer, config.hidden_dropout_prob, config.n_labels
    )

    collate_fn = bert_collate(tokenizer)
    score_cls = score_functions.SCORE_FUNCTION_REGISTRY[config.score_type]
    if issubclass(score_cls, score_functions.LogitsScoreFunction):
        score_function = score_cls(temperature=config.temperature)
    elif issubclass(score_cls, score_functions.AbstractMahalanobisScore):
        kwargs = get_kwargs_for_mahalanobis_score(score_cls, transformer_classifier, config)
        score_function = score_cls(**kwargs)
    else:
        raise ValueError(f'Unknown score class {score_cls}')

    module = TransformerModule(config, transformer_classifier,
                               score_function, collate_fn, tokenizer,
                               train_dataset=None, val_dataset=None, test_dataset=None)


    module.load_state_dict(torch.load("model.pt"))
    # module = module.to(device)

    test__dataset = datasets.get_test_transformers(
        tokenizer=tokenizer,
        dataset_name=config.data_name,
        data_path='/home/abdurrahman.beyaz/Downloads/task/Maha_OOD/data/clinc/test.json',
        version=config.version,
        ood_type=config.ood_type)

    module.predict = predict.__get__(module)

    print(module.predict(test__dataset))