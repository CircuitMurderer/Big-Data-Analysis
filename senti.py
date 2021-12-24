from paddlenlp.data import JiebaTokenizer, Stack, Pad, Tuple
from paddlenlp.embeddings import TokenEmbedding
from paddlenlp.datasets import load_dataset
from paddle.optimizer import AdamW
from paddle.io import DataLoader
import paddle.nn.functional as f
import paddle.nn as nn
import paddle

from functools import partial
from os.path import isfile

from model import TextCNN, train, evaluate
import pandas as pd


class Config:
    def __init__(self):
        self.epochs = 10
        self.lr = 1e-3
        self.max_seq_len = 256
        self.batch_size = 256

        self.root = './data/'
        self.train_path = self.root + 'train.tsv'
        self.dev_path = self.root + 'dev.tsv'
        self.test_path = self.root + 'test.tsv'

        self.save = './model/'
        self.ly_save = self.save + 'layer.pdparams'
        self.opt_save = self.save + 'opt.pdopt'

        self.embed_name = 'w2v.wiki.target.word-char.dim300'
        self.filters = 256
        self.dropout = 0.5
        self.classes = 2


class Tokenizer:
    def __init__(self, voc):
        self.vocab = voc
        self.UNK = '[UNK]'
        self.PAD = '[PAD]'
        self.tokenizer = JiebaTokenizer(self.vocab)
        self.unk_id = self.vocab.token_to_idx.get(self.UNK)
        self.pad_id = self.vocab.token_to_idx.get(self.PAD)

    def text_to_id(self, txt, max_s=512):
        input_id = []
        for tok in self.tokenizer.cut(txt):
            tok_id = self.vocab.token_to_idx.get(tok, self.unk_id)
            input_id.append(tok_id)
        return input_id[:max_s]


def read_handle(path, is_train=True, sep='\t'):
    dt = pd.read_csv(path, sep=sep)
    for _, row in dt.iterrows():
        yield {'label': row['label'], 'text_a': row['text_a']} \
            if is_train else {'text_a': row['text_a']}


def convert_exp(exp, tok, max_s):
    text_a = exp['text_a']
    text_a_id = tok.text_to_id(text_a, max_s)
    return (text_a_id, exp['label']) if 'label' in exp else (text_a_id,)


class Senti:
    def __init__(self):
        self.config = Config()
        self.embedding = TokenEmbedding(embedding_name=self.config.embed_name)

        self.tokenizer = Tokenizer(self.embedding.vocab)
        self.trans = partial(convert_exp, tok=self.tokenizer, max_s=self.config.max_seq_len)

        self.pad = lambda sample, fn=Tuple(
            Pad(dtype='int64', pad_val=self.tokenizer.pad_id, axis=0),
        ): fn(sample)

        self.pred_set = None
        self.pred_loader = None

        self.model = TextCNN(self.config, self.embedding)

        if isfile(self.config.ly_save):
            layer_dict = paddle.load(self.config.ly_save)
            self.model.set_state_dict(layer_dict)
        else:
            raise FileNotFoundError("Can't find model file. Please run 'senti.py' to train the model first.")

    def load(self, pred_path: str):
        if pred_path.endswith('csv'):
            sep = ','
        elif pred_path.endswith('tsv'):
            sep = '\t'
        else:
            raise NameError("The datasets' format error, which must be .csv or .tsv.")

        self.pred_set = load_dataset(read_handle, path=pred_path, is_train=False, sep=sep, lazy=False)
        self.pred_set.map(self.trans)

        self.pred_loader = DataLoader(dataset=self.pred_set, batch_size=len(self.pred_set),
                                      shuffle=False, return_list=True, collate_fn=self.pad)

    def predict(self, pred_path=None):
        if pred_path:
            self.load(pred_path)

        if self.pred_loader is None or self.pred_set is None:
            raise RuntimeError('No data has been loaded. Please load data first.')

        pred = []
        for _, (bat,) in enumerate(self.pred_loader):
            output = self.model(bat)
            # temp = paddle.argmax(f.softmax(output, axis=1), axis=1)
            # pred.extend(temp.numpy().tolist())

            temp = f.softmax(output, axis=1).numpy().tolist()
            for i0, i1 in temp:
                if i0 < 0.35:
                    pred.append(1)
                elif i0 > 0.65:
                    pred.append(-1)
                else:
                    pred.append(0)

        return pred


if __name__ == '__main__':
    config = Config()
    embedding = TokenEmbedding(embedding_name=config.embed_name)

    tokenizer = Tokenizer(embedding.vocab)
    trans = partial(convert_exp, tok=tokenizer, max_s=config.max_seq_len)

    print('Loading data...')
    train_set = load_dataset(read_handle, path=config.train_path, is_train=True, sep='\t', lazy=False)
    dev_set = load_dataset(read_handle, path=config.dev_path, is_train=True, sep='\t', lazy=False)
    test_set = load_dataset(read_handle, path=config.test_path, is_train=True, sep='\t', lazy=False)

    train_set.map(trans)
    dev_set.map(trans)
    test_set.map(trans)
    print('Done.')

    padding_train = lambda sample, fn=Tuple(
        Pad(dtype='int64', pad_val=tokenizer.pad_id, axis=0),
        Stack(dtype='int64'),
    ): fn(sample)

    padding_test = lambda sample, fn=Tuple(
        Pad(dtype='int64', pad_val=tokenizer.pad_id, axis=0),
    ): fn(sample)

    print('Collecting data...')
    train_loader = DataLoader(dataset=train_set,
                              batch_size=config.batch_size,
                              shuffle=True,
                              return_list=True,
                              collate_fn=padding_train)
    dev_loader = DataLoader(dataset=dev_set,
                            batch_size=config.batch_size,
                            shuffle=False,
                            return_list=True,
                            collate_fn=padding_train)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=config.batch_size,
                             shuffle=False,
                             return_list=True,
                             collate_fn=padding_train)

    print('Done.')

    model = TextCNN(config, embedding)
    optimizer = AdamW(learning_rate=config.lr, parameters=model.parameters())
    criterion = nn.CrossEntropyLoss()

    if isfile(config.ly_save):  # and isfile(config.opt_save):
        layer_dic = paddle.load(config.ly_save)
        # opt_dic = paddle.load(config.opt_save)
        model.set_state_dict(layer_dic)
        # optimizer.set_state_dict(opt_dic)
    else:
        train(model, config, optimizer, criterion, train_loader, dev_loader)
        paddle.save(model.state_dict(), config.ly_save)
        # paddle.save(optimizer.state_dict(), config.opt_save)

    print('On test set:')
    acc, f1 = evaluate(model, test_loader)
    print('Accuracy:', acc, 'F1:', f1)

