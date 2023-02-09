import torch
from torch.utils.data import Dataset, DataLoader
from torchtext import data
from torchtext import datasets
from collections import Counter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
import random
import logging
import gensim.downloader
from tqdm import tqdm


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

random.seed(42)
torch.manual_seed(42)

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    train_data = UDPOSTags("train")
    val_data = UDPOSTags("val", train_data.text_vocab, train_data.tag_vocab)
    test_data = UDPOSTags("test", train_data.text_vocab, train_data.tag_vocab)
    batch_size = 256
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_data, batch_size=200, shuffle=False, collate_fn=pad_collate)

    embeddings = train_data.text_vocab.vocab.vectors
    text_pad_idx = train_data.text_vocab.vocab.stoi[train_data.text_vocab.pad_token]
    tag_pad_idx = train_data.tag_vocab.vocab.stoi[train_data.tag_vocab.pad_token]

    model = PosBiLSTM(len(train_data.tag_vocab.vocab), embeddings, text_pad_idx)
    model.to(dev)
    train_model(model, train_loader, val_loader, tag_pad_idx, epochs=20)


class UDPOSTags(Dataset):
    def __init__(self, split="train", text_vocab=None, tag_vocab=None):
        TEXT = data.Field(lower=True)
        UD_TAGS = data.Field(unk_token=None)
        fields = (("text", TEXT), ("udtags", UD_TAGS))
        train_data, valid_data, test_data = datasets.UDPOS.splits(fields)
        if split=="train":
            self.lang_data = train_data
        elif split=="val":
            self.lang_data = valid_data
        else:
            self.lang_data = test_data

        self.text_data = [vars(e)["text"] for e in self.lang_data.examples]
        self.tags_data = [vars(e)["udtags"] for e in self.lang_data.examples]

        if text_vocab:
            self.text_vocab = text_vocab
        else:
            TEXT.build_vocab(self.lang_data, min_freq=2, vectors="glove.6B.300d", unk_init=torch.Tensor.normal_)
            self.text_vocab = TEXT
        if tag_vocab:
            self.tag_vocab = tag_vocab
        else:
            UD_TAGS.build_vocab(self.lang_data)
            self.tag_vocab = UD_TAGS

    def __len__(self):
        return len(self.tags_data)

    def __getitem__(self, idx):
        numeralized_text = [self.text_vocab.vocab.stoi[t] for t in self.text_data[idx]]
        numeralized_tags = [self.tag_vocab.vocab.stoi[t] for t in self.tags_data[idx]]
        return torch.tensor(numeralized_text), torch.tensor(numeralized_tags)


class WordEncoder(torch.nn.Module):
    def __init__(self, embeddings, pad_idx):
        super().__init__()
        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=pad_idx)

    def forward(self, x):
        return F.dropout(self.embeddings(x), 0.5)


class PosClassifier(torch.nn.Module):
    def __init__(self, insize, outsize):
        super().__init__()
        self.linear = nn.Linear(insize, outsize)

    def forward(self, x):
        output = self.linear(x)
        return output


class PosBiLSTM(torch.nn.Module):
    def __init__(self, outsize, embeddings, pad_idx, dropout=0.5, hidden_dim=256):
        super().__init__()
        self.embed = WordEncoder(embeddings, pad_idx)
        self.bi_lstm = nn.LSTM(embeddings.shape[1], hidden_dim//2, 2, batch_first=True, bidirectional=True, dropout=dropout)
        self.classifier = PosClassifier(hidden_dim, outsize)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.LSTM):
            for param_name,weights in module.named_parameters():
                if "weight_hh" in param_name:
                    torch.nn.init.eye_(weights)
                if "weight_ih" in param_name:
                    torch.nn.init.orthogonal_(weights)
                if "bias" in param_name:
                    torch.nn.init.constant_(weights, 0.5)
    
    def forward(self, x, s):
        x = self.embed(x)
        x_pack = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        output, _ = self.bi_lstm(x_pack)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = self.classifier(output)
        return output


def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
    return xx_pad, yy_pad, x_lens


def calc_acc(y, y_pred):
    y_pred = torch.argmax(y_pred, 1)
    no_pad_idxs = y.nonzero()
    y = y[no_pad_idxs]
    y_pred = y_pred[no_pad_idxs]
    correct = (y_pred == y).float().sum()
    total = y.shape[0]
    return correct/total


def train_model(model, train_loader, val_loader, tag_pad_idx, epochs=2000, lr=0.003):
    crit = torch.nn.CrossEntropyLoss(ignore_index=tag_pad_idx)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=0.00001)
    best_val_acc = 0
    for i in range(epochs):
        model.train()
        sum_loss = 0
        acc = 0
        for j, (x, y, l) in tqdm(enumerate(train_loader), total=len(train_loader)):
            x = x.to(dev)
            y = y.to(dev)

            y_pred = model(x, l)

            y = y.view(-1)
            y_pred = y_pred.view(-1, y_pred.shape[-1])

            loss = crit(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            acc += calc_acc(y, y_pred)
        val_loss, val_acc = validation_metrics(model, val_loader)
        if val_acc > best_val_acc:
            torch.save(model, "model.pth")
        logging.info("epoch %d train loss %.3f, train acc %.3f, val loss %.3f, val acc %.3f" % 
                    (i, sum_loss/len(train_loader), acc/len(train_loader), val_loss, val_acc))


def validation_metrics (model, loader):
    model.eval()
    acc = 0
    sum_loss = 0
    crit = torch.nn.CrossEntropyLoss()
    for i, (x, y, l) in enumerate(loader):
        x = x.to(dev)
        y= y.to(dev)
        y_hat = model(x, l)

        y = y.view(-1)
        y_hat = y_hat.view(-1, y_hat.shape[-1])

        loss = crit(y_hat, y)
        acc += calc_acc(y, y_hat)
        sum_loss += loss.item()

    return sum_loss/len(loader), acc/len(loader)


if __name__ == "__main__":
    main()