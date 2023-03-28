# Basic Python imports
import random
import pickle
import os

# Imports from Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchtext import data
from torchtext import datasets
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

# Imports from NLTK
from nltk.tokenize import word_tokenize

# Imports for plotting
import matplotlib.pyplot as plt

# Imports for logging and display progress
import logging
from tqdm import tqdm
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# Specify random seed to allow reproducible results
random.seed(42)
torch.manual_seed(42)

# Utilize GPU if available
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Specify the directory which saved model live
model_dir = os.getcwd() + "/model"

def main():
    train_data = UDPOSTags("train")
    val_data = UDPOSTags("val", train_data.text_field, train_data.tag_field)
    test_data = UDPOSTags("test", train_data.text_field, train_data.tag_field)
    batch_size = 256

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)

    embeddings = train_data.text_field.vocab.vectors
    text_pad_idx = train_data.text_field.vocab.stoi[train_data.text_field.pad_token]
    tag_pad_idx = train_data.tag_field.vocab.stoi[train_data.tag_field.pad_token]

    model = PosBiLSTM(len(train_data.tag_field.vocab), embeddings, text_pad_idx)
    model.to(dev)
    best_model = PosBiLSTM(len(train_data.tag_field.vocab), embeddings, text_pad_idx)
    best_model_loss = float('inf')
    if os.path.exists(f"{model_dir}/best_model.pth"):
        best_model.load_state_dict(torch.load(f"{model_dir}/best_model.pth"))
        best_model.to(dev)
        best_model_loss, _ = validation_metrics(best_model, val_loader)
    logging.info("Best Model Loss %.3f" %(best_model_loss))
    _, test_acc = validation_metrics(best_model, test_loader) if best_model else (None, None)
    print(test_acc)
    plots = train_model(model, best_model_loss, train_loader, val_loader, tag_pad_idx, epochs=30, lr=0.0003)
    plot_loss(plots)


class UDPOSTags(Dataset):
    def __init__(self, split="train", text_field=None, tag_field=None):
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
        self.tag_data = [vars(e)["udtags"] for e in self.lang_data.examples]

        if text_field:
            self.text_field = text_field
        else:
            TEXT.build_vocab(self.lang_data, min_freq=2, vectors="glove.6B.300d", unk_init=torch.Tensor.normal_)
            self.text_field = TEXT
        if tag_field:
            self.tag_field = tag_field
        else:
            UD_TAGS.build_vocab(self.lang_data)
            self.tag_field = UD_TAGS

    def __len__(self):
        return len(self.tag_data)

    def __getitem__(self, idx):
        numeralized_text = [self.text_field.vocab.stoi[t] for t in self.text_data[idx]]
        numeralized_tags = [self.tag_field.vocab.stoi[t] for t in self.tag_data[idx]]
        return torch.tensor(numeralized_text), torch.tensor(numeralized_tags)


class WordEncoder(torch.nn.Module):
    def __init__(self, embeddings, pad_idx):
        super().__init__()
        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=pad_idx)

    def forward(self, x):
        return self.embeddings(x)


class PosClassifier(torch.nn.Module):
    def __init__(self, insize, outsize, dropout):
        super().__init__()
        self.linear = nn.Linear(insize, outsize)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = self.linear(self.dropout(x))
        return output


class PosBiLSTM(torch.nn.Module):
    def __init__(self, outsize, embeddings, pad_idx, dropout=0.25, hidden_dim=256):
        super().__init__()
        self.embed = WordEncoder(embeddings, pad_idx)
        self.bi_lstm = nn.LSTM(embeddings.shape[1], hidden_dim//2, 2, batch_first=True, bidirectional=True, dropout=dropout)
        self.classifier = PosClassifier(hidden_dim, outsize, dropout)
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


def calc_correct(y, y_pred):
    y_pred = torch.argmax(y_pred, 1)
    no_pad_idxs = y.nonzero()
    y = y[no_pad_idxs]
    y_pred = y_pred[no_pad_idxs]
    correct = (y_pred == y).float().sum()
    return correct, y.shape[0]


def train_model(model, best_model_loss, train_loader, val_loader, tag_pad_idx, epochs=2000, lr=0.003):
    crit = torch.nn.CrossEntropyLoss(ignore_index=tag_pad_idx)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr, amsgrad=True)
    best_val_loss = float('inf')
    plots = {'train':[], 'val':[]}
    for i in range(epochs):
        model.train()
        sum_loss = 0
        correct = 0
        total = 0
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

            batch_correct, batch_total = calc_correct(y, y_pred)
            correct += batch_correct
            total += batch_total
            sum_loss += loss.item()*batch_total


        val_loss, val_acc = validation_metrics(model, val_loader)
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            if best_model_loss:
                if best_val_loss <= best_model_loss:
                    logging.info("saving best val loss %.3f, best model loss %.3f" %(best_val_loss, best_model_loss))
                    save_model(model, train_loader.dataset.text_field, train_loader.dataset.tag_field)
            else:
                logging.info("saving best val loss %.3f" %(best_val_loss))
                save_model(model, train_loader.dataset.text_field, train_loader.dataset.tag_field)
        logging.info("epoch %d train loss %.3f, train acc %.3f, val loss %.3f, val acc %.3f" % 
                    (i, sum_loss/total, correct/total, val_loss, val_acc))
        plots['train'].append((i, sum_loss/total, correct/total))
        plots['val'].append((i, val_loss, val_acc))
    return plots


def validation_metrics(model, loader):
    model.eval()
    correct = 0
    sum_loss = 0
    total = 0
    crit = torch.nn.CrossEntropyLoss()
    for i, (x, y, l) in enumerate(loader):
        x = x.to(dev)
        y= y.to(dev)
        y_hat = model(x, l)

        y = y.view(-1)
        y_hat = y_hat.view(-1, y_hat.shape[-1])

        loss = crit(y_hat, y)

        batch_correct, batch_total = calc_correct(y, y_hat)
        correct += batch_correct
        total += batch_total
        sum_loss += loss.item()*batch_total

    return sum_loss/total, correct/total


def plot_loss(plots):
    epochs_train = [i[0] for i in plots['train']]
    losses_train = [i[1] for i in plots['train']]
    epochs_val = [i[0] for i in plots['val']]
    losses_val = [i[1] for i in plots['val']]
    plt.plot(epochs_train, losses_train, label="train")
    plt.plot(epochs_val, losses_val, label="val")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend
    plt.show()


def save_model(model, 
               text_field, 
               tag_field, 
               model_path='best_model.pth', 
               text_field_path='best_text_field.pkl', 
               tag_field_path='best_tag_field.pkl'):
    try:
        os.mkdir(model_dir)
    except OSError as error:
        pass
    text_field_out = open(f"{model_dir}/{text_field_path}", 'wb')
    tag_field_out = open(f"{model_dir}/{tag_field_path}", 'wb')
    pickle.dump(f"{model_dir}/{text_field}", text_field_out)
    pickle.dump(f"{model_dir}/{tag_field}", tag_field_out)
    torch.save(model.state_dict(), f"{model_dir}/{model_path}")


def tag_sentence(sentence, model, text_field, tag_field):
    tokens = word_tokenize(sentence.lower())
    tokens_torch = torch.Tensor([[text_field.vocab.stoi[token] for token in tokens]]).long()

    model.eval()
    tags = model(tokens_torch.cuda(), [len(tokens_torch[0])])
    tags = [tag_field.vocab.itos[tag] for tag in torch.argmax(tags,2)[0]]
    return (tokens, tags)


if __name__ == "__main__":
    main()