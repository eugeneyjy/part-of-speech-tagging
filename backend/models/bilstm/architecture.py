import pickle
import torch
import torch.nn as nn
from torchtext import data
from torchtext import datasets
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
    
def from_pretrained(path):
    text_field = pickle.load(open(f"{path}/text_field.pkl", "rb"))
    tag_field = pickle.load(open(f"{path}/tag_field.pkl", "rb"))
    model = PosBiLSTM(len(tag_field.vocab), text_field.vocab.vectors, text_field.vocab.stoi[text_field.pad_token])
    model.load_state_dict(torch.load(f"{path}/model.pth"))

    return model, text_field, tag_field