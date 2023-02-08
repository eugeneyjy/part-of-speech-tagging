import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.legacy import data
from torchtext.legacy import datasets
from torchtext.vocab import GloVe
import string
from collections import Counter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from nltk.tokenize import word_tokenize

def main():
    udpos_train = UDPOSTags(split="train")
    batch_size = 1
    udpos_train_loader = DataLoader(udpos_train, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)

class Vocabulary:

    def __init__(self, corpus):
        self.word2idx, self.idx2word, self.freq = self.build_vocab(corpus)
        self.size = len(self.word2idx)

    def text2idx(self, text):
        if type(text) == list:
            tokens = text 
        else:
            tokens = self.tokenize(text)
        return [self.word2idx[t] if t in self.word2idx.keys() else self.word2idx['UNK'] for t in tokens]

    def idx2text(self, idxs):
        return [self.idx2word[i] if i in self.idx2word.keys() else 'UNK' for i in idxs]

    def tokenize(self, text):
        tokens = text.lower()
        tokens = word_tokenize(tokens)
        return tokens

    def build_vocab(self, corpus):
        cntr = Counter()

        if type(corpus[0]) == list:
            for text in corpus:
                cntr.update(text)
        else:
            for text in corpus:
                tokens = self.tokenize(text)
                cntr.update(tokens)
        
        freq = {t:c for t, c in cntr.items()}
        include_tokens = [t for t, c in cntr.items() if c >= 10]
        word2idx = {t:i+1 for i, t in enumerate(include_tokens)}
        idx2word = {i+1:t for i, t in enumerate(include_tokens)}
        word2idx['UNK'] = len(include_tokens) + 1
        idx2word[len(include_tokens)+1] = 'UNK'
        word2idx[''] = 0
        idx2word[0] = ''

        return word2idx, idx2word, freq


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
            self.text_vocab = Vocabulary(self.text_data)
        if tag_vocab:
            self.tag_vocab = tag_vocab
        else:
            self.tag_vocab = Vocabulary(self.tags_data)

    def __len__(self):
        return len(self.tags_data)

    def __getitem__(self, idx):
        numeralized_text = self.text_vocab.text2idx(self.text_data[idx])
        numeralized_tags = self.tag_vocab.text2idx(self.tags_data[idx])
        return torch.tensor(numeralized_text), torch.tensor(numeralized_tags)

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
    return xx_pad, yy_pad, x_lens

if __name__ == "__main__":
    main()