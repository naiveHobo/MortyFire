import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from vocabulary import Vocabulary


class RickAndMortyData(Dataset):

    def __init__(self, text, seq_length, vocab=None):
        self.text = text
        self.seq_length = seq_length
        if vocab is None:
            self.vocab = Vocabulary()
            self.vocab.add_text(self.text)
        else:
            self.vocab = vocab
        self.text = self.vocab.clean_text(text)
        self.tokens = self.vocab.tokenize(self.text)

    def __len__(self):
        return len(self.tokens) - self.seq_length

    def __getitem__(self, idx):
        x = [self.vocab[word] for word in self.tokens[idx:idx + self.seq_length]]
        y = [self.vocab[self.tokens[idx + self.seq_length]]]
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        return x, y


if __name__ == '__main__':
    with open('data/rick_and_morty.txt', 'r') as f:
        text = f.read()
    data = RickAndMortyData(text=text, seq_length=20)
    data_loader = DataLoader(data, batch_size=128, shuffle=True)
    data_iter = iter(data_loader)
    x, y = data_iter.next()
    print(x.size())
    print(x)
    print()
    y = y.reshape(-1)
    print(y.size())
    print(y)
