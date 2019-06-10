import torch.nn as nn


class MortyFire(nn.Module):

    def __init__(self, vocab_size, embed_size, lstm_size, seq_length, num_layers, dropout=0.5, bidirectional=False,
                 train_on_gpu=True):
        nn.Module.__init__(self)

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.lstm_size = lstm_size
        self.seq_length = seq_length
        self.embed_size = embed_size
        self.train_on_gpu = train_on_gpu
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, lstm_size, num_layers, dropout=dropout, batch_first=True,
                            bidirectional=bidirectional)
        self.fc = nn.Linear(lstm_size * 2, vocab_size)

    def forward(self, batch, hidden):
        batch_size = batch.size(0)
        embeds = self.embedding(batch)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.lstm_size * 2)
        output = self.fc(lstm_out)
        output = output.view(batch_size, -1, self.vocab_size)
        out = output[:, -1]
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        layers = self.num_layers if not self.bidirectional else self.num_layers * 2
        if self.train_on_gpu:
            hidden = (weight.new(layers, batch_size, self.lstm_size).zero_().cuda(),
                      weight.new(layers, batch_size, self.lstm_size).zero_().cuda())
        else:
            hidden = (weight.new(layers, batch_size, self.lstm_size).zero_(),
                      weight.new(layers, batch_size, self.lstm_size).zero_())
        return hidden
