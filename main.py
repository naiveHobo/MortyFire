import os
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from data_generator import RickAndMortyData
from vocabulary import Vocabulary
from model import MortyFire
from generate import generate

ap = ArgumentParser()

ap.add_argument("--mode", required=True, type=str, help="train|generate",
                choices=["train", "generate"])
ap.add_argument("--vocab_path", default=None, type=str, help="path to load vocabulary from")
ap.add_argument("--model_path", default='mortyfire.model', type=str, help="path to load trained model from")
ap.add_argument("--checkpoint_dir", default='checkpoints/', type=str, help="path to save checkpoints")
ap.add_argument("--script_len", default=200, type=int, help="length of script")
ap.add_argument("--temperature", default=1.0, type=float, help="diversity in script generated")
ap.add_argument("--start", default='rick', type=str, help="starting word of script")

args = ap.parse_args()

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')

data_path = 'data/rick_and_morty.txt'

epochs = 10
batch_size = 128
lstm_size = 128
seq_length = 64
num_layers = 2
bidirectional = True
embeddings_size = 300
dropout = 0.5
learning_rate = 0.001

with open(data_path, 'r') as f:
    text = f.read()

vocab = Vocabulary()

if args.vocab_path is None:
    vocab.add_text(text)
    vocab.save('data/vocab.pkl')
else:
    vocab.load(args.load_vocab)

print(vocab)

model = MortyFire(vocab_size=len(vocab), lstm_size=lstm_size, embed_size=embeddings_size, seq_length=seq_length,
                  num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, train_on_gpu=train_on_gpu)

if train_on_gpu:
    model.cuda()

print(model)

if args.mode == "train":

    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    dataset = RickAndMortyData(text=text, seq_length=seq_length, vocab=vocab)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    writer = SummaryWriter()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    losses = []
    batch_losses = []
    global_step = 0

    print("\nInitializing training...")
    for epoch in range(1, epochs + 1):
        print("Epoch: {:>4}/{:<4}".format(epoch, epochs))

        model.train()
        hidden = model.init_hidden(batch_size)

        for batch, (inputs, labels) in enumerate(data_loader):

            labels = labels.reshape(-1)

            if labels.size()[0] != batch_size:
                break

            h = tuple([each.data for each in hidden])
            model.zero_grad()

            if train_on_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            output, h = model(inputs, h)

            loss = criterion(output, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            hidden = h

            losses.append(loss.item())
            batch_losses.append(loss.item())

            if batch % 10 == 0:
                print("step [{}/{}]\t loss: {:4f}".format(batch, len(dataset) // batch_size, np.average(batch_losses)))
                writer.add_scalar('loss', loss, global_step)
                batch_losses = []

            global_step += 1

        print("\n----- Generating text -----")
        for temperature in [0.2, 0.5, 1.0]:
            print('----- Temperatue: {} -----'.format(temperature))
            print(generate(model, start_seq=args.start, vocab=vocab, temperature=temperature, length=100))
            print()

        torch.save(model.state_dict(),
                   os.path.join(args.checkpoint_dir, "mortyfire-{}-{:04f}.model".format(epoch, np.average(losses))))
        epoch_losses = []

    writer.close()

    print("\nSaving model [{}]".format(args.model_path))
    torch.save(model.state_dict(), args.model_path)

else:
    print("\nLoading model from [{}]".format(args.model_path))
    model.load_state_dict(torch.load(args.model_path))
    script = generate(model, start_seq=args.start, vocab=vocab, temperature=args.temperature, length=args.script_len)
    print()
    print('----- Temperatue: {} -----'.format(args.temperature))
    print(script)
    print()
