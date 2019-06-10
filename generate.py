import torch
import torch.nn.functional
import numpy as np


train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')


def _pick_word(probabilities, temperature):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :return: String of the predicted word
    """
    probabilities = np.log(probabilities) / temperature
    exp_probs = np.exp(probabilities)
    probabilities = exp_probs / np.sum(exp_probs)
    pick = np.random.choice(len(probabilities), p=probabilities)
    return pick


def generate(model, start_seq, vocab, length=100, temperature=1.0):
    model.eval()

    tokens = vocab.clean_text(start_seq)
    tokens = vocab.tokenize(tokens)

    # create a sequence (batch_size=1) with the prime_id
    current_seq = np.full((1, model.seq_length), vocab['<pad>'])
    for idx, token in enumerate(tokens):
        current_seq[-1][idx - len(tokens)] = vocab[token]
    predicted = tokens

    for _ in range(length):
        if train_on_gpu:
            current_seq = torch.LongTensor(current_seq).cuda()
        else:
            current_seq = torch.LongTensor(current_seq)

        hidden = model.init_hidden(current_seq.size(0))

        output, _ = model(current_seq, hidden)

        p = torch.nn.functional.softmax(output, dim=1).data
        if train_on_gpu:
            p = p.cpu()

        probabilities = p.numpy().squeeze()

        word_i = _pick_word(probabilities, temperature)

        # retrieve that word from the dictionary
        word = vocab[int(word_i)]
        predicted.append(word)

        # the generated word becomes the next "current sequence" and the cycle can continue
        current_seq = current_seq.cpu().data.numpy()
        current_seq = np.roll(current_seq, -1, 1)
        current_seq[-1][-1] = word_i

    gen_sentences = ' '.join(predicted)

    gen_sentences = vocab.add_punctuation(gen_sentences)

    return gen_sentences
