from tokeniser import sentence_tokeniser, tokeniser, clean_text_neural
import torch
from collections import Counter
from torch import nn
from torch.utils.data import DataLoader
import argparse
import re
import math
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
    ):
        self.args = args
        self.words = self.get_words()
        self.unique_words = self.get_unique_words()

        self.index_to_word = self.get_index_to_word()
        self.word_to_index = self.get_word_to_index()

        self.words_indexes = self.word_indices()
        self.size = len(self.unique_words)

    def get_words(self):
        f = open("corpus/Ulysses - James Joyce.txt", "r")
        text = f.read()
        f.close()
        data = sentence_tokeniser(text)
        text = ""

        for line in data:
            # tokens = tokenize(line)
            words = " ".join(tokeniser(line))
            text += "<start> " + words + " <end> "
        return clean_text_neural(text.split(' '))

    def get_unique_words(self):
        count = Counter(self.words)
        sorted_words = sorted(count, key=count.get, reverse=True)
        return sorted_words

    def get_word_to_index(self):
        word_to_index = {}
        for i, word in enumerate(self.unique_words):
            word_to_index[word] = i
        return word_to_index

    def get_index_to_word(self):
        index_to_word = {}
        for i, word in enumerate(self.unique_words):
            index_to_word[i] = word
        return index_to_word

    def word_indices(self):
        word_indices = []
        for word in self.words:
            word_indices.append(self.word_to_index[word])
        return word_indices

    def __len__(self):
        total = len(self.words_indexes)
        return total - self.args.sequence_length

    def __getitem__(self, index):
        context = torch.tensor(self.words_indexes[index:index+self.args.sequence_length]).to(device)
        target = torch.tensor(self.words_indexes[index+1:index+self.args.sequence_length+1]).to(device)
        return (context, target)


class Model(nn.Module):
    def __init__(self, dataset, hidden_size, embedding_dim, num_layers):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.vocab_size = dataset.size
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, self.vocab_size)
    
    def init_state(self, sequence_length):
        context = torch.zeros(
            self.num_layers, sequence_length, self.hidden_size).to(device)
        target = torch.zeros(self.num_layers, sequence_length,
                             self.hidden_size).to(device)
        return (context, target)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.linear(output)
        return logits, state

def train(dataset, model, args):
    # Early stopping
    last_loss = 100

    model.train()

    learning_rate = 0.01
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(args.max_epochs):
        hidden_state, cell_state = model.init_state(args.sequence_length)

        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()

            y_pred, (hidden_state, cell_state) = model(x, (hidden_state, cell_state))
            loss = criterion(y_pred.transpose(1, 2), y)

            hidden_state, cell_state = hidden_state.detach(), cell_state.detach()
            
            loss.backward()
            optimizer.step()

            if batch % 100 == 0 or batch == len(dataloader):
                print('[Epoch: {}, Batch: {}] loss: {:.5}'.format(epoch,
                      batch, loss.item()))

            last_loss = loss.item()

    # Save model
    torch.save(model.state_dict(), 'model2.pt')


def get_probability(dataset, model, text):
    model.eval()

    text = tokeniser(text)
    words = text
    # print(words)
    for i in range(len(words)):
        if words[i] not in dataset.unique_words:
            words[i] = "<unk>"
    prob = 1
    for i in range(1, len(words)):
        
        x = torch.tensor([[dataset.word_to_index[w]
                         for w in words[:i]]]).to(device)

        state_h, state_c = model.init_state(i)
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        # print(last_word_logits)
        # p = torch.nn.functional.softmax(
        #     last_word_logits, dim=0).cpu().detach().numpy()
        # word_index = dataset.word_to_index[words[i]]
        # print(dataset.index_to_word[word_index], p[word_index])
        prob *= torch.nn.functional.softmax(
            last_word_logits, dim=0).cpu().detach().numpy()[dataset.word_to_index[words[i]]]

    return prob


def get_perplexity(dataset, model, text):
    return math.pow(get_probability(dataset, model, text), -1/len(text.split()))


def perp_to_file():
    File_object = open("train.txt", "r")
    data = File_object.read()
    File_object.close()

    data = data.split("\n")
    
    sum = 0
    perps = []
    count = 0
    print(data)
    for line in data:
        try:
            x = get_perplexity(dataset, model, line)
            sum += x
            perps.append(line + "\t" + str(x) + "\n")
            count += 1
            if count % 100 == 0:
                print(count)
        except:
            continue
    
    with open("perps.txt", "w") as f:
        f.writelines(perps)

    print("Sum: {}, Count: {}".format(sum, count))
    print(sum/count)


parser = argparse.ArgumentParser()
parser.add_argument('--max-epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--sequence-length', type=int, default=4)
parser.add_argument('--dataset', type=str,
                    default="corpus/Ulysses - James Joyce.txt")
args = parser.parse_args()

full_dataset = args.dataset

# Create dataset
f = open(full_dataset, "r")
data = f.read()
f.close()

data = sentence_tokeniser(data)

random.shuffle(data)

train_data = data[:int(len(data)*0.8)]
test_data = data[int(len(data)*0.8):]

f = open("train.txt", "w")
f.writelines(train_data)
f.close()

f= open("test.txt", "w")
f.writelines(test_data)
f.close()

args.dataset = "train.txt"

dataset = Dataset(args)
model = Model(dataset, 128, 100, 1)
model.to(device)

# train(dataset, model, args)

# input_path = sys.argv[1]

# Load saved model

model.load_state_dict(torch.load('model.pt'))

sentence = input("Input Sentence: ")
print(get_probability(dataset, model, sentence))
# perp_to_file()
