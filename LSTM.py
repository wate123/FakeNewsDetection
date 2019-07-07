import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Word2VecFeature import Word2VecFeatureGenerator
from sklearn.model_selection import train_test_split

from utils import NewsContent

from gensim.models.word2vec import LineSentence
import numpy as np
import pandas as pd

torch.manual_seed(1)
torch.cuda.is_available()


w2v = Word2VecFeatureGenerator()
vocab_size = len(w2v.model.wv.vocab)
w2v_weight = torch.FloatTensor(w2v.model.wv.vectors)
embedding = nn.Embedding.from_pretrained(w2v_weight)
EMBEDDING_DIM = 300
HIDDEN_DIM = 6


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(embedding.view(len(input), self.batch_size, -1))

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)

model = LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_size=6, num_layers=1)

loss_fn = torch.nn.MSELoss(size_average=False)

optimiser = torch.optim.Adam(model.parameters(), lr=0.2)

#####################
# Train model
#####################
num_epochs = 10


sample_train = pd.read_csv("w2v_feature.csv").sample(800)
labels = sample_train["label"]
sample_train = sample_train.drop('label', axis=1).values
X_train, X_test, y_train, y_test = train_test_split(sample_train, labels, test_size=0.2, random_state=1)

hist = np.zeros(num_epochs)

for t in range(num_epochs):
    # Clear stored gradient
    model.zero_grad()

    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    model.hidden = model.init_hidden()

    print(type(X_train))
    # Forward pass
    y_pred = model(X_train)

    loss = loss_fn(y_pred, y_train)
    if t % 100 == 0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    # Zero out gradient, else they will accumulate between epochs
    optimiser.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimiser.step()