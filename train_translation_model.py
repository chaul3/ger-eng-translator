import random
import time
from pathlib import Path
import torch.nn as nn
import torch
import torch.optim as optim

from helper_functions import plot_loss_curves, make_dictionary
from seq2seq_model import Encoder
torch.manual_seed(1)

# device
device = 'cpu'

# All hyperparameters
learning_rate = 0.3
number_of_epochs = 500
rnn_hidden_size = 512
mini_batch_size = 64
unk_threshold = 3
character_level = False
max_line_length = 150

# more interesting text
dataset = 'translations_5k'
dataset = 'translations_sanity_check'
# input and output filenames
input_filename = Path('data') / 'seq-2-seq' / f'{dataset}.txt'
model_save_name = Path('experiments')  / 'seq-2-seq' / f'{dataset}-model.pt'

# -- Step 1: Read training data and split into train, dev and test splits
training_data = []
with open(input_filename) as text_file:
    for line in text_file.readlines():

        source = line.split("\t")[0]
        target = line.split("\t")[-1]
        # default is loading the corpus as sequence of words
        training_data.append([source.lower().strip().split(" "),target.lower().strip().split(" ")])

# full corpus
corpus_size = len(training_data)

# corpus split into three splits
training_data = training_data[:-round(corpus_size / 5)]
validation_data = training_data[-round(corpus_size / 5):-round(corpus_size / 10)]
test_data = training_data[-round(corpus_size / 10):]

print(
    f"\nTraining corpus has {len(training_data)} train, {len(validation_data)} validation and {len(test_data)} test sentences")

all_source_sentences = [pair[0] for pair in training_data]
all_target_sentences = [pair[1] for pair in training_data]
#print(all_source_sentences)

source_dictionary = make_dictionary(all_source_sentences, unk_threshold=unk_threshold)
target_dictionary = make_dictionary(all_target_sentences, unk_threshold=unk_threshold)

print("training_data")
print(training_data)
# initialize translator and send to device
model = Encoder(source_dictionary=source_dictionary,
                target_dictionary=target_dictionary,
                embedding_size=256,
                rnn_hidden_size=rnn_hidden_size,  # hidden states in the LSTM
                is_character_level=character_level,
                device=device,
                )
print(model)

# --- Do a training loop

# define a simple SGD optimizer with a learning rate of 0.1
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# log the losses and accuracies
train_loss_per_epoch = []
validation_loss_per_epoch = []
validation_perplexity_per_epoch = []

# remember the best model
best_model = None
best_epoch = 0

# Go over the training dataset multiple times
for epoch in range(number_of_epochs):
    print(f"\n - Epoch {epoch}")

    start = time.time()

    # shuffle training data at each epoch
    random.shuffle(training_data)

    train_loss = 0.

    import more_itertools

    # each forward pass is now over a batch of sentences

    for sentences in more_itertools.chunked(training_data, mini_batch_size):
        # Step 4a. Remember that PyTorch accumulates gradients.

        # We need to clear them out before each instance
        model.zero_grad()
        #print(source_dictionary)
        # make one-hot inputs
        one_hot_inputs = model.make_onehot_vectors_source(sentences).T

        # make one-hot targets
        one_hot_targets = model.make_onehot_vectors_target(sentences).T

        #print(log_probabilities_for_each_class.size())
        # run forward decode
        log_probabilities_for_each_class = model.forward(one_hot_inputs,one_hot_targets, teacher_forcing_ratio=0.75)
        # calculate loss

        output_dim = log_probabilities_for_each_class.shape[-1]

        log_probabilities_for_each_class = log_probabilities_for_each_class.view(-1, output_dim)
        #one_hot_targets = one_hot_targets[1:].view(-1)


        loss = criterion(log_probabilities_for_each_class, one_hot_targets)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip=1)

        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(training_data)

    # Evaluate and print accuracy at end of each epoch
    validation_perplexity, validation_loss = model.evaluate(validation_data)

    # remember best model:
    if validation_perplexity < best_validation_perplexity:
        print(f"new best model found!")
        best_epoch = epoch
        best_validation_perplexity = validation_perplexity

        # always save best model
        torch.save(model, model_save_name)

    # print losses
    print(f"training loss: {train_loss}")
    print(f"validation loss: {validation_loss}")
    print(f"validation perplexity: {validation_perplexity}")

    # append to lists for later plots
    train_loss_per_epoch.append(train_loss)
    validation_loss_per_epoch.append(validation_loss)
    validation_perplexity_per_epoch.append(validation_perplexity)

    end = time.time()
    print(f'{round(end - start, 3)} seconds for this epoch')

# load best model and do final test
best_model = torch.load(model_save_name)
test_accuracy = best_model.evaluate(test_data)