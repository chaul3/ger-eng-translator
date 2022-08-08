import random
import time
from pathlib import Path

import torch
import torch.optim as optim

from helper_functions import plot_loss_curves, make_dictionary
from language_model import LanguageModel

torch.manual_seed(1)

# device (could be 'cpu', 'cuda:0', 'cuda:1', etc.)
device = 'cuda:0'
device= 'cpu'

#dataset = 'motivational_quotes'
# dataset = 'equations'
#
dataset = 'sanity_check'

# input and output filenames
input_filename = Path('data') / 'language_modeling' / f'{dataset}.txt'
model_save_name = Path('experiments')  / 'language_modeling' / f'{dataset}-model.pt'

# All hyperparameters
learning_rate = 0.3
number_of_epochs = 500
rnn_hidden_size = 512
mini_batch_size = 64
unk_threshold = 3
character_level = False
max_line_length = 150

print(f"Training language model with \n - rnn_hidden_size: {rnn_hidden_size}\n - learning rate: {learning_rate}"
      f" \n - max_epochs: {number_of_epochs} \n - mini_batch_size: {number_of_epochs} \n - unk_threshold: {unk_threshold}")

# -- Step 1: Get a small amount of training data
training_data = []
with open(input_filename) as text_file:
    for line in text_file.readlines():

        # skip long lines
        if len(line) > max_line_length:
            continue

        if character_level:
            training_data.append([char for char in line])
        else:
            # default is loading the corpus as sequence of words
            training_data.append(line.lower().strip().split(" "))

corpus_size = len(training_data)

training_data = training_data[:-round(corpus_size / 5)]
validation_data = training_data[-round(corpus_size / 5):-round(corpus_size / 10)]
test_data = training_data[-round(corpus_size / 10):]
print(
    f"\nTraining corpus has {len(training_data)} train, {len(validation_data)} validation and {len(test_data)} test sentences")
#print(training_data)
word_dictionary = make_dictionary(training_data, unk_threshold=unk_threshold)
#print(word_dictionary)
# -- Step 3: Make word dictionary and init classifier

# --- Step b: Initialize the classifier with pre-trained embeddings
model = LanguageModel(vocabulary=word_dictionary,
                      embedding_size=100,
                      rnn_hidden_size=rnn_hidden_size,  # hidden states in the LSTM
                      is_character_level=character_level,
                      device=device,
                      )
model.to(device)
print(model)

# --- Step 4: Do a training loop
# define the loss and a simple SGD optimizer with a learning rate of 0.1
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# log the losses and accuracies
train_loss_per_epoch = []
validation_loss_per_epoch = []
validation_perplexity_per_epoch = []

# remember the best model
best_model = None
best_epoch = 0
best_validation_perplexity = 100000.

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

        # make one-hot inputs
        one_hot_inputs = model.make_onehot_vectors(sentences, as_targets=False)

        # Step 4c. Run our forward pass.
        log_probabilities_for_each_class, hidden = model.forward(one_hot_inputs)

        # make one-hot targets
        one_hot_targets = model.make_onehot_vectors(sentences, as_targets=True)

        # calculate loss
        loss = model.calculate_loss(log_probabilities_for_each_class, one_hot_targets)

        train_loss += loss.item()

        loss.backward()
        optimizer.step()

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

# print final score
print("\n -- Training Done --")
print(f" - using model from epoch {best_epoch} for final evaluation")
print(f" - final score: {test_accuracy}")

# make plots
hyperparams = {"rnn_hidden_size": rnn_hidden_size, "lr": learning_rate}
approach_name = "Language Model"
plot_loss_curves(train_loss_per_epoch, validation_loss_per_epoch, validation_perplexity_per_epoch,
                 approach_name=approach_name,
                 validation_label='Validation perplexity',
                 hyperparams=hyperparams)

# load the model again
language_model = torch.load(model_save_name)

# if dataset is sanity_check, it should generate "this is not a love song"
if dataset == 'sanity_check':
    language_model.generate_text(prefix="this is")

# if dataset is equations, it should generate actual math
if dataset == 'equations':
    language_model.generate_text(prefix="one plus one equals")
    language_model.generate_text(prefix="two plus two equals")
    language_model.generate_text(prefix="three plus three equals")
    language_model.generate_text(prefix="four plus four equals")
    language_model.generate_text(prefix="five plus five equals")

# if dataset is motivational_quotes, it should generate ... motivation quotes?
if dataset == 'motivational_quotes':
    language_model.generate_text(prefix="life is")
    language_model.generate_text(prefix="marriage is")
    language_model.generate_text(prefix="luck is")
