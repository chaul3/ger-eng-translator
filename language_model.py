import math
from typing import Dict

import torch
import torch.nn.functional as F


class LanguageModel(torch.nn.Module):  # inherits from nn.Module!

    def __init__(self,
                 vocabulary: Dict[str, int],
                 embedding_size: int,
                 rnn_hidden_size: int,
                 is_character_level: bool = False,
                 device: str = 'cpu',
                 ):

        super(LanguageModel, self).__init__()

        # remember device, vocabulary and character-level
        self.device = device
        self.is_character_level = is_character_level
        self.vocabulary = vocabulary

        # add special symbols to vocabulary
        if '<START>' not in vocabulary:
            vocabulary['<START>'] = len(vocabulary)
        if '<STOP>' not in vocabulary:
            vocabulary['<STOP>'] = len(vocabulary)

        # One-hot embeddings of words (or characters)
        self.embedding = torch.nn.Embedding(len(vocabulary), embedding_size)

        # The LSTM takes embeddings as inputs, and outputs hidden states of dimensionality hidden dim
        self.lstm = torch.nn.LSTM(self.embedding.embedding_dim,
                                  rnn_hidden_size,
                                  batch_first=True,
                                  num_layers=1,
                                  )

        # The hidden2tag linear layer takes LSTM output and projects to tag space
        self.hidden2tag = torch.nn.Linear(rnn_hidden_size, len(vocabulary))

    def make_onehot_vectors(self, sentences, as_targets: bool = False):
        onehot_mini_batch = []

        lengths = [len(sentence) for sentence in sentences]
        lengths_plus = [len(sentence) + 1 for sentence in sentences]
        longest_sequence_in_batch = max(lengths)

        for sentence in sentences:

            onehot_for_sentence = [self.vocabulary["<START>"]]

            # move a window over the text
            for word in sentence:

                # look up ngram index in dictionary
                if word in self.vocabulary:
                    onehot_for_sentence.append(self.vocabulary[word])
                else:
                    onehot_for_sentence.append(self.vocabulary["<UNK>"] if "<UNK>" in self.vocabulary else 0)

            # append a STOP index
            onehot_for_sentence.append(self.vocabulary["<STOP>"])

            # fill the rest with PAD indices
            for i in range(longest_sequence_in_batch - len(sentence)):
                onehot_for_sentence.append(self.vocabulary["<PAD>"])

            onehot_mini_batch.append(onehot_for_sentence)

        onehot_mini_batch = torch.tensor(onehot_mini_batch, device=self.device)
        #print(onehot_mini_batch)
        if as_targets:
            return onehot_mini_batch[:, 1:]
        else:
            return onehot_mini_batch[:, :-1]

    def calculate_loss(self, log_probabilities_for_each_class, targets):

        # flatten all predictions and targets for the whole mini-batch into one long list
        flattened_log_probabilities_for_each_class = log_probabilities_for_each_class.flatten(end_dim=1)
        flattened_targets = targets.flatten()

        # Optional: Use loss masking
        loss_mask = torch.ones(len(self.vocabulary), device=self.device)
        loss_mask[self.vocabulary['<PAD>']] = 0.

        # calculate loss
        loss = torch.nn.functional.nll_loss(
            input=flattened_log_probabilities_for_each_class,
            target=flattened_targets,
            weight=loss_mask
        )
        return loss

    def forward(self, one_hot_sentences, hidden=None):

        # embed the sentences
        embeds = self.embedding(one_hot_sentences)

        # send through LSTM
        lstm_out, hidden = self.lstm(embeds, hidden)

        # project into tag space
        features = self.hidden2tag(lstm_out)

        # then pass that through log_softmax
        return F.log_softmax(features, dim=2), hidden

    def generate_text(self, prefix: str = None, max_symbols: int = 100, stop_symbol='<STOP>', temperature=1.0):

        original_prefix = prefix

        if not prefix:
            prefix = []
        else:
            prefix = [char for char in prefix] if self.is_character_level else prefix.split(" ")

        prefix.insert(0, "<START>")

        inv_map = {v: k for k, v in self.vocabulary.items()}

        input = torch.tensor([[self.vocabulary[symbol] for symbol in prefix]], device=self.device)

        hidden = None
        generated_words = []
        for i in range(max_symbols):
            output, hidden = self.forward(input, hidden)
            word_weights = output[:, -1, :].squeeze().div(temperature).exp().cpu()
            word_weights[self.vocabulary["<UNK>"]] = 0 # a hack to set UNK to zero so we don't generate those
            word_idx = torch.multinomial(word_weights, 1)[0]
            input = torch.tensor([[word_idx]], device=self.device)

            if stop_symbol and word_idx == self.vocabulary[stop_symbol]: break

            generated_words.append(inv_map[word_idx.item()])

        separator = '' if self.is_character_level else ' '
        if original_prefix:
            print(f"'{original_prefix}' continues as '{separator.join(generated_words)}'")
        else:
            print(f"Generating sample text: {separator.join(generated_words)}")

    def evaluate(self, test_data):

        with torch.no_grad():
            # first generate some sample text
            self.generate_text()

            # then evaluate
            aggregate_loss = 0.

            # go through all test data points
            for sentence in test_data:
                # make one-hot inputs
                one_hot_inputs = self.make_onehot_vectors([sentence], as_targets=False)

                # Step 4c. Run our forward pass.
                log_probabilities_for_each_class, hidden = self.forward(one_hot_inputs)

                # make one-hot targets
                one_hot_targets = self.make_onehot_vectors([sentence], as_targets=True)

                # calculate loss
                aggregate_loss += self.calculate_loss(log_probabilities_for_each_class, one_hot_targets)

            aggregate_loss = aggregate_loss / len(test_data)

            return math.exp(aggregate_loss), aggregate_loss
