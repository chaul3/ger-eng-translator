# TODO: implement!
import math
import random
from typing import Dict

import torch
import torch.nn.functional as F


class Encoder(torch.nn.Module):  # inherits from nn.Module!

    def __init__(self,
                 source_dictionary: Dict[str, int],
                 target_dictionary: Dict[str, int],
                 embedding_size: int,
                 rnn_hidden_size: int,
                 is_character_level: bool = False,
                 device: str = 'cpu',
                 ):

        super(Encoder, self).__init__()

        # remember device, vocabulary and character-level
        self.device = device
        self.is_character_level = is_character_level
        self.source_dictionary = source_dictionary
        self.target_dictionary = target_dictionary

        # add special symbols to source_dictionary
        if '<START>' not in source_dictionary:
            source_dictionary['<START>'] = len(source_dictionary)

        if '<STOP>' not in source_dictionary:
            source_dictionary['<STOP>'] = len(source_dictionary)

        if '<PAD>' not in source_dictionary:
            source_dictionary['<PAD>'] = len(source_dictionary)

        # add special symbols to source_dictionary
        if '<START>' not in target_dictionary:
            target_dictionary['<START>'] = len(target_dictionary)

        if '<STOP>' not in target_dictionary:
            target_dictionary['<STOP>'] = len(target_dictionary)

        if '<SEP>' not in target_dictionary:
            target_dictionary['<SEP>'] = len(target_dictionary)

        # One-hot embeddings of words (or characters)
        self.embedding_source = torch.nn.Embedding(len(source_dictionary), embedding_size)
        self.embedding_target = torch.nn.Embedding(len(target_dictionary), embedding_size)

        # The LSTM takes embeddings as inputs, and outputs hidden states of dimensionality hidden dim
        self.lstm_source = torch.nn.LSTM(self.embedding_source.embedding_dim,
                                         rnn_hidden_size,
                                         batch_first=False,
                                         num_layers=1


                                        )

        # The LSTM takes embeddings as inputs, and outputs hidden states of dimensionality hidden dim
        self.lstm_target = torch.nn.LSTM(
                                         self.embedding_target.embedding_dim,

                                         rnn_hidden_size,
                                         batch_first=False,
                                         num_layers=1
                                         )

        # The hidden2tag linear layer takes LSTM output and projects to tag space
        self.hidden2tag = torch.nn.Linear(rnn_hidden_size, len(target_dictionary))

    def make_onehot_vectors_target(self, sentences):
        onehot_mini_batch = []

        sentences=[sentence[1] for sentence in sentences]

        onehot_mini_batch = []

        lengths = [len(sentence) for sentence in sentences]
        lengths_plus = [len(sentence) + 1 for sentence in sentences]
        longest_sequence_in_batch = max(lengths)

        for sentence in sentences:

            onehot_for_sentence = [self.target_dictionary["<START>"]]

            # move a window over the text
            for word in sentence:

                # look up ngram index in dictionary
                if word in self.target_dictionary:
                    onehot_for_sentence.append(self.target_dictionary[word])
                else:
                    onehot_for_sentence.append(self.target_dictionary["<UNK>"] if "<UNK>" in self.target_dictionary else 0)

            # append a STOP index
            onehot_for_sentence.append(self.target_dictionary["<STOP>"])

            # fill the rest with PAD indices
            for i in range(longest_sequence_in_batch - len(sentence)):
                onehot_for_sentence.append(self.target_dictionary["<PAD>"])

            onehot_mini_batch.append(onehot_for_sentence)

        onehot_mini_batch = torch.tensor(onehot_mini_batch, device=self.device)
        return onehot_mini_batch

    def make_onehot_vectors_source(self, sentences):
        onehot_mini_batch = []

        sentences=[sentence[0] for sentence in sentences]

        onehot_mini_batch = []

        lengths = [len(sentence) for sentence in sentences]
        lengths_plus = [len(sentence) + 1 for sentence in sentences]
        longest_sequence_in_batch = max(lengths)

        for sentence in sentences:

            onehot_for_sentence = [self.source_dictionary["<START>"]]

            # move a window over the text
            for word in sentence:

                # look up ngram index in dictionary
                if word in self.source_dictionary:
                    onehot_for_sentence.append(self.source_dictionary[word])
                else:
                    onehot_for_sentence.append(self.source_dictionary["<UNK>"] if "<UNK>" in self.source_dictionary else 0)

            # append a STOP index


            # fill the rest with PAD indices
            for i in range(longest_sequence_in_batch - len(sentence)):
                onehot_for_sentence.append(self.source_dictionary["<PAD>"])

            onehot_mini_batch.append(onehot_for_sentence)

        onehot_mini_batch = torch.tensor(onehot_mini_batch, device=self.device)

        return onehot_mini_batch

    def calculate_loss(self, log_probabilities_for_each_class, targets):

        # flatten all predictions and targets for the whole mini-batch into one long list
        flattened_log_probabilities_for_each_class = log_probabilities_for_each_class.flatten(end_dim=1)
        flattened_targets = targets.flatten()

        # Optional: Use loss masking
        loss_mask = torch.ones(len(self.source_dictionary), device=self.device)
        loss_mask[self.source_dictionary['<PAD>']] = 0.

        # calculate loss
        loss = torch.nn.functional.nll_loss(
            input=flattened_log_probabilities_for_each_class,
            target=flattened_targets,
            weight=loss_mask
        )
        return loss

    def forward_encode(self, one_hot_sentences, hidden=None):
        # embed the sentences
        embeds = self.embedding_source(one_hot_sentences)

        # send through LSTM
        lstm_out, (hidden, cell) = self.lstm_source(embeds, hidden)

        return lstm_out, hidden, cell

    def forward_decode(self, one_hot_sentence, hidden,cell):
        #trg_len=one_hot_sentence.shape[0]
        #batch=one_hot_sentence.shape[1]
        input_trg = one_hot_sentence.unsqueeze(0)
        #input_trg = [one_hot_sentence]
        print(input_trg)
        embeds = self.embedding_target(input_trg)
        #print(cell.size())
        # embeds = self.embedding_target(input_trg)
        print(embeds.size(), (hidden.size(), cell.size()))

        lstm_out, (hidden, cell) = self.lstm_target(embeds, (hidden, cell))

        features = self.hidden2tag(lstm_out)

        prediction = F.log_softmax(features, dim=2)

        return prediction, hidden, cell

    def forward(self, one_hot_sentences_input,one_hot_sentences_target, teacher_forcing_ratio = 0.5):

        batch_size = one_hot_sentences_target.shape[1]
        trg_len = one_hot_sentences_target.shape[0]
        trg_vocab_size = len(self.target_dictionary)

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        lstm_out, hidden, cell = self.forward_encode(one_hot_sentences_input)

        # first input to the decoder is the <sos> tokens
        input_encode = one_hot_sentences_target[0, :]

        for t in range(1,trg_len):
            # embed the sentences

            output, hidden, cell = self.forward_decode(input_encode, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input_encode = one_hot_sentences_target[t] if teacher_force else top1

        # multiply all encoder outputs with current output of decoder to get all attention scores
        attention_scores = torch.matmul(lstm_out.squeeze(), output.squeeze())
        # softmax attention scores to get distribution
        attention_distribution = torch.nn.functional.softmax(attention_scores, dim=0)
        # do a weighted sum to get attention output
        weighted = lstm_out * attention_distribution.unsqueeze(1)
        attention_output = torch.sum(weighted, axis=1).unsqueeze(dim=0)
        # concat attention output to decoder rnn output and send through linear layer
        logits = self.hidden2tag(torch.cat((output, attention_output), 2))
        return logits

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
