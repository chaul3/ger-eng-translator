import math
from textwrap import wrap
from typing import Dict

import matplotlib.pyplot as plt


def make_dictionary(data, unk_threshold: int = 0) -> Dict[str, int]:
    '''
    Makes a dictionary of words given a list of tokenized sentences.
    :param data: List of (sentence, label) tuples
    :param unk_threshold: All words below this count threshold are excluded from dictionary and replaced with UNK
    :return: A dictionary of string keys and index values
    '''

    # First count the frequency of each distinct ngram
    word_frequencies = {}
    for sent in data:
        for word in sent:
            if word not in word_frequencies:
                word_frequencies[word] = 0
            word_frequencies[word] += 1

    # Assign indices to each distinct ngram
    word_to_ix = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in word_frequencies.items():
        if freq > unk_threshold:  # only add words that are above threshold
            word_to_ix[word] = len(word_to_ix)

    # Print some info on dictionary size
    print(f"At unk_threshold={unk_threshold}, the dictionary contains {len(word_to_ix)} words")
    return word_to_ix


def plot_loss_curves(loss_train,
                     loss_val,
                     accuracy_val,
                     approach_name: str,
                     hyperparams,
                     validation_label='Validation accuracy'):
    last_finished_epoch = len(loss_train)
    epochs = range(1, last_finished_epoch + 1)
    hyperparam_pairs = [f"{key}{hyperparams[key]}" for key in hyperparams]

    file_name = f"experiments/loss-curves-{approach_name}-" + "-".join(hyperparam_pairs).replace("/", "-") + ".png"
    title_text = ", ".join([f"{key}:{hyperparams[key]}" for key in hyperparams])

    fig, ax1 = plt.subplots()

    color = 'g'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, loss_train, 'r', label='Training loss')
    ax1.plot(epochs, loss_val, 'g', label='Validation loss')
    ax1.tick_params(axis='y', labelcolor=color)
    title = ax1.set_title("\n".join(wrap(title_text, 60)))
    title.set_y(1.05)
    fig.subplots_adjust(top=0.8)
    ax1.grid()

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'k'  # k := black
    ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot(epochs, accuracy_val, 'black', label=validation_label)
    ax2.tick_params(axis='y', labelcolor=color)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.xticks(range(5, math.floor((last_finished_epoch + 1) / 5) * 5, 5))
    plt.savefig(file_name)
    plt.show()
