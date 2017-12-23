import random
from time import time

import dynet as dy
import numpy as np

STUDENT = {"Name": "Tomer Gill", "ID": "318459450"}

GOOD = 0
BAD = 1


class LSTMNET:
    def __init__(self, num_layers, embed_dim, in_dim, hid_dim, out_dim, vocab_size):
        params = dy.ParameterCollection()
        self.__pc = params
        self.__builder = dy.LSTMBuilder(num_layers, embed_dim, in_dim, self.__pc)
        self.__E = params.add_lookup_parameters((vocab_size, embed_dim))
        self.__W1 = params.add_parameters((embed_dim, hid_dim))  # TODO SIZES
        self.__b1 = params.add_parameters(hid_dim)
        self.__W2 = params.add_parameters((hid_dim, out_dim))
        self.__b2 = params.add_parameters(out_dim)

    def __call__(self, inputs):
        dy.renew_cg()
        W1 = dy.parameter(self.__W1)
        W2 = dy.parameter(self.__W2)
        b1 = dy.parameter(self.__b1)
        b2 = dy.parameter(self.__b2)
        E = dy.parameter(self.__E)

        # Inserting the inputs to the LSTM
        s = self.__builder.initial_state()
        embeded = [E[i] for i in inputs]
        s.transduce(embeded)
        x = s.output()

        # Inserting the LSTM's output vector to the MLP1 and return the output
        h1 = dy.tanh(W1 * x + b1)
        return dy.softmax(W2 * h1 + b2)

    def compute_loss(self, inputs, expected_output):
        probs = self(inputs)
        return -dy.log(dy.pick(probs, expected_output))

    def predict(self, inputs):
        probs = self(inputs)
        return np.argmax(probs.npvalue())


def accuracy_on(net, word_and_tags):
    """

    :type net: LSTMNET
    :param net:
    :param word_and_tags:
    :return:
    """
    batch_preds = []
    tags = []
    for word, tag in word_and_tags:
        inputs = list(word)
        batch_preds.append(net(inputs))
        tags.append(GOOD if tag == "good" else BAD)
    dy.forward(batch_preds)
    good = 0.0
    for i, pred in enumerate(batch_preds):
        if np.argmax(pred.npvalue()) == tags[i]:
            good += 1
    return good / len(batch_preds)


def train_on(net, trainer, words_and_tags, epoches, acc_words_and_tags):
    """

    :type net: LSTMNET
    :param net:
    :param words_and_tags:
    :param epoches:
    :return:
    """
    start_time = time()
    copy = list(words_and_tags)
    for i in xrange(epoches):
        losses = []
        random.shuffle(copy)
        for word, tag in copy:
            inputs = list(word)
            loss = net.compute_loss(inputs, GOOD if tag == "good" else BAD)
            losses.append(loss)
        batch_loss = dy.esum(losses) / len(copy)
        avg_loss = batch_loss.value()
        batch_loss.backward()
        trainer.update()
        acc = accuracy_on(net, acc_words_and_tags)
        passed_time = time() - start_time
        print i, avg_loss, acc, passed_time
