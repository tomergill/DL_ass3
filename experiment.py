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
        self.pc = params
        self.__builder = dy.LSTMBuilder(num_layers, embed_dim, in_dim, self.pc)
        self.__E = params.add_lookup_parameters((vocab_size, embed_dim))
        self.__W1 = params.add_parameters((hid_dim, in_dim))
        self.__b1 = params.add_parameters(hid_dim)
        self.__W2 = params.add_parameters((out_dim, hid_dim))
        self.__b2 = params.add_parameters(out_dim)

    def __call__(self, inputs):
        dy.renew_cg()
        W1 = dy.parameter(self.__W1)
        W2 = dy.parameter(self.__W2)
        b1 = dy.parameter(self.__b1)
        b2 = dy.parameter(self.__b2)
        # E = dy.parameter(self.__E)

        # Inserting the inputs to the LSTM
        s = self.__builder.initial_state()
        embeded = [dy.lookup(self.__E, i) for i in inputs]
        outputs = s.transduce(embeded)
        x = outputs[-1]

        # Inserting the LSTM's output vector to the MLP1 and return the output
        h1 = dy.tanh(W1 * x + b1)
        return dy.softmax(W2 * h1 + b2)

    def compute_loss(self, inputs, expected_output):
        probs = self(inputs)
        return -dy.log(dy.pick(probs, expected_output))

    def predict(self, inputs):
        probs = self(inputs)
        return np.argmax(probs.npvalue())


def accuracy_on(net, word_and_tags, char2int):
    """

    :param char2int:
    :type net: LSTMNET
    :param net:
    :param word_and_tags:
    :return:
    """
    word_and_tags = list(word_and_tags)  # make a copy
    random.shuffle(word_and_tags)
    good = 0.0
    for word, tag in word_and_tags:
        inputs = [char2int[c] for c in list(word)]
        output = net(inputs)
        t = GOOD if tag == "good" else BAD
        if np.argmax(output.npvalue()) == t:
            good += 1
    return good / len(word_and_tags)


def train_on(net, trainer, words_and_tags, epoches, acc_words_and_tags, char2int):
    """

    :param char2int:
    :param trainer:
    :type net: LSTMNET
    :param net:
    :param words_and_tags:
    :param epoches:
    :param acc_words_and_tags:
    :return:
    """
    print "+----+--------+----------------+----------+"
    print "| it |  loss  | total_time (s) | dev_acc  |"
    print "+----+--------+----------------+----------+"
    start_time = time()
    copy = list(words_and_tags)
    for i in xrange(epoches):
        tot_loss = 0.0
        random.shuffle(copy)
        for word, tag in copy:
            inputs = [char2int[c] for c in list(word)]
            loss = net.compute_loss(inputs, GOOD if tag == "good" else BAD)
            tot_loss += loss.value()
            loss.backward()
            trainer.update()
        avg_loss = tot_loss / len(copy)
        acc = accuracy_on(net, acc_words_and_tags, char2int)
        passed_time = time() - start_time
        print "| %2d | %1.4f | %14.5f | %6.2f %% |" % (i, avg_loss, passed_time, acc * 100)
        print "+----+--------+----------------+----------+"


def read_words_file(path, tag=None):
    """
    
    :type tag: str
    :param path: 
    :param tag: 
    :return: 
    """
    return [(line[:-1] if tag is None else (line[:-1], tag))
            for line in file(path) if line != "\n"]


def predict_on(net, words, char2int):
    preds = []
    for word in words:
        inputs = [char2int[c] for c in list(word)]
        out = net(inputs)
        preds.append("good" if np.argmax(out.npvalue()) == GOOD else "bad")
    return preds


def main():
    vocab = [str(i) for i in range(10)] + list("abcd")
    int2char = vocab
    char2int = {c: i for i, c in enumerate(int2char)}

    num_layers = 1
    embed_dim = 50
    in_dim = 150
    hid_dim = 100
    out_dim = 2

    net = LSTMNET(num_layers, embed_dim, in_dim, hid_dim, out_dim, len(vocab))
    trainer = dy.AdamTrainer(net.pc)

    train_file, dev_file, test_file = "_examples", "_dev", "_test"

    train = read_words_file("pos" + train_file, "good") + read_words_file("neg" + train_file, "bad")
    dev = read_words_file("pos" + dev_file, "good") + read_words_file("neg" + dev_file, "bad")
    test = read_words_file("pos" + test_file) + read_words_file("neg" + test_file)

    # Train and check accuracy each iteration
    epoches = 5
    print "######################################################"
    print "Run parameters:"
    print "*\tLSTM layers: %d" % num_layers
    print "*\tEmbedding layer size: %d" % embed_dim
    print "*\tMLP1's input layer size: %d" % in_dim
    print "*\tMLP1's hidden layer size: %d" % hid_dim
    print "*\tTrain set size: %d" % len(train)
    print "*\tDev set size: %d" % len(dev)
    print "*\tTest set size: %d" % len(test)
    print "######################################################"
    print "\nTraining:"
    train_on(net, trainer, train, epoches, dev, char2int)

    # Predict on Test
    print "\nPredicting on TEST:"
    predictions = predict_on(net, test, char2int)
    output = open("test1.pred", "w")
    for i, word in enumerate(test):
        string = "{:54}\t{}".format(word, predictions[i])
        print string
        output.write(string + "\n")
    output.close()


if __name__ == "__main__":
    main()
