import random
from time import time
import numpy as np
import dynet as dy
import part_3_model as mdl
from sys import argv
import itertools

UNKNOWN = "UUUNKKK"


def read_train_and_dev_data(train_file, dev_file):
    train_sentences = []
    sentence = []
    tags = []
    words_set = [UNKNOWN]
    tags_set = []
    dev_sentences = []

    count = 0

    for line in file(train_file):
        if line != "\n":
            word, tag = line[:-1].split()
            sentence.append(word)
            tags.append(tag)
        else:
            if dev_file is None and count == 9:  # every 10 sentences give one to the DEV
                dev_sentences.append((sentence, tags))
            else:
                if len(sentence) > 0:
                    train_sentences.append((sentence, tags))
                words_set.extend(sentence)
                tags_set.extend(tags)
            sentence = []
            tags = []
            count = (count + 1) % 10

    words_set, tags_set = set(words_set), set(tags_set)

    if dev_file is not None:
        for line in file(dev_file):
            if line != "\n":
                word, tag = line[:-1].split()
                sentence.append(word)
                tags.append(tag)
            else:
                dev_sentences.append((sentence, tags))
                sentence = []
                tags = []

    return train_sentences, dev_sentences, words_set, tags_set


def accuracy_on(net, dev_set, ignored_tag=-1):
    """
    Using the net, predicts a tag for each word in each sentence and then compares it to the tags.

    :type net: mdl.AbstractNet
    :param net: Neural network
    :param dev_set: A list of tuples in the form of (sentence, tags) - sentence is a list of
    words, and tags is a list of the tags of those words.
    :param ignored_tag: Tag to ignore if good match
    :return: number of good predictions divided by the total number of words
    """
    total = good = 0.0
    random.shuffle(dev_set)
    sentences, tags = zip(*dev_set)
    # predictions = net.predcit_batch(sentences)
    predictions = [net.predict(sentence) for sentence in sentences]
    for i, preds in enumerate(predictions):
        total += len(tags[i])
        good += np.sum(np.equal(np.array([p if p != ignored_tag or tags[i] != ignored_tag else -1
                                          for j, p in enumerate(preds)]), np.array(tags[i])))
    return good / total


def train_on(net, trainer, train_set, dev_set, epoches, ignored_tag=-1):
    """


    :type net: mdl.AbstractNet
    :param net:
    :param trainer:
    :param train_set:
    :param dev_set:
    :param epoches:
    :param ignored_tag:
    :return:
    """
    copy = list(train_set)
    total_words = reduce(lambda x, y: x + len(y), train_set, 0.0)
    batch_size = 500
    print "Iteration,Sentences/100,Average Loss,Epoch Time,DEV Accuracy"
    dev_copy = list(dev_set)
    count_sentences = 0
    for i in xrange(epoches):
        total_loss = 0.0
        start_time = time()
        random.shuffle(copy)
        for start in range(len(copy) / batch_size + 1):
            batch = copy[start*batch_size: (start+1)*batch_size]
            # batch_losses = net.loss_on_batch(batch)
            # total_loss += np.sum(batch_losses.npvalue())
            # batch_losses.backward()
            # trainer.update()
            count_sentences += len(batch) / 100

            for sentence, tags in batch:
                losses = net.get_loss(sentence, tags)
                loss = dy.esum(losses)
                total_loss += loss.value()
                loss.backward()
                trainer.update()

            acc = accuracy_on(net, dev_copy, ignored_tag)
            avg_loss = total_loss / total_words
            epoch_time = time() - start_time
            print "{:9d},{:13d},{:12f},{:9f}s,{:11f}%".format(i, count_sentences, avg_loss,
                                                              epoch_time, 100 * acc)


def main():
    # PARAMETERS #
    k = 3
    num_layers = 1
    embed_dim = 64
    lstm1_dim = 32
    in_dim = 32
    epoches = 5

    # PROGRAM ARGUMENTS #
    length = len(argv)
    if length < 4:
        print "Error - not enough arguments for program"
        exit(1)

    repr_choice = argv[1]
    train_file = argv[2]
    model_file = argv[3]
    dev_file = None
    ignore_tag = None

    if length >= 5:
        dev_file = argv[4]

    if length >= 6 and len(argv[5]) == 1:
        ignore_tag = argv[5]
    elif "ner" in train_file or "NER" in train_file:
        ignore_tag = "O"

    # READ DATA #
    train_data, dev_data, words_set, tags_set = read_train_and_dev_data(train_file, dev_file)

    if repr_choice != 'b':  # a, b, c
        I2W = sorted(list(words_set))
        W2I = {w: i for i, w in enumerate(I2W)}

    if repr_choice == 'b' or repr_choice == 'd':  # b, d
        I2C = reduce(lambda l1, l2: l1 + l2, [list(word) for word in words_set])
        I2C = sorted(list(set(I2C)))
        # I2C = [chr(i) for i in range(0, 256)]
        C2I = {c: i for i, c in enumerate(I2C)}

    if repr_choice == 'c':  # c
        P2I = {w[:k]: i for i, w in enumerate(I2W)}
        S2I = {w[-k:]: i for i, w in enumerate(I2W)}
        WI2PI = [P2I[w[:k]] for w in I2W]  # word index to prefix index
        WI2SI = [S2I[w[-k:]] for w in I2W]  # likewise, with suffix instead

    I2T = sorted(list(tags_set))
    T2I = {t: i for i, t in enumerate(I2T)}

    # CREATE NET #
    if repr_choice == 'a':
        net = mdl.WordEmbeddedNet(num_layers, embed_dim, lstm1_dim, in_dim, len(T2I), len(W2I))
        # List of words' indexes and tags' indexes
        train = [([W2I[word] for word in sentence], [T2I[tag] for tag in tags])
                 for sentence, tags in train_data]
        dev = [([W2I[word] if word in words_set else W2I[UNKNOWN] for word in sentence],
                [T2I[tag] for tag in tags]) for sentence, tags in dev_data]

    elif repr_choice == 'b':
        net = mdl.CharEmbeddedLSTMNet(num_layers, embed_dim, lstm1_dim, in_dim, len(T2I), len(C2I))
        # list of each word's chars' indexes and tags' indexes
        train = [([[C2I[c] for c in list(word)] for word in sentence], [T2I[tag] for tag in tags])
                 for sentence, tags in train_data]
        dev = [([[C2I[c] for c in list(word)] for word in sentence], [T2I[tag] for tag in tags])
               for sentence, tags in dev_data]
    elif repr_choice == 'c':
        net = mdl.WordAndSubwordEmbeddedNet(num_layers, embed_dim, lstm1_dim, in_dim, len(T2I),
                                            len(W2I), WI2PI, WI2SI)
        # List of words' indexes and tags' indexes
        train = [([W2I[word] for word in sentence], [T2I[tag] for tag in tags])
                 for sentence, tags in train_data]
        dev = [([W2I[word] if word in words_set else W2I[UNKNOWN] for word in sentence],
                [T2I[tag] for tag in tags]) for sentence, tags in dev_data]
    else:
        net = mdl.WordEmbeddedAndCharEmbeddedLSTMNet(num_layers, embed_dim, lstm1_dim, in_dim,
                                                     len(T2I), len(C2I), len(W2I))
        # list of tuples of (words' indexes, word's chars' indexes) and tags' indexes
        train = [([(W2I[word], [C2I[c] for c in list(word)]) for word in sentence],
                  [T2I[tag] for tag in tags]) for sentence, tags in train_data]
        dev = [([(W2I[word] if word in words_set else W2I[UNKNOWN],
                  [C2I[c] for c in list(word)]) for word in sentence],
                [T2I[tag] for tag in tags]) for sentence, tags in dev_data]
    trainer = dy.AdamTrainer(net.pc)

    print "sentences in train: {}, sentences in dev: {}".format(len(train), len(dev))
    print "Parameters: file = {}, k = {}, num_layers = {}, embed_dim = {}, lstm1_dim = {}, " \
          "in_dim = {}, epoches = {}\n".format(train_file, k, num_layers, embed_dim, lstm1_dim,
                                               in_dim, epoches)

    # TRAIN #
    if ignore_tag is None:
        train_on(net, trainer, train, dev, epoches)
    else:
        train_on(net, trainer, train, dev, epoches, ignored_tag=ignore_tag)

    # SAVE NET #
    if repr_choice == "a":
        mdl.save_net_and_params_to(net, model_file, num_layers, embed_dim, lstm1_dim, in_dim,
                                   len(I2T), I2T, vocab_size=len(W2I), I2W=I2W,
                                   unknown_word_index=W2I[UNKNOWN])
    elif repr_choice == "b":
        mdl.save_net_and_params_to(net, model_file, num_layers, embed_dim, lstm1_dim, in_dim,
                                   len(I2T), I2T, I2C=I2C, char_vocab_size=len(I2C))
    elif repr_choice == "c":
        mdl.save_net_and_params_to(net, model_file, num_layers, embed_dim, lstm1_dim, in_dim,
                                   len(I2T), I2T, vocab_size=len(W2I), I2W=I2W,
                                   unknown_word_index=W2I[UNKNOWN], word_to_pre_index=WI2PI,
                                   word_to_suf_index=WI2SI)
    else:  # d
        mdl.save_net_and_params_to(net, model_file, num_layers, embed_dim, lstm1_dim, in_dim,
                                   len(I2T), I2T, vocab_size=len(W2I), I2W=I2W,
                                   unknown_word_index=W2I[UNKNOWN], I2C=I2C,
                                   char_vocab_size=len(I2C))

    # ############ TODO DELET TIS
    # print "\nTEST:\n"
    # sentence = []
    # for line in file("pos/temp"):
    #     if line != "\n":
    #         word, tag = line[:-1].split()
    #         sentence.append(word)
    #     else:
    #         if len(sentence) > 0:
    #             predictions = net.predict([W2I[w] for w in sentence])
    #             for word, prediction in itertools.izip(sentence, predictions):
    #                 print "{}\t{}".format(word, I2T[prediction])
    #         print "\n******\n"
    #         sentence = []

    #######################################################
    import os
    duration = 1  # second
    freq = 440  # Hz
    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
    #######################################################


if __name__ == "__main__":
    main()
