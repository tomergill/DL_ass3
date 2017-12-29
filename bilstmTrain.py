import part_3_model as mdl
from sys import argv

UNKNOWN = "UUUNKKK"


def read_train_and_dev_data(train_file, dev_file):
    train_sentences = []
    sentence = []
    tags = []
    words_set = {UNKNOWN}
    tags_set = set()
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
                words_set = words_set.union(set(sentence))
                tags_set = tags_set.union(set(tags))
            sentence = []
            tags = []
            count = (count + 1) % 10

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


def main():
    k = 3
    num_layers = 1
    embed_dim = 50
    lstm1_dim = 150
    in_dim = 200

    length = len(argv)
    if length < 4:
        print "Error - not enough arguments for program"
        exit(1)

    repr_choice = argv[1]
    train_file = argv[2]
    model_file = argv[3]
    dev_file = None
    if length == 5:
        dev_file = argv[4]

    # READ DATA #
    train, dev, words_set, tags_set = read_train_and_dev_data(train_file, dev_file)

    if repr_choice != 'b':  # a, b, c
        I2W = sorted(list(words_set))
        W2I = {w: i for i, w in enumerate(I2W)}

    if repr_choice == 'b' or repr_choice == 'd':  # b, d
        I2C = reduce(lambda l1, l2: l1 + l2, [list(word) for word in words_set])
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
    elif repr_choice == 'b':
        net = mdl.CharEmbeddedLSTMNet(num_layers, embed_dim, lstm1_dim, in_dim, len(T2I), len(C2I))
    elif repr_choice == 'c':
        net = mdl.WordAndSubwordEmbeddedNet(num_layers, embed_dim, lstm1_dim, in_dim, len(T2I),
                                            len(W2I), WI2PI, WI2SI)
    else:
        net = mdl.WordEmbeddedAndCharEmbeddedLSTMNet(num_layers, embed_dim, lstm1_dim, in_dim,
                                                     len(T2I), len(C2I), len(W2I))


if __name__ == "__main__":
    main()
