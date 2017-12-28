import random
import dynet as dy
import experiment as model

VOCAB = [str(idx) for idx in xrange(10)] \
        + [chr(c) for c in xrange(ord('a'), ord('z'))] \
        + [chr(c) for c in xrange(ord('A'), ord('Z'))]


def generate_double(max_len=50):
    length = random.randint(1, max_len)
    w = "".join([VOCAB[random.randint(0, len(VOCAB) - 1)] for _ in xrange(length)])
    return 2 * w


def generate_bad(max_len=None, max_change_number=10):
    if max_len is None:
        pal = generate_double()
    else:
        pal = generate_double(max_len)

    opal = pal

    if max_change_number > len(pal):
        max_change_number = len(pal) / 2

    for _ in xrange(max_change_number):
        i = random.randint(0, len(pal) - 1)
        pal = pal[:i] + VOCAB[random.randint(0, len(VOCAB) - 1)] + pal[i+1:]

    if pal == opal:
        return generate_bad(max_len, max_change_number + 2)
    return pal


def generate_word_list(half_size, tag=True, not_in=None, max_len=50,
                       max_change_number=10):
    not_in = [] if not_in is None else not_in
    good_words = set()
    bad_words = set()
    while len(good_words) != half_size and len(bad_words) != half_size:
        while True:
            good = generate_double(max_len)
            if (good, "good") not in not_in:
                break

        while True:
            bad = generate_bad(max_len, max_change_number)
            if (bad, "bad") not in not_in:
                break

        if tag:
            good_words.add((good, "good"))
            bad_words.add((bad, "bad"))
        else:
            good_words.add(good)
            bad_words.add(bad)
    return list(good_words) + list(bad_words)


def is_palindrome(p):
    half = len(p) /2
    return p[:half] == p[half:]


if __name__ == "__main__":
    # create train, dev & test set
    train = generate_word_list(500)
    dev = generate_word_list(250, not_in=train)
    test = generate_word_list(100, not_in=train+dev, tag=False)

    int2char = VOCAB
    char2int = {c: index for index, c in enumerate(int2char)}

    num_layers = 1
    embed_dim = 50
    in_dim = 150
    hid_dim = 100
    out_dim = 2

    net = model.LSTMNET(num_layers, embed_dim, in_dim, hid_dim, out_dim, len(VOCAB))
    trainer = dy.AdamTrainer(net.pc)

    # Train and check accuracy each iteration
    epoches = 15
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
    model.train_on(net, trainer, train, epoches, dev, char2int)

    # Predict on Test
    print "\nWrong Predictions on TEST:"
    predictions = model.predict_on(net, test, char2int)
    good_preds = 0.0
    output = open("double.pred", "w")
    for idx, word in enumerate(test):
        string = "{}\t{}".format(word, predictions[idx])

        pali = is_palindrome(word)
        pred = predictions[idx]
        if (pali and pred == "good") or (not pali and pred == "bad"):
            good_preds += 1.0
        else:
            print idx, string

        # print string
        output.write(string + "\n")
    output.close()
    print "\n###########\t Accuracy on pred is {}%".format(good_preds / len(test) * 100)
