import random
import dynet as dy
import experiment as model

VOCAB = [str(idx) for idx in xrange(10)] \
        + [chr(c) for c in xrange(ord('a'), ord('z'))] \
        + [chr(c) for c in xrange(ord('A'), ord('Z'))]


def generate_palindrome(max_len=50):
    length = random.randint(1, max_len)
    s = "".join([VOCAB[random.randint(0, len(VOCAB) - 1)] for _ in xrange(length)])
    if random.randint(0, 1) == 1:  # odd length palindrome
        return s + VOCAB[random.randint(0, len(VOCAB) - 1)] + s[::-1]
    else:  # even length palindrome
        return s + s[::-1]


def generate_bad(max_len=None, max_change_number=10):
    if max_len is None:
        pal = generate_palindrome()
    else:
        pal = generate_palindrome(max_len)

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
    words = []
    for _ in xrange(half_size):
        while True:
            good = generate_palindrome(max_len)
            if (good, "good") not in not_in:
                break

        while True:
            bad = generate_bad(max_len, max_change_number)
            if (bad, "bad") not in not_in:
                break

        if tag:
            words.append((good, "good"))
            words.append((bad, "bad"))
        else:
            words.append(good)
            words.append(bad)
    return words


def is_palindrome(p):
    halflen = len(p) / 2
    fhalf = p[:halflen]
    if len(p) % 2 == 0:
        shalf = p[halflen:]
    else:
        shalf = p[halflen + 1:]
    return fhalf == shalf[::-1]


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
    model.train_on(net, trainer, train, epoches, dev, char2int)

    # Predict on Test
    print "\nWrong Predictions on TEST:"
    predictions = model.predict_on(net, test, char2int)
    good_preds = 0.0
    output = open("pali.pred", "w")
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
    print "###########\t Accuracy on pred is {}%".format(good_preds / len(test) * 100)
