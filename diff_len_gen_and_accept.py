import random
import dynet as dy
import experiment as model

VOCAB = ['a', 'b']


def get_other(c):
    return VOCAB[1] if c == VOCAB[0] else VOCAB[0]


def generate_good(max_len=50):
    i, j = 0, 0
    while i == j:
        i, j = random.randint(0, max_len), random.randint(0, max_len)
    return i * 'a' + j * 'b'


def generate_bad(max_len=None, max_change_number=10):
    # action = random.randint(-9,10)
    # if action <= 0:
    k = random.randint(1, max_len)
    return k * 'a' + k * 'b'
    # else:
    #     g = generate_good() if max_len == None else generate_good(max_len)
    #     j = random.randint(1, max_change_number)
    #     for _ in xrange(j):
    #         while is_good(g):
    #             i = random.randint(0, len(g) - 1)
    #             g = g[:i] + get_other(g[i]) + g[i+1:]
    #     return g


def generate_word_list(half_size, tag=True, not_in=None, max_len=50,
                       max_change_number=10):
    not_in = [] if not_in is None else not_in
    words = []
    for _ in xrange(half_size):
        while True:
            good = generate_good(max_len)
            if (good, "good") not in not_in:
                break

        while True:
            bad = generate_bad(max_len, max_change_number)
            if (bad, "bad") not in not_in:
                break

        words.append((good, "good"))
        words.append((bad, "bad"))

    return words if tag else [w for w, t in words]


def is_good(s):
    if s == "":
        return False
    i = 0
    while i < len(s) and s[i] == 'a':
        i += 1
    j = 0
    while i + j < len(s) and s[i + j] == 'b':
        j += 1
    if len(s) == i + j:
        return i != j
    return False



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

        isgood = is_good(word)
        pred = predictions[idx]
        if (isgood and pred == "good") or (not isgood and pred == "bad"):
            good_preds += 1.0
        else:
            print idx, string

        # print string
        output.write(string + "\n")
    output.close()
    print "\n###########\t Accuracy on pred is {}%".format(good_preds / len(test) * 100)
