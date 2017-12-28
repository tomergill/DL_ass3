import random
import dynet as dy
import experiment as model

VOCAB = ['a', 'b', 'c', 'd']


def generate_abcdmnsum(max_len=25):
    n, m = 0, 0
    while m == 0 and n == 0:
        n, m = random.randint(0, max_len), random.randint(0, max_len)
    return (m+n) * 'a' + n * 'b' + m * 'c' + (m+n) * 'd'


def generate_bad(max_len=None, max_change_number=10):
    if max_len is None:
        abcd = generate_abcdmnsum()
    else:
        abcd = generate_abcdmnsum(max_len)

    oabcd = abcd

    if max_change_number > len(abcd):
        max_change_number = len(abcd) / 2

    for _ in xrange(max_change_number):
        i = random.randint(0, len(abcd) - 1)
        action = random.randint(0,2)
        if action == 0:  # replace a char
            abcd = abcd[:i] + VOCAB[random.randint(0, len(VOCAB) - 1)] + abcd[i+1:]
        elif action == 1:  # remove a char
            abcd = abcd[:i] + abcd[i+1:]
        else:  # add a char
            abcd = abcd[:i+1] + VOCAB[random.randint(0, len(VOCAB) - 1)] + abcd[i + 1:]

    if abcd == oabcd or abcd == "":
        return generate_bad(max_len, max_change_number + 2)
    return abcd


def generate_word_list(half_size, tag=True, not_in=None, max_len=50,
                       max_change_number=10):
    not_in = [] if not_in is None else not_in
    words = []
    for _ in xrange(half_size):
        while True:
            good = generate_abcdmnsum(max_len)
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


def is_good(s):
    if s == "":
        return False
    plus = 0
    while s[plus] == 'a':
        plus += 1
    if s[plus] == 'd':
        return False
    n = 0
    while s[plus +n] == 'b':
        n += 1
    m = 0
    while s[plus + n + m] == 'c':
        m += 1
    if s[plus + n+m] == 'a' or s[plus + n+m] == 'b' or len(s[plus + m+n:]) != m + n or plus != m+n:
        return False
    for c in s[plus + m+n:]:
        if c != 'd':
            return False
    return True


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
    output = open("abcmnsum.pred", "w")
    for idx, word in enumerate(test):
        string = "{}\t{}".format(word, predictions[idx])

        is_abcd = is_good(word)
        pred = predictions[idx]
        if (is_abcd and pred == "good") or (not is_abcd and pred == "bad"):
            good_preds += 1.0
        else:
            print idx, string

        # print string
        output.write(string + "\n")
    output.close()
    print "\n###########\t Accuracy on pred is {}%".format(good_preds / len(test) * 100)
