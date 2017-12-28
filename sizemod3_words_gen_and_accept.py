import random
import dynet as dy
import experiment as model

VOCAB = [str(idx) for idx in xrange(10)]


# def generate_good(max_len=10):
#     num = random.randint(1, 10 ** max_len)
#     return str(3 * num)
#
#
# def generate_bad(max_len=None, max_change_number=10):
#     good = int(generate_good())
#     d = random.randint(1, 2)
#     if random.randint(0, 1) == 1:
#         d = -d
#     return str(good + d)
#
#
# def generate_word_list(half_size, tag=True, not_in=None, max_len=50,
#                        max_change_number=10):
#     not_in = [] if not_in is None else not_in
#     good_words = set()
#     bad_words = set()
#     while len(good_words) != half_size and len(bad_words) != half_size:
#         while True:
#             good = generate_good(max_len)
#             if (good, "good") not in not_in:
#                 break
#
#         while True:
#             bad = generate_bad(max_len, max_change_number)
#             if (bad, "bad") not in not_in:
#                 break
#
#         if tag:
#             good_words.add((good, "good"))
#             bad_words.add((bad, "bad"))
#         else:
#             good_words.add(good)
#             bad_words.add(bad)
#     return list(good_words) + list(bad_words)


def is_good(p):
    return int(p) % 3 == 0


if __name__ == "__main__":
    # create train, dev & test set
    m3e0 = range(3, 3001, 3)
    m3e2 = range(2, 3001, 3)
    m3e1 = range(1, 3001, 3)

    tr0 = random.sample(m3e0, 500)
    tr1 = random.sample(m3e1, 250)
    tr2 = random.sample(m3e2, 250)

    train = [(str(i), "good") for i in tr0] + \
            [(str(i), "bad") for i in tr1] + \
            [(str(i), "bad") for i in tr2]

    m3e0 = [i for i in m3e0 if i not in tr0]
    m3e1 = [i for i in m3e1 if i not in tr1]
    m3e2 = [i for i in m3e2 if i not in tr2]

    dev0 = random.sample(m3e0, 250)
    dev1 = random.sample(m3e1, 125)
    dev2 = random.sample(m3e2, 125)

    dev = [(str(i), "good") for i in dev0] + \
          [(str(i), "bad") for i in dev1] + \
          [(str(i), "bad") for i in dev2]

    m3e0 = [i for i in m3e0 if i not in dev0]
    m3e1 = [i for i in m3e1 if i not in dev1]
    m3e2 = [i for i in m3e2 if i not in dev2]

    te0 = random.sample(m3e0, 100)
    te1 = random.sample(m3e1, 50)
    te2 = random.sample(m3e2, 50)

    test = [str(i) for i in te0 + te1 + te2]

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
    output = open("mod3.pred", "w")
    for idx, word in enumerate(test):
        string = "{}\t{}".format(word, predictions[idx])

        pali = is_good(word)
        pred = predictions[idx]
        if (pali and pred == "good") or (not pali and pred == "bad"):
            good_preds += 1.0
        else:
            print idx, string

        # print string
        output.write(string + "\n")
    output.close()
    print "\n###########\t Accuracy on pred is {}%".format(good_preds / len(test) * 100)
