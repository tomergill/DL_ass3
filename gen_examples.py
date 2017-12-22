import random as rn
from math import exp
from sys import argv

STUDENT = {"Name": "Tomer Gill", "ID": "318459450"}


def generate_positive(maxdigits=10):
    """
    Generates a positive example of our language
    :param maxdigits Maximal number of digits
    :return: string in the form of [1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+
    """
    digits = [rn.randint(0, maxdigits) for _ in range(5)]
    rand = [rn.randint(int(exp(digit)), int(exp(digit + 1) - 1)) for digit in digits]
    return "{0}a{1}b{2}c{3}d{4}".format(str(rand[0]), str(rand[1]), str(rand[2]), str(rand[3]),
                                        str(rand[4]))


def generate_negative(maxdigits=10):
    """
    Generates a negative example of our language
    :param maxdigits Maximal number of digits
    :return: string in the form of [1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+
    """
    digits = [rn.randint(0, maxdigits) for _ in range(5)]
    rand = [rn.randint(int(exp(digit)), int(exp(digit + 1) - 1)) for digit in digits]
    return "{0}a{1}c{2}b{3}d{4}".format(str(rand[0]), str(rand[1]), str(rand[2]), str(rand[3]),
                                        str(rand[4]))


def generate_many(num, positive=True, without=[]):
    """
    Generates many words using the generate_positive and generate_negative functions
    :param without: If a generated word is already in this, generate again
    :param num: How many words to generate
    :param positive: If positive, generates positive examples. Otherwise, negative
    :return: List of num generated words
    """
    gen = generate_positive if positive else generate_negative
    lst = []
    for _ in xrange(num):
        word = gen()
        while word in without:
            word = gen()
        lst.append(word)
    return lst


def __main():
    num = 500
    positive = True

    # Reading command line args
    if len(argv) == 3:
        if int(argv[1]) > 0:
            num = int(argv[1])
        if argv[2] in ("neg", "n", "negative"):
            positive = False

    ex_type = "pos" if positive else "neg"

    train_path, dev_path, test_path = ex_type + "_examples", ex_type + "_dev", ex_type + "_test"

    TRAIN = generate_many(num, positive)
    train_file = open(train_path, "w")
    for word in TRAIN:
        train_file.write(word + "\n")
    train_file.close()

    DEV = generate_many(num / 2, positive, without=TRAIN)
    dev_file = open(dev_path, "w")
    for word in DEV:
        dev_file.write(word + "\n")
    dev_file.close()

    TEST = generate_many(num / 10, positive, TRAIN + DEV) + generate_many(num / 10, not positive)
    rn.shuffle(TEST)
    test_file = open(test_path, "w")
    for word in TEST:
        test_file.write(word + "\n")
    test_file.close()



if __name__ == "__main__":
    __main()
