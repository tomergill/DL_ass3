import random as rn
from math import exp
from sys import argv


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


def generate_many(num, positive=True):
    """
    Generates many words using the generate_positive and generate_negative functions
    :param num: How many words to generate
    :param positive: If positive, generates positive examples. Otherwise, negative
    :return: List of num generated words
    """
    gen = generate_positive if positive else generate_negative
    return [gen() for _ in xrange(num)]


def __main():
    num = 500
    positive = True

    # Reading command line args
    if len(argv) == 3:
        if int(argv[1]) > 0:
            num = int(argv[1])
        if argv[2] in ("neg", "n", "negative"):
            positive = False

    words = generate_many(num, positive)
    for word in words:
        print word


if __name__ == "__main__":
    __main()
