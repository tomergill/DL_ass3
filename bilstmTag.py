import itertools
import part_3_model as mdl
from sys import argv


def make_sentences(file_name):
    sentences = []
    sentence = []
    for line in file(file_name):
        if line != "\n":
            sentence.append(line[-1])
        else:
            sentences.append(sentence)
            sentence = []
    return sentences


def main():
    if argv < 4:
        print "Usage error - not enough arguments"
    repr_choice = argv[1]
    model_file = argv[2]
    input_file = argv[3]

    loaded = mdl.load_net_and_params_from(model_file)
    net, I2T, unk_index = loaded[0], loaded[1], loaded[-1]
    if repr_choice != "b":
        I2W = loaded[2]
        unkown = I2W[unk_index]
        W2I = {w: i for i, w in enumerate(I2W)}
    else:
        I2C = loaded[2]
        C2I = {c: i for i, c in enumerate(I2C)}
    if repr_choice == "c":
        I2C = loaded[3]

    sentences = make_sentences(input_file)
    if repr_choice != "b":
        words = [[W2I[word] for word in sentence] for sentence in sentences]
    if repr_choice == "b" or repr_choice == "d":
        chars = [[[C2I[c] for c in list(word)] for word in sentence] for sentence in sentences]

    if repr_choice == "a" or repr_choice == "c":
        inp = words
    elif repr_choice == "b":
        inp = chars
    else:
        inp = [zip(sentence, characters) for sentence, characters in itertools.izip(words, chars)]

    predictions = net.predict_batches(inp)

    for sentence, preds in itertools.izip(sentences, predictions):
        for word, pred in itertools.izip(sentence, preds):
            print "{}\t{}".format(word, I2T[pred])
        print ""


if __name__ == '__main__':
    main()
