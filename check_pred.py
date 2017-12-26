import re


def check(good_pattern, words_and_tags):
    good = 0.0
    wrong = []
    for i, (word, tag) in enumerate(words_and_tags):
        real_tag = "good" if re.match(good_pattern, word) is not None else "bad"
        if real_tag == tag:
            good += 1
        else:
            wrong.append((i, word, tag))
    return good / len(words_and_tags), wrong


def main():
    num_pattern = '[0-9]+'
    good_pattern = num_pattern + 'a' + num_pattern + 'b' + num_pattern + 'c' + num_pattern + 'd' + num_pattern

    f = "test1.pred"
    words_and_tags = [tuple(line[:-1].split(" \t")) for line in file(f) if line != "\n"]
    acc, wrong = check(good_pattern, words_and_tags)

    print "Accuracy: {}%\n".format(acc * 100)
    print "Wrong:"
    print "line tag word"
    for i, word, tag in wrong:
        print i, tag, word


if __name__ == "__main__":
    main()
