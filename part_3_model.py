import numpy as np
import dynet as dy


class AbstractNet:
    """
    Class for an abstract neural network for part 3.
    Holds 2 biLSTMs (2 builders for each, one for each direction), an embedding matrix and a
    MLP with 1 hidden layer.
    All subclasses needs to implement the repr function.
    """
    FORWARD = 0  # index of forward builder
    BACKWARD = 1  # index of backward builder

    def __init__(self, num_layers, embed_dim, lstm1_dim, half_in_dim, classes_number, vocab_size):
        """
        Initialize the dynet.ParameterCollection (called pc) and the parameters.

        :param num_layers: Number of layers each LSTM will have
        :param embed_dim: Size of each embedding vector
        :param lstm1_dim: Dimension of the first biLSTM's output vectors
        :param half_in_dim: Dimension of the second biLSTM's output vectors, which in turn are the
        input vectors for the MLP1.
        :param classes_number: Number of different classes the input can be part of. Also the
        dimension of the MLP1's output vector.
        :param vocab_size: Size of vocabulary. AKA how many rows the embedding matrix will have.
        """
        self.pc = dy.ParameterCollection()
        biLSTM1 = [dy.LSTMBuilder(num_layers, embed_dim, lstm1_dim, self.pc),
                   dy.LSTMBuilder(num_layers, embed_dim, lstm1_dim, self.pc)]
        biLSTM2 = [dy.LSTMBuilder(num_layers, 2 * lstm1_dim, half_in_dim, self.pc),
                   dy.LSTMBuilder(num_layers, 2 * lstm1_dim, half_in_dim, self.pc)]
        self._biLSTMs = (biLSTM1, biLSTM2)
        self._E = self.pc.add_lookup_parameters((vocab_size, embed_dim))
        self._W = self.pc.add_parameters((classes_number, 2 * half_in_dim))
        self._b = self.pc.add_parameters(classes_number)

    def repr(self, sentence):
        """
        Abstract method.
        Each network's vector representation for a sentence
        :param sentence: Sentence to be represented
        :return: A list of vector representation for each unit in sentence
        """
        raise NotImplementedError

    def __call__(self, sentence, renew_graph=True):
        """
        Inputs sentence to the network: Get representation of sentence, which is fed to the first
        biLSTM, getting b_1 to b_n. Then inserted into the second biLSTM, thus getting b'_1 up
        to b'_n. Then each b'_i is fed to the MLP1 and the the output returns after a softmax is
        applied.

        :param sentence: Input sentence.
        :return: Softmax vector of the output vector of the net.
        """
        if renew_graph:
            dy.renew_cg()
        rep = self.repr(sentence)

        layer1, layer2 = self._biLSTMs
        s_f, s_b = layer1[AbstractNet.FORWARD].initial_state(), layer1[
            AbstractNet.BACKWARD].initial_state()
        outs_f, outs_b = s_f.transduce(rep), s_b.transduce(rep[::-1])
        bs = [dy.concatenate([bf, bb]) for bf, bb in zip(outs_f, outs_b)]

        s_f, s_b = layer2[AbstractNet.FORWARD].initial_state(), layer2[
            AbstractNet.BACKWARD].initial_state()
        outs_f, outs_b = s_f.transduce(bs), s_b.transduce(bs[::-1])
        btags = [dy.concatenate([bf, bb]) for i, (bf, bb) in enumerate(zip(outs_f, outs_b))]

        W, b = dy.parameter(self._W), dy.parameter(self._b)
        outs = [dy.softmax(W * x + b) for x in btags]
        return outs

    def compute_loss(self, sentence, expected_outputs):
        """
        Inputs to network and returns the cross-entropy loss.

        :param sentence: Input setence.
        :param expected_outputs: List of each word's class' index (output of word i should be
        expected_outputs[i]). With this computes the loss.
        :return: Cross-entropy loss for this sentence (dynet.Expression, should call value()).
        """
        probs = self(sentence)
        return [-dy.log(dy.pick(prob, expected)) for prob, expected in zip(probs, expected_outputs)]

    def predict(self, sentence):
        """
        Inputs to network and returns the index of the class with the maximal probability
        (argmax of output).

        :param sentence: Input sentence
        :return: List of predicted indexes of classes per word. The i-th element is the index of
        the predicted class of teh i-th word in sentence.
        """
        probs = self(sentence)
        return [np.argmax(prob.npvalue()) for prob in probs]

    def predcit_batch(self, sentences):
        probs = []
        all_probs = []
        for sentence in sentences:
            p = self(sentence, renew_graph=False)
            probs.append(p)
            all_probs.extend(p)
        dy.forward(p)
        return [[np.argmax(word.npvalue()) for word in sentence] for sentence in probs]

    def loss_on_batch(self, sentences_and_tags):
        losses = []
        total = 0
        for sentence, tags in sentences_and_tags:
            probs = self(sentence, renew_graph=False)
            total += len(tags)
            losses.extend([-dy.log(dy.pick(prob, tag)) for prob, tag in zip(probs, tags)])
        return dy.esum(losses) / total

    def save_to(self, file_name):
        self.pc.save(file_name)

    def load_from(self, file_name):
        self.pc.populate(file_name)


# Option (a)
class WordEmbeddedNet(AbstractNet):
    """
    Option (a):
    Part 3 network, where the input sentence is embedded (each word is embedded to a vector using
    the embedding matrix).
    """

    def repr(self, sentence):
        """
        Represents each word in sentence with it's embedding vector.

        :param sentence: List of words' indexes
        :return: A list of the embedded vector of each word
        """
        return [dy.lookup(self._E, i) for i in sentence]


# Option (b)
class CharEmbeddedLSTMNet(AbstractNet):
    """
    Option (b):
    Part 3 network, where each word in the input sentence is broken apart to it's characters,
    the characters are embedded to vectors using the embedding matrix and then inputted into a
    LSTM, which output is the word vector representation.
    """

    def __init__(self, num_layers, embed_dim, lstm1_dim, half_in_dim, classes_number, char_vocab_size):
        """
        Initializes the network like the base class, and also initializes the character LSTM.

        :param num_layers: Number of layers each LSTM will have
        :param embed_dim: Size of each embedding vector
        :param lstm1_dim: Dimension of the first biLSTM's output vectors
        :param half_in_dim: Dimension of the second biLSTM's output vectors, which in turn are the
        input vectors for the MLP1.
        :param classes_number: Number of different classes the input can be part of. Also the
        dimension of the MLP1's output vector.
        :param char_vocab_size: How many chars in the vocabulary. AKA how many rows the embedding
        matrix will have.
        """
        AbstractNet.__init__(self, num_layers, embed_dim, lstm1_dim, half_in_dim, classes_number,
                             char_vocab_size)
        self.char_LSTM = dy.LSTMBuilder(num_layers, embed_dim, embed_dim, self.pc)

    def repr(self, sentence):
        """
        Each word's representation is the chars LSTM output for the embedded vectors for each
        char in the word (in order)

        :param sentence: List of lists of chars' indexes (words)
        :return: vector outputs of the embedded char-by-char lstm.
        """
        s = self.char_LSTM.initial_state()
        return [s.transduce([dy.lookup(self._E, c) for c in word]) for word in sentence]


# Option (c)
class WordAndSubwordEmbeddedNet(AbstractNet):
    """
    Option (c):
    Part 3 network, where each word in the input sentence is represented as it's embedding +
    it's prefix embedding + it's suffix embedding.
    """

    def __init__(self, num_layers, embed_dim, lstm1_dim, half_in_dim, classes_number, vocab_size,
                 word_to_pre_index, word_to_suf_index):
        """
        Initializes the base net, and the embedding matrices for the prefixes and the suffixes.

        :param num_layers: Number of layers each LSTM will have
        :param embed_dim: Size of each embedding vector
        :param lstm1_dim: Dimension of the first biLSTM's output vectors
        :param half_in_dim: Dimension of the second biLSTM's output vectors, which in turn are the
        input vectors for the MLP1.
        :param classes_number: Number of different classes the input can be part of. Also the
        dimension of the MLP1's output vector.
        :param vocab_size: Size of vocabulary. AKA how many rows the embedding matrix will have.
        :param word_to_pre_index: List of indexes, sized vocab_size. Given word's index i then
        word_to_pre_index will return the index of it's prefix.
        :param word_to_suf_index: List of indexes, sized vocab_size. Given word's index i then
        word_to_suf_index will return the index of it's suffix.
        """
        AbstractNet.__init__(self, num_layers, embed_dim, lstm1_dim, half_in_dim, classes_number,
                             vocab_size)
        self._PE = self.pc.add_lookup_parameters((len(word_to_pre_index), embed_dim))  # prefixes
        self._SE = self.pc.add_lookup_parameters((len(word_to_suf_index), embed_dim))  # suffixes
        self._W2PI = word_to_pre_index
        self._W2SI = word_to_suf_index

    def repr(self, sentence):
        """
        Represents each word as the sum of it's, it's prefix and it's suffix embedding.

        :param sentence: List of words' indexes
        :return: A list of embedded vectors, each is the sum of the 3 embedding vectors for each
        word.
        """
        return [dy.lookup(self._E, word) + dy.lookup(self._PE, self._W2PI[word]) +
                dy.lookup(self._SE, self._W2SI[word]) for word in sentence]


# Option (d)
class WordEmbeddedAndCharEmbeddedLSTMNet(CharEmbeddedLSTMNet):
    """
    Option (d):
    Part 3 network, where each words represented with 2 concatenated vectors:

    1.  Each word is embedded to a vector using the word embedding matrix

    2.  Each word in the input sentence is broken apart to it's characters,
        the characters are embedded to vectors using the embedding matrix and then inputted
        into a LSTM, which output is the word vector representation.

    Then the 2 concatenated vectors are an input to a linear layer, which it's output is the word
    representation.
    """

    def __init__(self, num_layers, embed_dim, lstm1_dim, half_in_dim, classes_number, char_vocab_size,
                 vocab_size):
        """
        Initializes the network like the base class (the built in embedding matrix is for the
        characters), and also the char LSTM, the embedding matrix for whole words and the linear
        layer components.

        :param num_layers: Number of layers each LSTM will have
        :param embed_dim: Size of each embedding vector
        :param lstm1_dim: Dimension of the first biLSTM's output vectors
        :param half_in_dim: Dimension of the second biLSTM's output vectors, which in turn are the
        input vectors for the MLP1.
        :param classes_number: Number of different classes the input can be part of. Also the
        dimension of the MLP1's output vector.
        :param char_vocab_size: How many chars in the vocabulary. AKA how many rows the embedding
        matrix will have.
        :param vocab_size: How many words are in the vocabulary. AKA how many rows the word
        embedding matrix will have.
        """
        CharEmbeddedLSTMNet.__init__(self, num_layers, embed_dim, lstm1_dim, half_in_dim, classes_number,
                                     char_vocab_size)
        self._WE = self.pc.add_lookup_parameters((vocab_size, embed_dim))  # word embedding
        self._W1 = self.pc.add_parameters((embed_dim, 2 * embed_dim))  # linear layer 4 embedding
        self._b1 = self.pc.add_parameters(embed_dim)

    def repr(self, sentence):
        """
        Represents the word as (a) concatenated to (b), then into a linear layer.

        :param sentence: List of pairs: a word's index and a list of the word's chars' indexes.
        :return: List with each word's representation.
        """
        words, chars = zip(*sentence)
        chars = CharEmbeddedLSTMNet.repr(self, list(chars))
        embedded = [dy.concatenate([dy.lookup(self._WE, word), embedded_chars])
                    for word, embedded_chars in zip(words, chars)]
        W1 = dy.parameter(self._W1)
        b1 = dy.parameter(self._b1)
        return [W1 * x + b1 for x in embedded]
