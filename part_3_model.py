import numpy as np
import dynet as dy


class AbstractNet:
    FORWARD = 0
    BACKWARD = 1

    def __init__(self, num_layers, embed_dim, lstm1_dim, in_dim, out_dim, vocab_size):
        self.pc = dy.ParameterCollection()
        biLSTM1 = [dy.LSTMBuilder(num_layers, embed_dim, lstm1_dim, self.pc),
                   dy.LSTMBuilder(num_layers, embed_dim, lstm1_dim, self.pc)]
        biLSTM2 = [dy.LSTMBuilder(num_layers, 2 * lstm1_dim, in_dim, self.pc),
                   dy.LSTMBuilder(num_layers, 2 * lstm1_dim, in_dim, self.pc)]
        self._biLSTMs = (biLSTM1, biLSTM2)
        self._E = self.pc.add_lookup_parameters((vocab_size, embed_dim))
        self._W = self.pc.add_parameters((out_dim, in_dim))
        self._b = self.pc.add_parameters(out_dim)

    def repr(self, sentence):
        raise NotImplementedError

    def __call__(self, sentence):
        dy.renew_cg()
        rep = self.repr(sentence)

        layer1, layer2 = self._biLSTMs
        s_f, s_b = layer1[AbstractNet.FORWARD].initial_state(), layer1[AbstractNet.BACKWARD].initial_state()
        outs_f, outs_b = s_f.transduce(rep), s_b.transduce(rep[::-1])
        bs = [dy.concatenate(bf, bb) for bf, bb in zip(outs_f, outs_b)]

        s_f, s_b = layer2[AbstractNet.FORWARD].initial_state(), layer2[AbstractNet.BACKWARD].initial_state()
        outs_f, outs_b = s_f.transduce(bs), s_b.transduce(bs[::-1])
        btags = [dy.concatenate(bf, bb) for bf, bb in zip(outs_f, outs_b)]

        W, b = dy.parameter(self._W), dy.parameter(self._b)
        outs = [dy.softmax(W*x+b) for x in btags]
        return outs

    def compute_loss(self, sentence, expected_outputs):
        probs = self(sentence)
        return [-dy.log(dy.pick(prob, expected)) for prob, expected in zip(probs, expected_outputs)]

    def predict(self, sentence):
        probs = self(sentence)
        return [np.argmax(prob.npvalue()) for prob in probs]


# Option (a)
class WordEmbeddedNet(AbstractNet):
    def repr(self, sentence):
        return [dy.lookup(self._E, i) for i in sentence]


# Option (b)
class CharEmbeddedLSTMNet(AbstractNet):
    def __init__(self, num_layers, embed_dim, lstm1_dim, in_dim, out_dim, vocab_size):
        AbstractNet.__init__(self, num_layers, embed_dim, lstm1_dim, in_dim, out_dim, vocab_size)
        self.char_LSTM = dy.LSTMBuilder(num_layers, embed_dim, embed_dim, self.pc)

    def repr(self, sentence):
        s = self.char_LSTM.initial_state()
        return [s.transduce([dy.lookup(self._E, c) for c in word]) for word in sentence]


# Option (c)
class WordAndSubwordEmbeddedNet(AbstractNet):
    def __init__(self, num_layers, embed_dim, lstm1_dim, in_dim, out_dim, vocab_size,
                 word_to_pre_index, word_to_suf_index):
        AbstractNet.__init__(self, num_layers, embed_dim, lstm1_dim, in_dim, out_dim, vocab_size)
        self._PE = self.pc.add_lookup_parameters((len(word_to_pre_index), embed_dim))  # prefixes
        self._SE = self.pc.add_lookup_parameters((len(word_to_suf_index), embed_dim))  # suffixes
        self._W2PI = word_to_pre_index
        self._W2SI = word_to_suf_index

    def repr(self, sentence):
        return [dy.lookup(self._E, word) + dy.lookup(self._PE, self._W2PI[word]) +
                dy.lookup(self._SE, self._W2SI[word]) for word in sentence]


# Option (d)
class WordEmbeddedAndCharEmbeddedLSTMNet(CharEmbeddedLSTMNet):
    def __init__(self, num_layers, embed_dim, lstm1_dim, in_dim, out_dim, char_vocab_size,
                 vocab_size):
        CharEmbeddedLSTMNet.__init__(self, num_layers, embed_dim, lstm1_dim, in_dim, out_dim,
                                     char_vocab_size)
        self._WE = self.pc.add_lookup_parameters((vocab_size, embed_dim))  # word embedding

    def repr(self, sentence):
        words, chars = zip(*sentence)
        chars = CharEmbeddedLSTMNet.repr(self, chars)
        return [dy.concatenate([dy.lookup(self._WE, word), characters])
                for word, characters in zip(words, chars)]