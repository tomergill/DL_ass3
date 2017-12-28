import numpy as np
import dynet as dy


class Net:
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
        s_f, s_b = layer1[Net.FORWARD].initial_state(), layer1[Net.BACKWARD].initial_state()
        outs_f, outs_b = s_f.transduce(rep), s_b.transduce(rep[::-1])
        bs = [dy.concatenate(bf, bb) for bf, bb in zip(outs_f, outs_b)]

        s_f, s_b = layer2[Net.FORWARD].initial_state(), layer2[Net.BACKWARD].initial_state()
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


