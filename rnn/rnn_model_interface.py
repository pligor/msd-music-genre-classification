class RnnModelInterface(object):
    def run_rnn_for_data(self, state_size, num_steps, epochs, train_data, valid_data, logits_gathering_enabled = False):
        raise NotImplementedError
