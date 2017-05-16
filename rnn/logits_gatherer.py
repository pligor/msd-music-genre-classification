import numpy as np
from mylibs.py_helper import merge_dicts
from collections import OrderedDict

class LogitsGatherer(object):
    def __init__(self):
        super(LogitsGatherer, self).__init__()
        # initialized by parents
        self.logits = None
        self.error = None
        self.accuracy = None
        self.inputs = None
        self.targets = None
        self.init_state = None
        self.outputs = None

    def getLogits(self, batch_size, data_provider, sess, state_size, extraFeedDict={}, verbose=True):
        # batch_size = 97 #factors of 9991: [1, 103, 97, 9991]
        total_error = 0.
        total_accuracy = 0.

        def zeroState():
            return np.zeros([batch_size, state_size])

        # chopping off to make batch size fit exactly
        length = len(data_provider.inputs) - (len(data_provider.inputs) % batch_size)
        if verbose and length == len(data_provider.inputs):
            print "data_provider is divided exactly by batch size"
        else:
            print "data_provider is NOT divided exactly by batch size"

        # np.empty
        all_logits = np.zeros((length, data_provider.num_classes))

        cur_state = zeroState()

        step = 0
        instances_order = data_provider._current_order
        #for step, ((input_batch, target_batch), segmentPartCounter) in enumerate(data_provider):
        for (input_batch, target_batch), segmentPartCounter in data_provider:
            batch_logits, cur_state, batch_error, batch_acc = sess.run(
                [self.logits, self.outputs, self.error, self.accuracy],
                feed_dict=merge_dicts({self.inputs: input_batch,
                                       self.targets: target_batch,
                                       self.init_state: cur_state}, extraFeedDict)
            )

            if (segmentPartCounter + 1) % data_provider.segment_part_count == 0:
                cur_state = zeroState()
                # only care about the last output of the training error and acc of the rnn
                # so include it in if-statement
                total_error += batch_error
                total_accuracy += batch_acc
                all_logits[step * len(batch_logits):(step + 1) * len(batch_logits), :] = batch_logits
                assert np.all(instances_order == data_provider._current_order)

                step += 1

        num_batches = data_provider.num_batches

        total_error /= num_batches
        total_accuracy /= num_batches

        assert np.all(all_logits != 0)  # all logits expected to be something else than zero

        logits_dict = OrderedDict(zip(instances_order, all_logits))

        return logits_dict, total_error, total_accuracy
