import numpy as np
from mylibs.py_helper import merge_dicts
from collections import OrderedDict
from mylibs.kaggle_helper import create_kaggle_submission_file


class KaggleRNN(object):
    def __init__(self, filename='msd-25-submission.csv'):
        super(KaggleRNN, self).__init__()
        # initialized by parents
        self.softmax = None
        self.inputs = None
        self.targets = None
        self.init_state = None
        self.outputs = None
        self.filename = filename

    def createKaggleFile(self, batch_size, data_provider, sess, state_size, verbose=True):
        assert data_provider.shuffle_order == False

        def zeroState():
            return np.zeros([batch_size, state_size])

        # chopping off to make batch size fit exactly
        length = len(data_provider.inputs) - (len(data_provider.inputs) % batch_size)
        if verbose and length == len(data_provider.inputs):
            print "data_provider is divided exactly by batch size"
        else:
            print "data_provider is NOT divided exactly by batch size"

        # np.empty
        test_predictions = np.zeros((length, data_provider.num_classes))

        cur_state = zeroState()

        step = 0
        # instances_order = data_provider._current_order
        for (input_batch, target_batch), segmentPartCounter in data_provider:
            batch_predictions, cur_state = sess.run(
                [self.softmax, self.outputs],
                feed_dict=merge_dicts({self.inputs: input_batch,
                                       self.targets: target_batch,
                                       self.init_state: cur_state,
                                       },
                                      {self.training: False} if hasattr(self, "training") else {}
                                      )
            )

            if (segmentPartCounter + 1) % data_provider.segment_part_count == 0:
                cur_state = zeroState()
                test_predictions[step * batch_size:(step + 1) * batch_size, :] = batch_predictions

                step += 1

        # instances_order, test_predictions
        # re order not necessary because shuffle was set to false
        assert np.all(test_predictions != 0)  # all expected to be something else than zero

        create_kaggle_submission_file(test_predictions, output_file=self.filename, overwrite=True)
