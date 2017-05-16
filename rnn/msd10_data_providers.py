from __future__ import division
from mlp.data_providers import MSD10GenreDataProvider
import numpy as np


class MSD10Genre_120_rnn_DataProvider(MSD10GenreDataProvider):
    """Simple wrapper data provider for training an autoencoder on MNIST."""

    def __init__(self, num_steps, segment_count=120, segment_len=25, which_set='train',
                 batch_size=100, max_num_batches=-1,
                 shuffle_order=True, rng=None, indices = None):
        """"""
        if indices is None:
            super(MSD10Genre_120_rnn_DataProvider, self).__init__(
                which_set, batch_size, max_num_batches, shuffle_order, rng)
        else:
            super(MSD10Genre_120_rnn_DataProvider, self).__init__(
                which_set, batch_size, max_num_batches, shuffle_order, rng,
                data_filtering=lambda inputs, targets: (inputs[indices], targets[indices]),
                initial_order = indices
            )

        self.segment_count = segment_count
        self.segment_len = segment_len

        segmentPartCount = segment_count / num_steps
        assert segmentPartCount == int(segmentPartCount)
        self.segment_part_count = int(segmentPartCount)
        self.num_steps = num_steps

        self.segment_part_counter = 0
        self.inputs_reshaped = None

        self.targets_batch = None

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        if self.segment_part_counter % self.segment_part_count == 0:
            inputs_batch, targets_batch = super(MSD10Genre_120_rnn_DataProvider,
                                                self).next()
            self.inputs_reshaped = inputs_batch.reshape(
                (self.batch_size, self.segment_part_count, self.num_steps, self.segment_len)
            )
            self.segment_part_counter = 0
            self.targets_batch = targets_batch

        cur_input_batch = self.inputs_reshaped[:, self.segment_part_counter, :, :]

        cur_segment_part_counter = self.segment_part_counter

        self.segment_part_counter += 1

        #print "inside MSD10Genre_120_rnn_DataProvider"
        #print self.targets_batch

        return (cur_input_batch, self.targets_batch), cur_segment_part_counter


#a = np.arange(120)
#np.reshape(a, (2, 3, 4, 5))
#np.reshape(a, (2, 4, 15))
class MSD10Genre_120_rnn_native_DataProvider(MSD10GenreDataProvider):
    """"""
    def __init__(self, num_steps = None, segment_count=120, segment_len=25, which_set='train',
                 batch_size=100, max_num_batches=-1,
                 shuffle_order=True, rng=None, indices = None):
        """"""
        if indices is None:
            super(MSD10Genre_120_rnn_native_DataProvider, self).__init__(
                which_set, batch_size, max_num_batches, shuffle_order, rng)
        else:
            super(MSD10Genre_120_rnn_native_DataProvider, self).__init__(
                which_set, batch_size, max_num_batches, shuffle_order, rng,
                data_filtering=lambda inputs, targets: (inputs[indices], targets[indices]))

        self.segment_len = segment_len
        self.segment_count = segment_count

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        inputs_batch, targets_batch = super(MSD10Genre_120_rnn_native_DataProvider,
                                            self).next()

        inputs_reshaped = inputs_batch.reshape(
            (self.batch_size, self.segment_count, self.segment_len)
        )

        return inputs_reshaped, targets_batch
