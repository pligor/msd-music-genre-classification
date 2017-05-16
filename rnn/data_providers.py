from __future__ import division
from mlp.data_providers import OneOfKDataProvider
import numpy as np
import os


class MSDDataProvider(OneOfKDataProvider):
    """Data provider for Million Song Dataset classification task."""

    def __init__(self, which_set='train', batch_size=100, max_num_batches=-1,
                 shuffle_order=True, rng=None,
                 data_filtering=lambda inputs, targets: (inputs, targets),
                 initial_order=None, num_classes=10):
        """Create a new Million Song Dataset data provider object.

        Args:
            which_set: One of 'train' or 'valid'. Determines which
                portion of the MSD data this object should provide.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        # check a valid which_set was provided
        assert which_set in ['train', 'valid', 'test', 'train_valid'], (
            'Expected which_set to be either train or valid. '
            'Got {0}'.format(which_set)
        )
        self.which_set = which_set
        self.num_classes = num_classes
        # construct path to data using os.path.join to ensure the correct path
        # separator for the current platform / OS is used
        # MLP_DATA_DIR environment variable should point to the data directory
        data_path = os.path.join(
            os.environ['MLP_DATA_DIR'],
            'msd-{}-genre-{}.npz'.format(num_classes, which_set))
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )
        # load data from compressed numpy file
        loaded = np.load(data_path)
        inputs, targets = data_filtering(loaded['inputs'], loaded['targets'])
        # flatten inputs to vectors and upcast to float32
        inputs = inputs.reshape((inputs.shape[0], -1)).astype('float32')
        # label map gives strings corresponding to integer label targets
        self.label_map = loaded['label_map']
        # pass the loaded data to the parent class __init__
        super(MSDDataProvider, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng, initial_order=initial_order)


class MSD120RnnDataProvider(MSDDataProvider):
    """Simple wrapper data provider for training an autoencoder on MNIST."""

    def __init__(self, num_steps, segment_count=120, segment_len=25, which_set='train',
                 batch_size=100, max_num_batches=-1,
                 shuffle_order=True, rng=None, indices=None, num_classes=10):
        """"""
        if indices is None:
            super(MSD120RnnDataProvider, self).__init__(
                which_set, batch_size, max_num_batches, shuffle_order, rng, num_classes=num_classes)
        else:
            super(MSD120RnnDataProvider, self).__init__(
                which_set, batch_size, max_num_batches, shuffle_order, rng,
                data_filtering=lambda inputs, targets: (inputs[indices], targets[indices]),
                initial_order=indices, num_classes=num_classes
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
            inputs_batch, targets_batch = super(MSD120RnnDataProvider,
                                                self).next()
            self.inputs_reshaped = inputs_batch.reshape(
                (self.batch_size, self.segment_part_count, self.num_steps, self.segment_len)
            )
            self.segment_part_counter = 0
            self.targets_batch = targets_batch

        cur_input_batch = self.inputs_reshaped[:, self.segment_part_counter, :, :]

        cur_segment_part_counter = self.segment_part_counter

        self.segment_part_counter += 1

        return (cur_input_batch, self.targets_batch), cur_segment_part_counter
