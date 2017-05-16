import numpy as np
from sklearn.model_selection import StratifiedKFold

from rnn.msd10_data_providers import MSD10Genre_120_rnn_DataProvider
from rnn.rnn_model_interface import RnnModelInterface


class CrossValidator(RnnModelInterface):
    which_set = 'train_valid'
    num_classes = 10

    def __init__(self, batch_size, seed):
        super(CrossValidator, self).__init__()
        self.seed = seed
        self.batch_size = batch_size

    def __getIterator(self, n_splits, num_steps):
        kFold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)

        data_provider = MSD10Genre_120_rnn_DataProvider(num_steps=num_steps, which_set=self.which_set,
                                                        batch_size=self.batch_size)

        inputLen = len(data_provider.inputs)

        assert (len(data_provider.inputs) / n_splits) % self.batch_size == 0

        return kFold.split(data_provider.inputs, data_provider.targets), inputLen

    def cross_validate(self, n_splits, state_size, num_steps, epochs):
        stats_list = []

        indsIterator, inputLen = self.__getIterator(n_splits, num_steps)

        for train_indices, valid_indices in indsIterator:
            train_data = MSD10Genre_120_rnn_DataProvider(num_steps=num_steps, which_set=self.which_set,
                                                         batch_size=self.batch_size, indices=train_indices)

            valid_data = MSD10Genre_120_rnn_DataProvider(num_steps=num_steps, which_set=self.which_set,
                                                         batch_size=self.batch_size, indices=valid_indices)

            stats, keys = self.run_rnn_for_data(state_size=state_size, num_steps=num_steps, epochs=epochs,
                                                train_data=train_data, valid_data=valid_data)

            stats_list.append(stats)

        return stats_list

    def cross_gather_logits(self, n_splits, state_size, num_steps, epochs,
                            onTrainEnd=lambda stats_arr, logits_vals_arr: None):
        stats_list = []

        indsIterator, inputLen = self.__getIterator(n_splits, num_steps)
        logits_arr = np.zeros( (inputLen, self.num_classes) )

        for train_indices, valid_indices in indsIterator:
            train_data = MSD10Genre_120_rnn_DataProvider(num_steps=num_steps, which_set=self.which_set,
                                                         batch_size=self.batch_size, indices=train_indices)

            valid_data = MSD10Genre_120_rnn_DataProvider(num_steps=num_steps, which_set=self.which_set,
                                                         batch_size=self.batch_size, indices=valid_indices)

            stats, keys, logits_dict = self.run_rnn_for_data(state_size=state_size, num_steps=num_steps,
                                                               epochs=epochs,
                                                               train_data=train_data,
                                                               valid_data=valid_data,
                                                               logits_gathering_enabled=True)

            logits_arr[logits_dict.keys()] = logits_dict.values()

            stats_list.append(stats)

            onTrainEnd(stats, logits_dict)

        return stats_list, logits_arr
