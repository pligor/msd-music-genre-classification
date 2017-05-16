from mlp.data_providers import MSD10GenreDataProvider
from operator import itemgetter
import numpy as np

# dp = MSD10Genre_Ordered(key_order, which_set = 'train', batch_size=batch_size)
# a, b = dp.next()
# a.shape, b.shape
# dp._current_order[:5]
# for i, (a, b) in enumerate(dp):
#     if i < 10:
#         print a.shape, b.shape
#     else: break
class MSD10Genre_Ordered(MSD10GenreDataProvider):
    """Note that curriculum learning is NOT randomized"""

    def __init__(self, key_order, which_set='train',
                 batch_size=100, max_num_batches=-1, reverse_order = False):
        """"""
        super(MSD10Genre_Ordered, self).__init__(
            which_set, batch_size, max_num_batches, shuffle_order=False, rng=None)

        self.key_order = key_order[::-1] if reverse_order else key_order

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        if self._curr_batch + 1 > self.num_batches:
            # no more batches in current iteration through data set so start
            # new epoch ready for another pass and indicate iteration is at end
            self.new_epoch()
            raise StopIteration()

        # create an index slice corresponding to current batch number
        batch_slice = slice(self._curr_batch * self.batch_size,
                            (self._curr_batch + 1) * self.batch_size)
        # print batch_slice
        inds = self.key_order[batch_slice]
        # print inds
        inputs_batch = self.inputs[inds]
        targets_batch = self.targets[inds]
        self._curr_batch += 1

        return inputs_batch, self.to_one_of_k(targets_batch)


# countBatches = total_songs / (batch_size * 8)
# countBatches
# for i in range(3):
#     for a, b in dp:
#         print a.shape, b.shape

#     print
class MSD10Genre_CurriculumLearning(MSD10Genre_Ordered):
    """Note that curriculum learning is NOT randomized"""

    def __init__(self, cross_entropies, which_set='train',
                 batch_size=100, max_num_batches=-1, curriculum_step=8, repetitions=1,
                 repeat_school_class=True, shuffle_cur_curriculum = False, reverse_order = False,
                 enable_auto_level_incr = True):
        """
        curriculum_step is how many batches will it train before it ends the epochs and
        needs retraining

        repetitions is how many times should a certain set of curriculum level be repeated
        before we go onto the next. Note that this has two implementations. One is
        to have the curriculum level be updated with a different frequency. This means more
        epochs are required. The other is to do the repetition internally. Here we are
        implementing the first one.
        """
        assert repetitions == int(repetitions) and repetitions > 0
        self.repetitions = int(repetitions)

        self.enable_auto_level_incr = enable_auto_level_incr

        cross_entropies_indexed = zip(range(len(cross_entropies)), cross_entropies)
        cross_entropies_sorted = sorted(cross_entropies_indexed, key=itemgetter(1))
        key_order = [tpl[0] for tpl in cross_entropies_sorted]

        self.curriculum_initial_level = curriculum_step
        self.curriculum_step = curriculum_step
        self.repetitionCounter = 1
        self.repeat_school_class = repeat_school_class
        self.shuffle_cur_curriculum = shuffle_cur_curriculum

        super(MSD10Genre_CurriculumLearning, self).__init__(
            key_order, which_set, batch_size, max_num_batches, reverse_order=reverse_order)

    def initializeCurBatch(self):
        # instead return only batches that belong to the current level
        if self.repeat_school_class:
            self._curr_batch = 0
        else:
            self._curr_batch = self.curriculum_level - self.curriculum_step  # we start from +1 the end of the previous curr level

        return self._curr_batch

    def graduateFromSchool_ifNecessary(self, batch_of_new_epoch):
        if batch_of_new_epoch >= self.num_batches:
            self.repeat_school_class = True
            self._curr_batch = 0

        return self._curr_batch

    def new_epoch(self):
        """Starts a new epoch (pass through data), possibly shuffling first."""
        #super(MSD10Genre_CurriculumLearning, self).new_epoch() #not use this anymore
        if hasattr(self, "curriculum_level"):
            self.tryUpdateCurriculumLevel()
        else:
            self.curriculum_level = self.curriculum_initial_level

        batch_of_new_epoch = self.graduateFromSchool_ifNecessary(
            self.initializeCurBatch()
        )

        if self.shuffle_cur_curriculum:
            self.shuffle()

        #print "curr level: %d" % self.curriculum_level
        #print "current batch: %d" % self._curr_batch

    def tryUpdateCurriculumLevel(self):
        if self.enable_auto_level_incr and (self.repetitionCounter % self.repetitions == 0):
            self.updateCurriculumLevel()

        self.repetitionCounter += 1

    def updateCurriculumLevel(self):
        self.curriculum_level += self.curriculum_step

    def reset(self):
        """Resets the provider to the initial state."""
        super(MSD10Genre_CurriculumLearning, self).reset()
        self.curriculum_level = self.curriculum_initial_level
        self.repetitionCounter = 0

    def shuffle(self):
        """Randomly shuffles order of data. BUT ONLY THOSE DATA THAT CORRESPOND TO CURRENT CURRICULUM LEVEL"""
        lowbound = self._curr_batch
        upperbound = self.curriculum_level
        interestedSlice = slice(lowbound, upperbound)

        interestedIndices = np.arange(lowbound, upperbound)
        #inputLen = self.inputs.shape[0]

        #curChaos = self._current_order
        #interestedChaos = curChaos[interestedSlice]

        #np.random.permutation()
        perm = self.rng.permutation(interestedIndices)
        self._current_order[interestedSlice] = self._current_order[perm]
        self.inputs[interestedSlice] = self.inputs[perm]
        self.targets[interestedSlice] = self.targets[perm]

    def next(self):
        # assert 0 <= self.curriculum_level < self.num_batches
        # let run fully after the end

        if self._curr_batch < self.curriculum_level:
            inputs_batch, targets_batch = super(MSD10Genre_CurriculumLearning, self).next()
            return inputs_batch, targets_batch
        else:
            self.new_epoch()
            raise StopIteration()
