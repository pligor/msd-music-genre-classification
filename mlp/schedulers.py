# -*- coding: utf-8 -*-
"""Training schedulers.

This module contains classes implementing schedulers which control the
evolution of learning rule hyperparameters (such as learning rate) over a
training run.
"""

from layers import DropoutLayer

import numpy as np

import logging

logger = logging.getLogger(__name__)


class AnnealedDropoutScheduler(object):
    """Start with a small momentum at the initial epochs and then proceed with a larger epoch"""

    def __init__(self, model, startInputInclProb, startHiddenInclProb, epochs, holdOnPercent=0.1):
        """Construct a new constant learning rate scheduler object.

        Args:
            startInputInclProb: initial inclusive probability of input values (recommended to be higher than startHiddenInclProb)
            startHiddenInclProb: initial inclusive probability of all hidden layers
        """

        assert startInputInclProb > 0 and startInputInclProb < 1, \
            "think that startInputInclProb is a probability, no we will not allow edge values"

        assert startHiddenInclProb > 0 and startHiddenInclProb < 1, \
            "think that startHiddenInclProb is a probability, no we will not allow edge values"

        assert holdOnPercent >= 0 and holdOnPercent <= 1, \
            "think that holdOnPercent is a probability"

        assert epochs == int(epochs)

        epochs = int(epochs)

        finalProbability = 1

        epochsWithProbOne = int(np.ceil(holdOnPercent * epochs))

        epochsAnnealing = epochs - epochsWithProbOne

        inputInclProbs = np.linspace(startInputInclProb, finalProbability, epochsAnnealing)
        hiddenInclProbs = np.linspace(startHiddenInclProb, finalProbability, epochsAnnealing)

        tailProbs = np.ones(epochsWithProbOne) * finalProbability

        self.inputInclProbs = np.concatenate((inputInclProbs, tailProbs))
        self.hiddenInclProbs = np.concatenate((hiddenInclProbs, tailProbs))

        assert len(self.inputInclProbs) == len(self.hiddenInclProbs) and len(self.inputInclProbs) == epochs

        self.model = model

    def assertProbsInModel(self, inputIncProb, hiddenInclProb):
        # here we take for granted that the model are from input to output <=> from first to last
        isFirst = True

        for i, layer in enumerate(self.model.layers):
            # logger.info(type(layer))
            if isinstance(layer, DropoutLayer):
                curProb = inputIncProb if isFirst else hiddenInclProb
                isFirst = False
                assert layer.incl_prob == curProb, \
                    "the probability of layer %d is %f while we were expecting %f" % (i, layer.incl_prob, curProb)

    def getProbsFromModel(self):
        # here we take for granted that the model are from input to output <=> from first to last

        probs = []

        for layer in self.model.layers:
            if isinstance(layer, DropoutLayer):
                probs.append(
                    layer.incl_prob
                )

        return probs

    def changeProbsInModel(self, inputIncProb, hiddenInclProb):
        # here we take for granted that the model are from input to output <=> from first to last
        isFirst = True

        for layer in self.model.layers:
            # logger.info(type(layer))
            if isinstance(layer, DropoutLayer):
                layer.incl_prob = inputIncProb if isFirst else hiddenInclProb
                isFirst = False

    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Run at the beginning of each epoch.

        Args:
            learning_rule: WE DO NOT DEAL WITH THE LEARNING RULE in this scheduler
                Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.

            epoch_number: Integer index of training epoch about to be run.
        """
        curInputInclProb = self.inputInclProbs[epoch_number]
        curHiddenInclProb = self.hiddenInclProbs[epoch_number]

        self.changeProbsInModel(curInputInclProb, curHiddenInclProb)

        # logger.info("input incl prob: %f" % curInputInclProb)
        # logger.info("hidden incl prob: %f" % curHiddenInclProb)

        logger.info("all probs from model: " + ", ".join([str(p) for p in self.getProbsFromModel()]))

        self.assertProbsInModel(curInputInclProb, curHiddenInclProb)


class MomentumCoefficientScheduler(object):
    """Start with a small momentum at the initial epochs and then proceed with a larger epoch"""

    def __init__(self, a_asymptoticMomentumCoefficient=0.9, gamma=9.9, taf=10):
        """Construct a new constant learning rate scheduler object.

        Args:
            learning_rate: Learning rate to use in learning rule.
        """
        assert a_asymptoticMomentumCoefficient >= 0 and a_asymptoticMomentumCoefficient <= 1
        assert gamma >= 0 and gamma <= taf
        assert taf >= 1

        self.taf = taf
        self.gamma = gamma
        self.a_asymptoticMomentumCoefficient = a_asymptoticMomentumCoefficient

    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Run at the beginning of each epoch.

        Args:
            learning_rule: Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.
            epoch_number: Integer index of training epoch about to be run.
        """
        learning_rule.mom_coeff = self.a_asymptoticMomentumCoefficient * (1 - self.gamma / (epoch_number + self.taf))


class ExponentialLearningRateScheduler(object):
    """Decrease Learning Rate Exponentially"""

    # learning_rate = 0

    def __init__(self, learning_rate=1.5, r=2.1):
        """Construct a new constant learning rate scheduler object.

        Args:
            learning_rate: Learning rate to use in learning rule.
        """
        self.learning_rate = learning_rate
        self.r = r

    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Run at the beginning of each epoch.

        Args:
            learning_rule: Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.
            epoch_number: Integer index of training epoch about to be run.
        """
        learning_rule.learning_rate = self.learning_rate * np.exp(-epoch_number / self.r)


class ReciprocalLearningRateScheduler(object):
    """Decrease the learning rate by the reciprocal"""

    # learning_rate = 0

    def __init__(self, learning_rate=0.01, r=1):
        """Construct a new constant learning rate scheduler object.

        Args:
            learning_rate: Learning rate to use in learning rule.
        """
        self.learning_rate = learning_rate
        self.r = r

    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Run at the beginning of each epoch.

        Args:
            learning_rule: Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.
            epoch_number: Integer index of training epoch about to be run.
        """
        learning_rule.learning_rate = self.learning_rate / (1 + (epoch_number / self.r))


class ConstantLearningRateScheduler(object):
    """Example of scheduler interface which sets a constant learning rate."""

    def __init__(self, learning_rate):
        """Construct a new constant learning rate scheduler object.

        Args:
            learning_rate: Learning rate to use in learning rule.
        """
        self.learning_rate = learning_rate

    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Runs at the beginning of each epoch.

        Args:
            learning_rule: Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.
            epoch_number: Integer index of training epoch about to be run.
        """
        learning_rule.learning_rate = self.learning_rate


class ExponentialLearningRateScheduler(object):
    """Exponential decay learning rate scheduler."""

    def __init__(self, init_learning_rate, decay_param):
        """Construct a new learning rate scheduler object.

        Args:
            init_learning_rate: Initial learning rate at epoch 0. Should be a
                positive value.
            decay_param: Parameter governing rate of learning rate decay.
                Should be a positive value.
        """
        self.init_learning_rate = init_learning_rate
        self.decay_param = decay_param

    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Runs at the beginning of each epoch.

        Args:
            learning_rule: Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.
            epoch_number: Integer index of training epoch about to be run.
        """
        learning_rule.learning_rate = (
            self.init_learning_rate * np.exp(-epoch_number / self.decay_param))


class ReciprocalLearningRateScheduler(object):
    """Reciprocal decay learning rate scheduler."""

    def __init__(self, init_learning_rate, decay_param):
        """Construct a new learning rate scheduler object.

        Args:
            init_learning_rate: Initial learning rate at epoch 0. Should be a
                positive value.
            decay_param: Parameter governing rate of learning rate decay.
                Should be a positive value.
        """
        self.init_learning_rate = init_learning_rate
        self.decay_param = decay_param

    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Runs at the beginning of each epoch.

        Args:
            learning_rule: Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.
            epoch_number: Integer index of training epoch about to be run.
        """
        learning_rule.learning_rate = (
            self.init_learning_rate / (1. + epoch_number / self.decay_param)
        )


class ReciprocalMomentumCoefficientScheduler(object):
    """Reciprocal growth momentum coefficient scheduler."""

    def __init__(self, max_mom_coeff=0.99, growth_param=3., epoch_offset=5.):
        """Construct a new reciprocal momentum coefficient scheduler object.

        Args:
            max_mom_coeff: Maximum momentum coefficient to tend to. Should be
                in [0, 1].
            growth_param: Parameter governing rate of increase of momentum
                coefficient over training. Should be >= 0 and <= epoch_offset.
            epoch_offset: Offset to epoch counter to in scheduler updates to
                govern how quickly momentum initially increases. Should be
                >= 1.
        """
        assert max_mom_coeff >= 0. and max_mom_coeff <= 1.
        assert growth_param >= 0. and growth_param <= epoch_offset
        assert epoch_offset >= 1.
        self.max_mom_coeff = max_mom_coeff
        self.growth_param = growth_param
        self.epoch_offset = epoch_offset

    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Runs at the beginning of each epoch.

        Args:
            learning_rule: Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.
            epoch_number: Integer index of training epoch about to be run.
        """
        learning_rule.mom_coeff = self.max_mom_coeff * (
            1. - self.growth_param / (epoch_number + self.epoch_offset)
        )
