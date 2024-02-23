import pandas as pd
import numpy as np
from ConfigSpace import Integer, Float
from ConfigSpace import ConfigurationSpace
from lenskit.data import sparse_ratings
from lenskit.algorithms import Recommender, Predictor
from lkauto.algorithms import implicit_wrapper


class ALS(implicit_wrapper.ALS):
    def __init__(self, factors, **kwargs):
        super().__init__(nnbrs=factors, **kwargs)

    @staticmethod
    def get_default_configspace(**kwargs):
        factors = Integer('factors', bounds=(50, 200), default=100, log=True)  # No default value given by LensKit
        regularization = Float('regularization', bounds=(0.001, 1), default=0.01, log=True)
        iterations = Integer('iterations', bounds=(10, 30), default=15, log=True)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([factors, regularization, iterations])

        return cs


class BPR(implicit_wrapper.BPR):
    def __init__(self, factors, **kwargs):
        super().__init__(nnbrs=factors, **kwargs)

    @staticmethod
    def get_default_configspace(**kwargs):
        factors = Integer('factors', bounds=(50, 200), default=100, log=True)  # No default value given by LensKit
        learning_rate = Float('learning_rate', bounds=(0.001, 0.1), default=0.01, log=True)
        regularization = Float('regularization', bounds=(0.001, 1), default=0.01, log=True)
        iterations = Integer('iterations', bounds=(60, 120), default=100, log=True)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([factors, learning_rate, regularization, iterations])

        return cs


class LMF(implicit_wrapper.LMF):
    def __init__(self, factors, **kwargs):
        super().__init__(nnbrs=factors, **kwargs)

    @staticmethod
    def get_default_configspace(**kwargs):
        factors = Integer('factors', bounds=(1, 1000), default=10, log=True)  # No default value given by LensKit
        learning_rate = Float('learning_rate', bounds=(0.001, 1), default=0.01, log=True)
        regularization = Float('regularization', bounds=(0.01, 1), default=0.6, log=True)
        iterations = Integer('iterations', bounds=(15, 50), default=30, log=True)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([factors, learning_rate, regularization, iterations])

        return cs
