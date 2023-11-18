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
        factors = Integer('factors', bounds=(1, 1000), default=10, log=True)  # No default value given by LensKit

        cs = ConfigurationSpace()
        cs.add_hyperparameters([factors])

        return cs


class BPR(implicit_wrapper.BPR):
    def __init__(self, factors, **kwargs):
        super().__init__(nnbrs=factors, **kwargs)

    @staticmethod
    def get_default_configspace(**kwargs):
        factors = Integer('factors', bounds=(1, 1000), default=10, log=True)  # No default value given by LensKit

        cs = ConfigurationSpace()
        cs.add_hyperparameters([factors])

        return cs


class LMF(implicit_wrapper.LMF):
    def __init__(self, factors, **kwargs):
        super().__init__(nnbrs=factors, **kwargs)

    @staticmethod
    def get_default_configspace(**kwargs):
        factors = Integer('factors', bounds=(1, 1000), default=10, log=True)  # No default value given by LensKit

        cs = ConfigurationSpace()
        cs.add_hyperparameters([factors])

        return cs
