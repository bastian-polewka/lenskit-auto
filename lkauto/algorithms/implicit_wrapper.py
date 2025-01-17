import logging
import inspect
import pandas as pd
import numpy as np

from lenskit.data import sparse_ratings
from lenskit.algorithms import Recommender, Predictor

from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization

_logger = logging.getLogger(__name__)

__all__ = [
    'BaseRec',
    'ALS',
    'BPR',
    'LMF',
]


class BaseRec(Recommender, Predictor):
    """
    Base class for Implicit-backed recommenders.

    Args:
        delegate(implicit.RecommenderBase):
            The delegate algorithm.

    Attributes:
        delegate(implicit.RecommenderBase):
            The :py:mod:`implicit` delegate algorithm.
        matrix_(scipy.sparse.csr_matrix):
            The user-item rating matrix.
        user_index_(pandas.Index):
            The user index.
        item_index_(pandas.Index):
            The item index.
    """

    def __init__(self, delegate):
        self.delegate = delegate
        self.weight = 1.0

    def fit(self, ratings, **kwargs):
        matrix, users, items = sparse_ratings(ratings, scipy=True)
        uir = matrix.tocsr()
        uir.data *= self.weight
        if getattr(self.delegate, 'item_factors', None) is not None:
            _logger.warn("implicit algorithm already trained, re-fit is usually a bug")

        _logger.info('training %s on %s matrix (%d nnz)', self.delegate, uir.shape, uir.nnz)

        self.delegate.fit(uir)

        self.matrix_ = matrix
        self.user_index_ = users
        self.item_index_ = items

        return self

    def recommend(self, user, n=None, candidates=None, ratings=None):
        try:
            uid = self.user_index_.get_loc(user)
        except KeyError:
            return pd.DataFrame({'item': []})

        matrix = self.matrix_[[uid], :]

        if candidates is None:
            i_n = n if n is not None else len(self.item_index_)
            recs, scores = self.delegate.recommend(uid, matrix, N=i_n)
        else:
            cands = self.item_index_.get_indexer(candidates)
            cands = cands[cands >= 0]
            recs, scores = self.delegate.recommend(uid, matrix, items=cands)

        if n is not None:
            recs = recs[:n]
            scores = scores[:n]

        rec_df = pd.DataFrame({
            'item_pos': recs,
            'score': scores,
        })
        rec_df['item'] = self.item_index_[rec_df.item_pos]
        return rec_df.loc[:, ['item', 'score']]

    def predict_for_user(self, user, items, ratings=None):
        try:
            uid = self.user_index_.get_loc(user)
        except KeyError:
            return pd.Series(np.nan, index=items)

        iids = self.item_index_.get_indexer(items)
        iids = iids[iids >= 0]

        ifs = self.delegate.item_factors[iids]
        uf = self.delegate.user_factors[uid]
        # convert back if these are on CUDA
        if hasattr(ifs, 'to_numpy'):
            ifs = ifs.to_numpy()
            uf = uf.to_numpy()
        scores = np.dot(ifs, uf.T)
        scores = pd.Series(np.ravel(scores), index=self.item_index_[iids])
        return scores.reindex(items)

    def __getattr__(self, name):
        if 'delegate' not in self.__dict__:
            raise AttributeError()
        dd = self.delegate.__dict__
        if name in dd:
            return dd[name]
        else:
            raise AttributeError()

    def get_params(self, deep=True):
        dd = self.delegate.__dict__
        sig = inspect.signature(self.delegate.__class__)
        names = list(sig.parameters.keys())
        return dict([(k, dd.get(k)) for k in names])

    def __str__(self):
        return 'Implicit({})'.format(self.delegate)


class ALS(BaseRec):
    """
    LensKit interface to :py:mod:`implicit.als`.
    """

    def __init__(self, *args, weight=40.0, **kwargs):
        """
        Construct an ALS recommender.  The arguments are passed as-is to
        :py:class:`implicit.als.AlternatingLeastSquares`.  The `weight`
        parameter controls the confidence weight for positive examples.
        """

        super().__init__(AlternatingLeastSquares(*args, **kwargs))
        self.weight = weight


class BPR(BaseRec):
    """
    LensKit interface to :py:mod:`implicit.bpr`.
    """

    def __init__(self, *args, **kwargs):
        """
        Construct an BPR recommender.  The arguments are passed as-is to
        :py:class:`implicit.bpr.BayesianPersonalizedRanking`.
        """
        super().__init__(BayesianPersonalizedRanking(*args, **kwargs))


class LMF(BaseRec):
    """
    LensKit interface to :py:mod:`implicit.lmf`.
    """

    def __init__(self, *args, **kwargs):
        """
        Construct an LMF recommender.  The arguments are passed as-is to
        :py:class:`implicit.lmf.LogisticMatrixFactorization`.
        """
        super().__init__(LogisticMatrixFactorization(*args, **kwargs))
