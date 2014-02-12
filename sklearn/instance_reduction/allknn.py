# -*- coding: utf-8 -*-
"""
All K-Nearest Neighbors
"""

# Author: Dayvid Victor <victor.dvro@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from scipy import sparse as sp

from ..base import BaseEstimator, ClassifierMixin, InstanceReductionMixin
from ..externals.six.moves import xrange
from ..metrics.pairwise import pairwise_distances
from ..utils.validation import check_arrays, atleast2d_or_csr
from ..neighbors.classification import KNeighborsClassifier
from .enn import EditedNearestNeighbors
from .base import InstanceReductionMixin


class AllKNearestNeighbors(BaseEstimator, ClassifierMixin, InstanceReductionMixin):
    """All K-Nearest Neighbors.

    Each class is represented by a set of prototypes, with test samples
    classified to the class with the nearest prototype.
    The All KNN removes the instances in the boundaries, maintaining 
    redudant samples. Creating a much more smooth decision region.
    It is similar to the Repeated-Edited Nearest Neighbors, but it has
    a different approach.

    Parameters
    ----------
    n_neighbors : int, optional (default = 3)
        Number of limit neighbors to use by default for :meth:`k_neighbors` queries.

    Attributes
    ----------
    `prototypes_` : array-like, shape = [indeterminated, n_features]
        Selected prototypes.

    `labels_` : array-like, shape = [indeterminated]
        Labels of the selected prototypes.

    `reduction_` : float, percentual of reduction.

    Examples
    --------
    >>> from sklearn.instance_reduction.allknn import AllKNearestNeighbors
    >>> import numpy as np
    >>> X = np.array([[-1, 0], [-0.8, 1], [-0.8, -1], [-0.5, 0] , [0.5, 0], [1, 0], [0.8, 1], [0.8, -1]])
    >>> y = np.array([1, 1, 1, 2, 1, 2, 2, 2])
    >>> all_kneigh = AllKNearestNeighbors()
    >>> all_kneigh.fit(X, y)
    AllKNearestNeighbors(n_neighbors=5)
    >>> print(all_kneigh.predict([[-0.6, 0.6]]))
    [1]
    >>> print all_kneigh.reduction_
    0.625

    See also
    --------
    sklearn.neighbors.KNeighborsClassifier: nearest neighbors classifier

    References
    ----------
    I. Tomek. An experiment with the edited nearest-neighbor rule. 
    IEEE Transactions on Systems, Man, and Cybernetics, 6(6):448–452, 1976.

    """

    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors

    def reduce(self, X, y):
        X, y = check_arrays(X, y, sparse_format="csr")

        classes = np.unique(y)
        self.classes_ = classes

        edited_nn = EditedNearestNeighbors(n_neighbors = 1)
        p_, l_, r_ = X, y, 1.0

        for k in range(1, self.n_neighbors + 1):
            if l_.shape[0] > k + 1:
                edited_nn.n_neighbors = k
                edited_nn.fit(p_, l_)
                p_ = edited_nn.prototypes_
                l_ = edited_nn.labels_
                r_ = edited_nn.reduction_
             
        self.prototypes_ = p_
        self.labels_ = l_
        self.reduction_ = 1.0 - float(l_.shape[0]) / y.shape[0]

        return self.prototypes_, self.labels_
   
