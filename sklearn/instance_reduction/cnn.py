# -*- coding: utf-8 -*-
"""
Condensed-Nearest Neighbors
"""

# Author: Dayvid Victor <victor.dvro@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from scipy import sparse as sp

from ..base import BaseEstimator, ClassifierMixin
from ..externals.six.moves import xrange
from ..metrics.pairwise import pairwise_distances
from ..utils.validation import check_arrays, atleast2d_or_csr
from ..neighbors.classification import KNeighborsClassifier
from .base import InstanceReductionMixin


class CondensedNearestNeighbors(BaseEstimator, ClassifierMixin, InstanceReductionMixin):
    """Condensed Nearest Neighbors.

    Each class is represented by a set of prototypes, with test samples
    classified to the class with the nearest prototype.
    The Condensed Nearest Neighbors removes the redundant instances,
    maintaining the samples in the decision boundaries.

    Parameters
    ----------
    n_neighbors : int, optional (default = 1)
        Number of neighbors to use by default for :meth:`k_neighbors` queries.

    Attributes
    ----------
    `prototypes_` : array-like, shape = [indeterminated, n_features]
        Selected prototypes.

    `labels_` : array-like, shape = [indeterminated]
        Labels of the selected prototypes.

    `reduction_` : float, percentual of reduction.

    Examples
    --------
    >>> from sklearn.instance_reduction.cnn import CondensedNearestNeighbors
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> cnn = CondensedNearestNeighbors()
    >>> cnn.fit(X, y)
    CondensedNearestNeighbors(n_neighbors=1)
    >>> print(cnn.predict([[-0.8, -1]]))
    [1]

    See also
    --------
    sklearn.neighbors.KNeighborsClassifier: nearest neighbors classifier

    Notes
    -----
    The Condensed Nearest Neighbor is one the first prototype selection
    technique in literature.

    References
    ----------
    P. E. Hart, The condensed nearest neighbor rule, IEEE Transactions on 
    Information Theory 14 (1968) 515–516.

    """

    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors

    def reduce(self, X, y):
        
        X, y = check_arrays(X, y, sparse_format="csr")

        prots_s = []
        labels_s = []

        classes = np.unique(y)
        self.classes_ = classes

        for cur_class in classes:
            mask = y == cur_class
            insts = X[mask]
            prots_s = prots_s + [insts[np.random.randint(0, insts.shape[0])]]
            labels_s = labels_s + [cur_class]


        knn = KNeighborsClassifier(n_neighbors = self.n_neighbors)
        knn.fit(prots_s, labels_s)
        for sample, label in zip(X, y):
            if knn.predict(sample) != [label]:
                prots_s = prots_s + [sample]
                labels_s = labels_s + [label]
                knn.fit(prots_s, labels_s)
       
        self.prototypes_ = np.asarray(prots_s)
        self.labels_ = np.asarray(labels_s)
        self.reduction_ = 1.0 - float(len(self.labels_))/len(y)
        return self.prototypes_, self.labels_
 
