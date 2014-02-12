# -*- coding: utf-8 -*-
"""
Edited-Nearest Neighbors
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


class EditedNearestNeighbors(BaseEstimator, ClassifierMixin, InstanceReductionMixin):
    """Edited Nearest Neighbors.

    Each class is represented by a set of prototypes, with test samples
    classified to the class with the nearest prototype.
    The Edited Nearest Neighbors  removes the instances in de boundaries,
    maintaining redudant samples.

    Parameters
    ----------
    n_neighbors : int, optional (default = 3)
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
    >>> from sklearn.instance_reduction.enn import EditedNearestNeighbors
    >>> import numpy as np
    >>> X = np.array([[-1, 0], [-0.8, 1], [-0.8, -1], [-0.5, 0] , [0.5, 0], [1, 0], [0.8, 1], [0.8, -1]])
    >>> y = np.array([1, 1, 1, 2, 1, 2, 2, 2])
    >>> editednn = EditedNearestNeighbors()
    >>> editednn.fit(X, y)
    EditedNearestNeighbors(n_neighbors=3)
    >>> print(editednn.predict([[-0.6, 0.6]]))
    [1]
    >>> print editednn.reduction_
    0.25

    See also
    --------
    sklearn.neighbors.KNeighborsClassifier: nearest neighbors classifier

    References
    ----------
    Ruiqin Chang, Zheng Pei, and Chao Zhang. A modified editing k-nearest
    neighbor rule. JCP, 6(7):1493–1500, 2011.

    """

    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors

    def reduce(self, X, y):
        X, y = check_arrays(X, y, sparse_format="csr")

        classes = np.unique(y)
        self.classes_ = classes

        mask = np.zeros(y.size, dtype=bool)
        knn = KNeighborsClassifier(n_neighbors = self.n_neighbors)
        knn.fit(X, y)

        for i in xrange(y.size):
            sample, label = X[i], y[i]
            if knn.predict(sample) == [label]:
                mask[i] = not mask[i]
            
        self.prototypes_ = np.asarray(X[mask])
        self.labels_ = np.asarray(y[mask])
        self.reduction_ = 1.0 - float(len(self.labels_))/len(y)
        return self.prototypes_, self.labels_
 
