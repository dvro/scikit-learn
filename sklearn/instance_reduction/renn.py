# -*- coding: utf-8 -*-
"""
Repeated Edited-Nearest Neighbors
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


class RepeatedEditedNearestNeighbors(BaseEstimator, ClassifierMixin, InstanceReductionMixin):
    """Repeated Edited Nearest Neighbors.

    Each class is represented by a set of prototypes, with test samples
    classified to the class with the nearest prototype.
    The Repeated Edited Nearest Neighbors  removes the instances in the 
    boundaries, maintaining redudant samples. Creating a much more smooth
    decision region.

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
    >>> from sklearn.instance_reduction.renn import RepeatedEditedNearestNeighbors
    >>> import numpy as np
    >>> X = np.array([[-1, 0], [-0.8, 1], [-0.8, -1], [-0.5, 0] , [0.5, 0], [1, 0], [0.8, 1], [0.8, -1]])
    >>> y = np.array([1, 1, 1, 2, 1, 2, 2, 2])
    >>> repeated_enn = RepeatedEditedNearestNeighbors()
    >>> repeated_enn.fit(X, y)
    RepeatedEditedNearestNeighbors(n_neighbors=3)
    >>> print(repeated_enn.predict([[-0.6, 0.6]]))
    [1]
    >>> print repeated_enn.reduction_
    0.25

    See also
    --------
    sklearn.neighbors.KNeighborsClassifier: nearest neighbors classifier

    References
    ----------
    Dennis L. Wilson. Asymptotic properties of nearest neighbor rules 
    using edited data. Systems, Man and Cybernetics, IEEE Transactions
    on, 2(3):408–421, July 1972.

    """

    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors

    def reduce(self, X, y):
        X, y = check_arrays(X, y, sparse_format="csr")

        classes = np.unique(y)
        self.classes_ = classes

        edited_nn = EditedNearestNeighbors(n_neighbors = self.n_neighbors)
        p_, l_, r_ = X, y, 1.0

        while r_ != 0:
            edited_nn.fit(p_, l_)
            p_ = edited_nn.prototypes_
            l_ = edited_nn.labels_
            r_ = edited_nn.reduction_
             
        self.prototypes_ = p_
        self.labels_ = l_
        self.reduction_ = 1.0 - float(l_.shape[0]) / y.shape[0]

        return self.prototypes_, self.labels_
 
    def fit(self, X, y):
        """
        Fit the NearestPrototype model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
            Note that centroid shrinking cannot be used with sparse matrices.
        y : array, shape = [n_samples]
            Target values (integers)
        """

        self.reduce(X, y)
        return self

    def predict(self, X, n_neighbors = 1):
        """Perform classification on an array of test vectors X.

        The predicted class C for each sample in X is returned.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples]

        Notes
        -----
        If the metric constructor parameter is "precomputed", X is assumed to
        be the distance matrix between the data to be predicted and
        ``self.centroids_``.
        """
        X = atleast2d_or_csr(X)
        if not hasattr(self, "prototypes_") or self.prototypes_ == None:
            raise AttributeError("Model has not been trained yet.")
        #return self.labels_[pairwise_distances(
        #    X, self.prototypes_, metric=self.metric).argmin(axis=1)]
        knn = KNeighborsClassifier(n_neighbors = n_neighbors)
        knn.fit(self.prototypes_, self.labels_)
        return knn.predict(X)

