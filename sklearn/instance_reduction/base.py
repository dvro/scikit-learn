"""Base and mixin classes for nearest neighbors"""
# Authors: Jake Vanderplas <vanderplas@astro.washington.edu>
#          Fabian Pedregosa <fabian.pedregosa@inria.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Sparseness support by Lars Buitinck <L.J.Buitinck@uva.nl>
#
# License: BSD 3 clause (C) INRIA, University of Amsterdam
import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.sparse import csr_matrix, issparse

from ..base import BaseEstimator
from ..metrics import pairwise_distances
from ..metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS
from ..utils import safe_asarray, atleast2d_or_csr, check_arrays
from ..utils.fixes import unique
from ..externals import six

from ..neighbors.classification import KNeighborsClassifier

class InstanceReductionWarning(UserWarning):
    pass


# Make sure that NeighborsWarning are displayed more than once
warnings.simplefilter("always", InstanceReductionWarning)

class InstanceReductionBase(six.with_metaclass(ABCMeta, BaseEstimator)):
    """Base class for instance reduction estimators."""

    @abstractmethod
    def __init__(self):
        pass


class InstanceReductionMixin(object):
    """Mixin class for all instance reduction techniques in scikit-learn"""


    @abstractmethod
    def reduce(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training set.

        y : array-like, shape = [n_samples]
            Labels for X.

        Returns
        -------
        P : array-like, shape = [indeterminated, n_features]
            Resulting training set.
        
        q : array-like, shape = [indertaminated]
            Labels for P
        """
        pass


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
        #X = atleast2d_or_csr(X)
        if not hasattr(self, "prototypes_") or self.prototypes_ == None:
            raise AttributeError("Model has not been trained yet.")
        #return self.labels_[pairwise_distances(
        #    X, self.prototypes_, metric=self.metric).argmin(axis=1)]
        knn = KNeighborsClassifier(n_neighbors = n_neighbors)
        knn.fit(self.prototypes_, self.labels_)
        return knn.predict(X)


