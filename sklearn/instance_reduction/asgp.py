# -*- coding: utf-8 -*-
"""
Self-Generating Prototypes
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
from ..decomposition import PCA
from .Group import Group

from .base import InstanceReductionMixin
from .sgp import SelfGeneratingPrototypes2

class AdaptiveSelfGeneratingPrototypes(SelfGeneratingPrototypes2):
    """Adaptive Self-Generating Prototypes

    The Adaptive Self-Generating Prototypes (ASGP) is a derivate of the
    Self-Generating Prototypes, specially designed to cope with imbalanced
    datasets. The ASGP is a centroid-based prototype generation algorithm 
    that uses the space spliting mechanism to generate prototypes in the 
    center of each cluster.

    Parameters
    ----------

    r_min: float, optional (default = 0.0)
        Determine the minimum size of a cluster [0.00, 0.20]

    r_mis: float, optional (default = 0.0)
        Determine the error tolerance before split a group


    Attributes
    ----------
    `prototypes_` : array-like, shape = [indeterminated, n_features]
        Selected prototypes.

    `labels_` : array-like, shape = [indeterminated]
        Labels of the selected prototypes.

    `reduction_` : float, percentual of reduction.

    Examples
    --------
    >>> from sklearn.instance_reduction.asgp import AdaptiveSelfGeneratingPrototypes
    >>> from sklearn.instance_reduction.sgp import SelfGeneratingPrototypes2
    >>> import numpy as np
    >>> X = np.array([[i] for i in range(1000)])
    >>> y = np.asarray(990 * [1] + 10 * [0])
    >>> asgp = AdaptiveSelfGeneratingPrototypes(r_min=0.2, r_mis=0.2)
    >>> asgp.fit(X, y)
    AdaptiveSelfGeneratingPrototypes(r_min=0.2, r_mis=0.2)
    >>> print list(set(asgp.labels_))
    [0, 1]
    >>> sgp2 = SelfGeneratingPrototypes2(r_min=0.2, r_mis=0.2)
    >>> sgp2.fit(X, y)
    SelfGeneratingPrototypes2(r_min=0.2, r_mis=0.2)
    >>> print list(set(sgp2.labels_))
    [1]

    See also
    --------
    sklearn.neighbors.KNeighborsClassifier: nearest neighbors classifier

    References
    ----------

    Dayvid V R Oliveira, Guilherme R Magalhaes, George D C Cavalcanti, and
    Tsang Ing Ren. Improved self-generating prototypes algorithm for imbalanced
    datasets. In Tools with Artificial Intelligence (ICTAI), 2012 IEEE 24th 
    International Conference on, volume 1, pages 904–909. IEEE, 2012.

    Hatem A. Fayed, Sherif R Hashem, and Amir F Atiya. Self-generating prototypes
    for pattern classification. Pattern Recognition, 40(5):1498–1509, 2007.
    """

    def __init__(self, r_min=0.0, r_mis=0.0):
        self.groups = None
        self.r_min = r_min
        self.r_mis = r_mis


    def generalization_step(self):
        #larger = max([len(g) for g in self.groups])
        labels = list(set([g.label for g in self.groups]))
        larger = {}
        for group in self.groups:
            if not group.label in larger:
                larger[group.label] = len(group)
            elif len(group) > larger[group.label]:
                larger[group.label] = len(group)

        for group in self.groups:
            if len(group) < self.r_min * larger[group.label]:
                self.groups.remove(group)

        return self.groups


    def merge(self):

        if len(self.groups) < 2:
            return self.groups

        knn = KNeighborsClassifier(n_neighbors = 2)

        merged = False
        for group in self.groups:
            reps_x = np.asarray([g.rep_x for g in self.groups])
            reps_y = np.asarray([g.label for g in self.groups])
            knn.fit(reps_x, reps_y)

            nn2_idx = knn.kneighbors(group.X, n_neighbors=2, return_distance=False)
            nn2_idx = nn2_idx.T[1]

            # could use a threshold
            if len(set(nn2_idx)) == 1 and reps_y[nn2_idx[0]] == group.label:
                ng_group = self.groups[nn2_idx[0]]
                ng2_idx = knn.kneighbors(ng_group.X, n_neighbors=2, return_distance=False)
                ng2_idx = ng2_idx.T[1]
                if len(set(ng2_idx)) == 1 and self.groups[ng2_idx[0]] == group:
                    group.add_instances(ng_group.X, update=True)
                    self.groups.remove(ng_group)
                    merged = True
                
        if merged:
            self.merge()

        return self.groups


    def pruning(self):

        if len(self.groups) < 2:
            return self.groups

        knn = KNeighborsClassifier(n_neighbors=1)
        pruned, fst = False, True
        
        while pruned or fst:
            index = 0
            pruned, fst = False, False

            while index < len(self.groups):
                group = self.groups[index]

                mask = np.ones(len(self.groups), dtype=bool)
                mask[index] = False
                reps_x = np.asarray([g.rep_x for g in self.groups])[mask]
                reps_y = np.asarray([g.label for g in self.groups])[mask]
                labels = knn.fit(reps_x, reps_y).predict(group.X)

                if (labels == group.label).all():
                    self.groups.remove(group)
                    pruned = True
                else:
                    index = index + 1

        return self.groups
            

    def reduce(self, X, y):
        X, y = check_arrays(X, y, sparse_format="csr")

        classes = np.unique(y)
        self.classes_ = classes

        minority_class = min(set(y), key = list(y).count)

        # loading inicial groups
        self.groups = []
        for label in classes:
            mask = y == label
            self.groups = self.groups + [Group(X[mask], label)]

        self.main_loop()
        self.generalization_step()
        min_groups = filter(lambda g: g.label == minority_class, self.groups)
        self.merge()
        self.pruning()
        max_groups = filter(lambda g: g.label != minority_class, self.groups)
        self.groups = min_groups + max_groups
        self.prototypes_ = np.asarray([g.rep_x for g in self.groups])
        self.labels_ = np.asarray([g.label for g in self.groups])
        self.reduction_ = 1.0 - float(len(self.labels_))/len(y)
        return self.prototypes_, self.labels_
 

