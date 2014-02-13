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

class SelfGeneratingPrototypes(BaseEstimator, ClassifierMixin, InstanceReductionMixin):
    """Self-Generating Prototypes

    Each class is represented by a set of prototypes, with test samples
    classified to the class with the nearest prototype.
    The Self-Generating Prototypes generates instances is a centroid-based
    prototype generation algorithm that uses the space spliting mechanism
    to generate prototypes in the center of each cluster.

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
    >>> from sklearn.instance_reduction.sgp import SelfGeneratingPrototypes
    >>> import numpy as np
    >>> X = np.array([[i] for i in range(1,13)])
    >>> X = X + np.asarray([0.1,0,-0.1,0.1,0,-0.1,0.1,-0.1,0.1,-0.1,0.1,-0.1])
    >>> y = np.array([1, 1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 1])
    >>> sgp = SelfGeneratingPrototypes()
    >>> sgp.fit(X, y)
    SelfGeneratingPrototypes(r_min=0.0, r_mis=0.0)
    >>> print sgp.reduction_
    0.25

    See also
    --------
    sklearn.neighbors.KNeighborsClassifier: nearest neighbors classifier

    References
    ----------
    Hatem A. Fayed, Sherif R Hashem, and Amir F Atiya. Self-generating prototypes
    for pattern classification. Pattern Recognition, 40(5):1498–1509, 2007.
    """

    def __init__(self, r_min=0.0, r_mis=0.0):
        self.groups = None
        self.r_min = r_min
        self.r_mis = r_mis

    def main_loop(self):
        exit_count = 0
        knn = KNeighborsClassifier(n_neighbors = 1)
        while exit_count < len(self.groups):
            index, exit_count = 0, 0
            while index < len(self.groups):

                group = self.groups[index]
                reps_x = np.asarray([g.rep_x for g in self.groups])
                reps_y = np.asarray([g.label for g in self.groups])
                knn.fit(reps_x, reps_y)
                
                nn_idx = knn.kneighbors(group.X, n_neighbors=1, return_distance=False)
                nn_idx = nn_idx.T[0]
                mask = nn_idx == index
                
                # if all are correctly classified
                if not (False in mask):
                    exit_count = exit_count + 1
                
                # if all are misclasified
                elif not (group.label in reps_y[nn_idx]):
                    pca = PCA(n_components=1)
                    pca.fit(group.X)
                    # maybe use a 'for' instead of creating array
                    d = pca.transform(reps_x[index])
                    dis = [pca.transform(inst)[0] for inst in group.X]
                    mask_split = (dis < d).flatten()
                    
                    new_X = group.X[mask_split]
                    self.groups.append(Group(new_X, group.label))
                    group.X = group.X[~mask_split]
                
                elif (reps_y[nn_idx] == group.label).all() and (nn_idx != index).any():
                    mask_mv = nn_idx != index
                    index_mv = np.asarray(range(len(group)))[mask_mv]
                    X_mv = group.remove_instances(index_mv)
                    G_mv = nn_idx[mask_mv]                        

                    for x, g in zip(X_mv, G_mv):
                        self.groups[g].add_instances([x])

                elif (reps_y[nn_idx] != group.label).sum()/float(len(group)) > self.r_mis:
                    mask_mv = reps_y[nn_idx] != group.label
                    new_X = group.X[mask_mv]
                    self.groups.append(Group(new_X, group.label))
                    group.X = group.X[~mask_mv]
                else:
                   exit_count = exit_count + 1

                if len(group) == 0:
                    self.groups.remove(group)
                else:
                    index = index + 1

                for g in self.groups:
                    g.update_all()

        return self.groups                     


    def generalization_step(self):
        larger = max([len(g) for g in self.groups])
        for group in self.groups:
            if len(group) < self.r_min * larger:
                self.groups.remove(group)
        return self.groups


    def reduce(self, X, y):
        X, y = check_arrays(X, y, sparse_format="csr")

        classes = np.unique(y)
        self.classes_ = classes

        # loading inicial groups
        self.groups = []
        for label in classes:
            mask = y == label
            self.groups = self.groups + [Group(X[mask], label)]

        self.main_loop()
        self.generalization_step()
        self.prototypes_ = np.asarray([g.rep_x for g in self.groups])
        self.labels_ = np.asarray([g.label for g in self.groups])
        self.reduction_ = 1.0 - float(len(self.labels_))/len(y)
        return self.prototypes_, self.labels_
        

class SelfGeneratingPrototypes2(SelfGeneratingPrototypes):
    """Self-Generating Prototypes 2

    The Self-Generating Prototypes 2 is the second version of the
    Self-Generating Prototypes algorithm.
    It has a higher generalization power, including the procedures
    merge and pruning.

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
    >>> from sklearn.instance_reduction.sgp import SelfGeneratingPrototypes2
    >>> import numpy as np
    >>> X = np.array([[i] for i in range(1,13)])
    >>> X = X + np.asarray([0.1,0,-0.1,0.1,0,-0.1,0.1,-0.1,0.1,-0.1,0.1,-0.1])
    >>> y = np.array([1, 1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 1])
    >>> sgp2 = SelfGeneratingPrototypes2()
    >>> sgp2.fit(X, y)
    SelfGeneratingPrototypes2(r_min=0.0, r_mis=0.0)
    >>> print sgp2.reduction_
    0.5

    See also
    --------
    sklearn.neighbors.KNeighborsClassifier: nearest neighbors classifier
    sklearn.instance_reduction.SelfGeneratingPrototypes: self-generating prototypes

    References
    ----------
    Hatem A. Fayed, Sherif R Hashem, and Amir F Atiya. Self-generating prototypes
    for pattern classification. Pattern Recognition, 40(5):1498–1509, 2007.
    """
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

        # loading inicial groups
        self.groups = []
        for label in classes:
            mask = y == label
            self.groups = self.groups + [Group(X[mask], label)]

        self.main_loop()
        self.generalization_step()
        self.merge()
        self.pruning()
        self.prototypes_ = np.asarray([g.rep_x for g in self.groups])
        self.labels_ = np.asarray([g.label for g in self.groups])
        self.reduction_ = 1.0 - float(len(self.labels_))/len(y)
        return self.prototypes_, self.labels_
 

