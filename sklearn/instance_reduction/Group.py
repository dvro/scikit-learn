import numpy as np

class Group:

    def __init__(self, X, label):
        self.X = X
        self.label = label
        if len(X) > 0:
            self.rep_x = np.mean(X, axis=0)
        else:
            self.rep = np.copy(X)

    def __add__(self, other):
        X = np.vstack((self.X, other.X))
        return Group(X, self.label, update=True)

    def __len__(self):
        length = self.X.shape[0] if self.X != None else 0
        return length

    def add_instances(self, X, update=False):
        self.X = np.vstack((self.X,X))
        if update:
            self.update_all()

    def remove_instances(self, indexes, update=False):
        _X = self.X[indexes]
        self.X = np.delete(self.X, indexes, axis=0)
        if update:
            self.update_all()
        return _X


    def update_all(self):
        self.rep_x = np.mean(self.X, axis=0)

