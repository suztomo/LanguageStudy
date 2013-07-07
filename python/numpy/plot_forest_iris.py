# This source is from the tutorial
# http://scikit-learn.github.io/scikit-learn.org/dev/auto_examples/ensemble/plot_forest_iris.html 

import numpy as np
import pylab as pl

from sklearn import clone
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

import pdb
# Parameters
n_classes = 3
n_estimators = 30
plot_colors = "bry"
plot_step = 0.02
pl.set_cmap(pl.cm.Paired)

# Load data
iris = load_iris()

plot_idx = 1

for pair in ([0, 1], [0, 2], [2, 3]):
    for model in (DecisionTreeClassifier(),
                  RandomForestClassifier(n_estimators=n_estimators),
                  ExtraTreesClassifier(n_estimators=n_estimators)):
         # We only take the two corresponding features
        X = iris.data[0::2, pair]
        y = iris.target[0::2]
        X_eval = iris.data[1::2, pair]
        Y_eval = iris.target[0::2]
#        pdb.set_trace()

        # Shuffle
        idx = np.arange(X.shape[0])
        np.random.seed(13)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # Standardize
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / std

        X_eval = (X_eval - mean) / std

        # Train
        clf = clone(model)
        # X is attributes and y is the answer for training data
        clf = model.fit(X, y)

        # Plot the decision boundary
        pl.subplot(3, 3, plot_idx)

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        # arange creates 1D array from min to max by step of 0.02
        # meshgrid creats 2D array with the ranges.
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        if isinstance(model, DecisionTreeClassifier):
            # xx and yy are 2D. We need to make them 2D arrays combined as zip
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            # Z has 1D array of prediciton. To draw contour, we need to make them to 2D again
            # Z.shape = (120582,) now
            # reshaping meaing to create a copy of an array by picking up from the first element array[0,0], array[0,1] ...
            # but the dimension is different. If the array doesn't have enough elements, it throws error
            # refer http://www.mathworks.co.jp/jp/help/matlab/ref/reshape.html for reshaping matrix
            Z = Z.reshape(xx.shape)
            # Z.shape = (378, 319) now

            cs = pl.contourf(xx, yy, Z)
        else:
            for tree in model.estimators_:
                Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                cs = pl.contourf(xx, yy, Z, alpha=0.1)

        Z = model.predict(X_eval)
        success, failure = 0, 0
        for (i, c) in enumerate(Y_eval):
            if Z[i] == Y_eval[i]:
                success += 1
            else:
                failure += 1
        print "%s : Accuracy %f (%d/%d)" % (str(model), success * 1.0 / (success+failure), success, success+failure)


        #pl.xlabel("%s / %s" % (iris.feature_names[pair[0]],
        #                       model.__class__.__name__))
        #pl.ylabel(iris.feature_names[pair[1]])
        pl.axis("tight")

        # Plot the training points
        for i, c in zip(xrange(n_classes), plot_colors):
            idx = np.where(Y_eval == i)
            pl.scatter(X_eval[idx, 0], X_eval[idx, 1], c=c, label=iris.target_names[i])
            
        pl.axis("tight")

        plot_idx += 1

pl.suptitle("Decision surfaces of a decision tree, of a random forest, and of "
            "an extra-trees classifier")
pl.show()
