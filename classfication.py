# -*- coding: utf-8 -*-
from shiftpixels import *
from sklearn.datasets.mldata import fetch_mldata
mnist = fetch_mldata('mnist-original', data_home='/Users/maxim/Python AI/Hands on ML/datasets')
mnist

X, y = mnist["data"], mnist["target"]

import matplotlib
import matplotlib.pyplot as plt

some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
           interpolation="nearest")
plt.axis("off")
plt.show()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

import numpy as np

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]




y_train_5 = (y_train == 5) 
y_test_5 = (y_test == 5)


from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

sgd_clf.predict([X_train[2]])

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))  # prints 0.9502, 0.96565 and 0.96495


from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=4, scoring="accuracy")

from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)

from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)

from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")


from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0, 1])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plot_roc_curve(fpr, tpr)
plt.show()


sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])

some_digit_scores = sgd_clf.decision_function([some_digit])


from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([some_digit])



from sklearn.ensemble import RandomForestClassifier
rdf_clf= RandomForestClassifier()
rdf_clf.fit(X_train, y_train)
rdf_clf.predict([some_digit])


cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")


X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


X_train_shifted_right=shift_image(X_train, 'right', 1)
X_train_shifted_left=shift_image(X_train, 'left', 1)
X_train_shifted_top=shift_image(X_train, 'top', 1)
X_train_shifted_bottom=shift_image(X_train, 'bottom', 1)
show_image(X_train_shifted_left[1])

X_train_augmented=np.append(X_train, X_train_shifted_right, axis=0)
X_train_augmented=np.append(X_train_augmented, X_train_shifted_left, axis=0)
X_train_augmented=np.append(X_train_augmented, X_train_shifted_top, axis=0)
X_train_augmented=np.append(X_train_augmented, X_train_shifted_bottom, axis=0)


y_train= y[:60000]
y_train_append=y_train
y_train=np.append(y_train, y_train_append, axis=0)
y_train=np.append(y_train, y_train_append, axis=0)
y_train=np.append(y_train, y_train_append, axis=0)
y_train=np.append(y_train, y_train_append, axis=0)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_augmented_scaled = scaler.fit_transform(X_train_augmented.astype(np.float64))

cross_val_score(sgd_clf, X_train_augmented_scaled, y_train, cv=3, scoring="accuracy")

from sklearn.linear_model import SGDClassifier
sgd_clf_aug = SGDClassifier(random_state=42)
sgd_clf_aug.fit(X_train_augmented_scaled, y_train)

cross_val_score(sgd_clf, X_train_augmented_scaled, y_train, cv=3, scoring="accuracy")

from sklearn.neighbors import KNeighborsClassifier
Kn_clf= KNeighborsClassifier()
Kn_clf.fit(X_train_scaled, y_train)
Kn_clf.predict([some_digit])

cross_val_score(Kn_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
