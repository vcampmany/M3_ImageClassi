from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools

def show_confusion_mat(train_labels, test_labels):
    conf_mat = confusion_matrix(test_labels, train_labels)
    np.set_printoptions(precision=2)

    plt.figure()
    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    classes = np.unique(test_labels)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:,np.newaxis]

    conf_mat = np.around(conf_mat,decimals=3)

    thresh = conf_mat.max()/2
    for i,j in itertools.product(range(conf_mat.shape[0]),range(conf_mat.shape[1])):
        plt.text(j,i, conf_mat[i,j], horizontalalignment="center", color="white" if conf_mat[i,j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("true label")
    plt.xlabel("predicted label")
    plt.show()
