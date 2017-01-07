import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp

def show_roc_curve(predictions, test_labels):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    n_classes = len(np.unique(test_labels))
    lw = 2

    #train_labels= label_binarize(train_labels, classes=['Opencountry','coast','forest','highway','inside_city','mountain','street','tallbuilding'])
    test_labels= label_binarize(test_labels, classes=['Opencountry','coast','forest','highway','inside_city','mountain','street','tallbuilding'])

    for i in range(n_classes):
        fpr[i], tpr[i],_ = roc_curve(test_labels[:,i], predictions[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'blue', 'pink'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    
    plt.plot([0,1],[0,1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('BoW ROC curve')
    plt.legend(loc="lower right")
    plt.show()