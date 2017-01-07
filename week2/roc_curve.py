import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc

def show_roc_curve(train_labels, test_labels):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    n_classes = len(np.unique(test_labels))
    lw = 2
    print('n_classes:')
    print(n_classes)
    for i in range(n_classes):
        fpr[i], tpr[i],_ = roc_curve(test_labels[:,i], train_labels[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    all_fpr = np.unique(np.concatenate([fpr[i] for i in range (n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes


    colors = cycle(['aqua', 'darkorange', 'conrflowerblue', 'green', 'red', 'blue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.figure()
    plt.plot([0,1],[0,1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('BoW ROC curve')
    plt.legend(loc="lower right")
    plt.show()