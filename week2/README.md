# M3_ImageClassi - Week 2

The code have been splitted in different files:
 - session2.py: main code
 - utils.py: generic functions
 - codebooks.py: functions related to codebooks. When called as main script, it pre-computes some codebooks.
 - cross_val.py: uses cross validation instead of test set.
 - create_folds.py: creates the k-fold data split, and stores it in a folder named "folds"
 - svm_kernels.py: implements the histogram intersection kernel
 - features.py: utils to extract features
 - roc_curve.py and confusion_mat.py: code to create roc curves and confusion matrices.

 The cross validation uses 5 folds which must have been created previously running "create_folds.py" and stored in "folds".

 All the codebooks computed are stored in "codebooks"
