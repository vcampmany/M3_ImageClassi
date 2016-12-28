# M3_ImageClassi

Python + common libraries (tested on 2.7.6)
OpenCV (tested on 2.4.11)
Numpy (tested on 1.11.2)
sklearn (tested on 0.16.1)

Download the dataset an place it into the root folder of the project with the name "MIT_split".
MIT_split/train and MIT_split/test

## Week 2
The code for week 2 is inside the folder 'week2'.
Data paths are relative to the root folder. 

The code have been separated in different files:
 - session2.py: main code
 - utils.py: generic functions
 - codebooks.py: functions related to codebooks. When called as main script, it pre-computes some codebooks.
 - cross_val.py: uses cross validation instead of test set.

 The cross validation uses 5 folds which has been created previously and stored in "week2/folds".

 All the codebooks computed are stored in "week2/codebooks"
