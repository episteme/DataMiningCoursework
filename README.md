Author: Marcus Peacock

The version of the script uploaded will run linearSVC with CountVectorizer using all features and using the top 1000 features.

The code used for training all the other classifiers with all other feature
representation is included and commented out.

The script will run from the directory containing all the sgm files.

Output will appear as follows:

reut2-000.sgm
reut2-001.sgm
reut2-002.sgm
reut2-003.sgm
reut2-004.sgm
reut2-005.sgm
reut2-006.sgm
reut2-007.sgm
reut2-008.sgm
reut2-009.sgm
reut2-010.sgm
reut2-011.sgm
reut2-012.sgm
reut2-013.sgm
reut2-014.sgm
reut2-015.sgm
reut2-016.sgm
reut2-017.sgm
reut2-018.sgm
reut2-019.sgm
reut2-020.sgm
reut2-021.sgm
Gathering articles..
vectorizer:  0
classifier:  0
0.890733056708
0.913375796178
0.913375796178
0.841972299884
0.929058116503
0.930866601753
0.90027158756
0.920300056306
0.922038257515
0.868939804637
vectorizer:  1
classifier:  0
0.88969571231
0.914331210191
0.914331210191
0.844575768703
0.92595355267
0.927625201939
0.893432418113
0.919385443519
0.920930232558
0.867380698131

With the scores being, in order: accuracy, weighted recall, micro recall, macro
recall, weighted precision, micro precision, macro precision, weighted f1
score, micro f1 score and macro f1 score.
