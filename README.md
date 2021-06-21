# Stack_Ensemble_Classification_Customer_Behavior
Predicting whether a customer will choose to do any of several actions.
The ensemble consists of 3 base learner algorithms, which are Gradient Boosting, Linear Discriminant Analysis (LDA) and Support Vector Machine (SVM). The meta-learner algorithm is Random Forest. 
Base Learners:
Gradient Boosting was chosen since there are lots of missing values in the data set, and Gradient Boosting algorithm evaluate missing values as containing information instead of omitting or imputing them. By including Gradient Boosting in the ensemble, we get to obtain a large amount of data which might be meaningful for learning.
Gradient Boosting is trained on the entire data set. Also, the data set it is trained on is one-hot encoded.
Linear Discriminant Analysis was included because it is fast and cost effective, besides providing diversity to the ensemble.
Support Vector Machine was included due to the idea being that; in an imbalanced data set as ours, classifiers whose objective is to minimize the error can be misleading. Since mislabeling the points in the minor class would be less costly than mislabeling major class, the algorithm may be biased to choose major class over minor class. As for SVM, the objective is to minimize the distance between the points and the hyperplane, also it only considers the points which are closer to the hyperplane in each class. Therefore, bias of imbalance would be less when using SVM, as I see it.
LDA and SVM were trained on a differently pre-processed set: Categorical features and constant columns were removed; missing values were imputed. 
Meta-learner:
While training random Forest algorithm on the outputs of base learners, 5-fold cross validation was applied to estimate the performance of our algorithm. In addition, stratification was applied in cross validation, since our data set is imbalanced. 
