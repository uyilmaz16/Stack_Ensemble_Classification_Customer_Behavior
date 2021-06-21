library(AUC)
library(onehot)
library(xgboost)
library(e1071)
library(imputeMissings)
library(h2o)
library(MASS)
library(janitor)

h2o.init()

X_train <- read.csv("hw08_training_data.csv", header = TRUE)
Y_train <- read.csv("hw08_training_label.csv", header = TRUE)
X_test <- read.csv("hw08_test_data.csv", header = TRUE)


factorcols <- sapply(X=1:ncol(X_train), function(c) sum(is.factor(X_train[,c])))
sum(factorcols)
factors <- which(factorcols == 1)

encoder <- onehot(X_train, addNA = TRUE, max_levels = Inf)
X_train_d <- predict(encoder, data = X_train)
X_test_d <- predict(encoder, data = X_test)

set.seed(2)

test_predictions <- matrix(0, nrow = nrow(X_test_d), ncol = ncol(Y_train))
colnames(test_predictions) <- colnames(Y_train)
test_predictions[,1] <- X_test[, 1]
for (outcome in 1:6) {
valid_customers <- which(is.na(Y_train[,outcome + 1]) == FALSE)

 X_nof <- X_train[valid_customers,-factors]
 yy <- Y_train[valid_customers, outcome + 1]
 
 #imputing NA's
 values <- compute(X_nof)
 X_nofm <- impute(X_nof, object = values, flag = FALSE)
 data_nofm <- data.frame(X_nofm, Y = as.factor(yy))
 
 #deleting constant columns
 data_nofm <- remove_constant(data_nofm, na.rm = FALSE)
 
 #SVM
 svmm <- svm(Y ~., data_nofm[,-1] ,  probability = TRUE, kernel = "radial")
 prdct <- predict(svmm, data_nofm, probability = TRUE)
 svm_matrix <- data.frame( V1 = as.matrix(attr(prdct,"probabilities"))[,2], Y = as.factor(yy))

 #LDA
 ldaa <- lda( Y ~ . , data_nofm[,-1], CV = TRUE)
 lda_matrix <- data.frame( X1 = ldaa[["posterior"]][,2], Y = as.factor(yy) )
 lda_1 <- lda( Y ~ . , data_nofm[,-1])
 
 #GBM
 boosting_model <- xgboost(data = X_train_d[valid_customers, -1], label = yy, nrounds = 20, objective = "binary:logistic")
 gbm_train <- predict(boosting_model, X_train_d[valid_customers, -1])
 
  #base learner outputs
  lda_train <-  as.matrix(lda_matrix[,1])
  svm_train <- as.matrix(svm_matrix[,1])
  
  data_2 <- cbind( gbm_train, svm_train, lda_train)
  
  colnames(data_2) <- c("X1", "X2", "X3")
  data_2 <- data.frame(data_2, Y = as.factor(yy))
  
  #stack training
  rf <- h2o.randomForest(x = 1:3, y = 4, as.h2o(data_2) , nfolds = 5, 
  fold_assignment = "Stratified", ntrees = 60, max_depth = 5, seed = 2)
  
  print(rf)
  
  #test preprocess
  factorcols <- sapply(X=1:ncol(X_test), function(c) sum(is.factor(X_test[,c])))
  factors <- which(factorcols == 1)
  test_nof <- X_test[,-factors]
  values <- compute(test_nof)
  test_nofc <- impute(test_nof, object = values, flag = FALSE)
  
  #lda prediction
  lda_test <- data.frame( X1 = predict(lda_1, test_nofc)[["posterior"]][,2])
  
  #svm prediction
  svm_test_matrix <- predict(svmm, test_nofc, probability = TRUE)
  svm_test <- as.matrix(attr(svm_test_matrix,"probabilities"))[,2]
  
  #gbm prediction
  gbm_test <- predict(boosting_model, X_test_d[,-1])
  
  #base learner outputs
  data_2t<- cbind(gbm_test, svm_test, lda_test)
  colnames(data_2t) <- c("X1", "X2", "X3")
  
  #stack prediction
  predictions <- as.matrix(predict(rf, as.h2o(data_2t)))[,3]
  test_predictions[, outcome + 1] <- predictions

}
write.table(test_predictions, file = "hw08_test_predictions.csv", row.names = FALSE, sep = ",")
