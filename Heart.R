### Download Data ####
#train <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.train"),header=FALSE)
#test <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.test"),header=FALSE)
train <- read.csv(url("https://raw.githubusercontent.com/mikeizbicki/datasets/master/csv/uci/SPECT.train"),header=FALSE)
test <- read.csv(url("https://raw.githubusercontent.com/mikeizbicki/datasets/master/csv/uci/SPECT.test"),header=FALSE)
names(train)[1] = 'Diagnosis'
names(test)[1] = 'Diagnosis'


### AdaBoost ####
library(fastAdaboost)
### Use CV to find the optimal no of boosting iterations ###
data <- train
#Randomly shuffle the data
data <-data[sample(nrow(data)),]
K <- nrow(data) # k-fold cv
M = 10
#Create K equally size folds
folds <- cut(seq(1,nrow(data)),breaks=K,labels=FALSE)

#Perform K fold cross validation
acc <- integer(K)
accuracy_m <- integer(M)

for (m in 1:M){
  
  for(i in 1:K){
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    test_cv <- data[testIndexes, ]
    train_cv <- data[-testIndexes, ]
    #Use the test and train data partitions however you desire...
    
    # Run Adaboost
    AdaB_heart <- adaboost(Diagnosis ~ . , data = train_cv, m)
    # Make predictions on test set
    pred_heart_AdaB <- predict(AdaB_heart, test_cv, type='response')
   
    acc[i] <- 1 - mean(pred_heart_AdaB$class != test_cv$Diagnosis)
  }
accuracy_m[m] <- mean(acc)
}
optimal_m <- which(accuracy_m == max(accuracy_m))
optimal_m

# Run AdaBoost based on the optimal no of boosting iterations (m)
AdaB_heart <- adaboost(Diagnosis ~ . , data = train, min(optimal_m)) # 'min' for Occam's Razor
# Make predictions on test set
pred_heart_AdaB <- predict(AdaB_heart, test, type='response')
# Evaluate accuracy
accuracy_heart <- mean(pred_heart_AdaB$class==test$Diagnosis)
accuracy_heart
# Generate the confusion matrix
conf_mat <- confusionMatrix(pred_heart_AdaB$class, as.factor(test$Diagnosis))

### AdaBoost: Produce chart of train vs test error wrt boosting iterations #####

B = 110
train_error <- integer(B)
test_error <- integer(B)

for (i in 1:B){
  mod <- adaboost(Diagnosis ~ ., data = train, i)
  pred_train <- predict(mod, train, type='response')
  pred_test <- predict(mod, test, type='response')
  train_error[i] <- mean(pred_train$class!=train$Diagnosis)
  test_error[i] <- mean(pred_test$class!=test$Diagnosis)
}

B = 110
plot(1:B, train_error[1:B], cex = 0.1, 
     main='AdaBoost Train vs Test Error: SPECT Data Set',
     xlab='Boosting Iterations', ylab='Error')
points(1:B, test_error[1:B], cex = 0.1)
lines(1:B, train_error[1:B], col=3)
lines(1:B, test_error[1:B], col=4)

legend(85, 0.175, legend=c("Test error","Train error"),
       col=c("blue", "green"), lty=1:1, cex=0.8)
abline(v=7,col = "gray60")
text(7,0.065,'Optimal',cex=.8)

### SVM ####
library(e1071)
SVM_heart <- svm(Diagnosis ~ .,train)

# Make predictions on test set
pred_heart <- predict(SVM_heart, test, type='response')
pred_heart <- ifelse(pred_heart>0.5, 1, 0)
# Evaluate accuracy
accuracy_heart <- sum(pred_heart==test$Diagnosis) / length(pred_heart)
accuracy_heart

### Gradient Boosting ####
set.seed(1000)
GB_heart <- gbm(Diagnosis ~ . ,data=train, distribution = "bernoulli", bag.fraction=1,
              n.trees = 1000, shrinkage = 0.01, interaction.depth = 6, cv.folds=nrow(train), verbose=F)

best.iter = gbm.perf(GB_heart, method="cv") # uses CV to find optimal no of boosting iterations

# Make predictions on test set
pred_heart_GB <- predict(GB_heart, test, n.trees=best.iter, type='response')
pred_heart_GB <- ifelse(pred_heart_GB>0.5, 1, 0)
# Evaluate accuracy
accuracy_heart <- sum(pred_heart_GB==test$Diagnosis) / length(pred_heart_GB)
accuracy_heart

### Random Forest ####
set.seed(1000)
library(randomForest)
train_fac <- train
train_fac$Diagnosis <- as.factor(train_fac$Diagnosis)
RF_heart <- randomForest(Diagnosis ~ ., train_fac)

# Make predictions on test set
pred_heart_rf <- predict(RF_heart, test)
# Evaluate accuracy
accuracy_heart <- sum(pred_heart_rf==test$Diagnosis) / length(pred_heart_rf)
accuracy_heart

### Logistic Regression ####
set.seed(1000)
GLM_heart <- glm(Diagnosis ~ ., family=binomial, train)

# Make predictions on test set
pred_heart <- predict(GLM_heart, test, type='response')
pred_heart <- ifelse(pred_heart>0.5, 1, 0)
# Evaluate accuracy
accuracy_heart <- sum(pred_heart==test$Diagnosis) / length(pred_heart)
accuracy_heart

### KNN ####
set.seed(1000) # TIES ARE BROKEN AT RANDOM THEREFORE KNN IS NOT COMPLETELY DETERMINISTIC IN THIS CASE
library(class)
### Use CV to find the optimal no of NN ###
data <- train
#Randomly shuffle the data
data <-data[sample(nrow(data)),]
K <- nrow(data) # k-fold cv
NN = 10
#Create K equally size folds
folds <- cut(seq(1,nrow(data)),breaks=K,labels=FALSE)

#Perform K fold cross validation
acc <- integer(K)
accuracy_nn <- integer(NN)

for (nn in 1:NN){
  
  for(i in 1:K){
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    test_cv <- data[testIndexes, ]
    train_cv <- data[-testIndexes, ]
    #Use the test and train data partitions however you desire...
    
    # Run KNN
    KNN_heart <- knn(train_cv[ ,-1], test_cv[ ,-1], train_cv$Diagnosis, k=nn)

    acc[i] <- 1 - mean(KNN_heart != test_cv$Diagnosis)
  }
  accuracy_nn[nn] <- mean(acc)
}
optimal_nn <- which(accuracy_nn == max(accuracy_nn))
optimal_nn
# Run KNN with optimal NN
KNN_heart <- knn(train[ ,-1], test[ ,-1], train$Diagnosis, k=min(optimal_nn)) # min for Occam's Razor
accuracy_KNN <- mean(KNN_heart == test$Diagnosis)
accuracy

### Decision Tree Stump (maxdepth=1) ####
library(rpart)
tree_heart <- rpart(Diagnosis ~ . , data = train, method='class', maxdepth=1)
# Plot decision stump
plot(tree_heart,margin=0.2, main='Decision Tree Stump - Weak Classifier')
text(tree_heart, use.n=TRUE, cex=.8)
# Make predictions on test set
pred_heart_tree <- predict(tree_heart, test, type='class')
# Evaluate accuracy
accuracy_heart_stump <- mean(pred_heart_tree==test$Diagnosis)
accuracy_heart_stump

### Random Guessing ####
set.seed(1000)
pred_rand <- runif(nrow(test), min = 0, max = 1)
pred_rand <- ifelse(pred_rand>0.5, 1, 0)
# Evaluate accuracy
accuracy_heart <- mean(pred_rand==test$Diagnosis)
accuracy_heart




### Gradient Boosting with XGBOOST ####
# Convert data to xgb.DMatrix
dtrain <- as.matrix(train)
dtest <- as.matrix(test)

#xgb.DMatrix takes a MATRIX as an input
dtrain <- xgb.DMatrix(dtrain[ ,-1], label = dtrain[ ,1])
dtest <- xgb.DMatrix(dtest[ ,-1], label = dtest[ ,1])

set.seed(1000)
nrounds <- 100
max_depth <- 6
eta <- 0.1
subsample <- 1

# Use CV to find optimal 'nrounds' (boosting iterations)
cv.res <- xgb.cv(data = dtrain, max_depth = max_depth, eta = eta, subsample=subsample,
                 nfold=nrow(train),
                 nthread = 2, nrounds = nrounds, objective = "binary:logistic")

# Plot the train vs test error chart (cv)
plot(1:nrounds, cv.res$evaluation_log$train_error_mean, 
     cex=0.5, col=3,ylim=c(0,0.5),
     main='train v.s test error (from cv)',
     xlab='no of boosting iterations',
     ylab='error')
points(1:nrounds, cv.res$evaluation_log$test_error_mean, cex=0.5, col=4)
best_iter <- min(which(cv.res$evaluation_log$test_error_mean==min(cv.res$evaluation_log$test_error_mean)))
abline(v=best_iter, col=6)

# Train the model with best_iter (cvalidated best no of boosting iterations)
xgb_mod <- xgboost(data=dtrain, 
                   nrounds=best_iter, 
                   eta=eta, 
                   max_depth=max_depth,
                   subsample=subsample,
                   objective="binary:logistic",
                   verbose=0
)


# Make Predictions
pred_xgb <- predict(xgb_mod, dtest)
pred_xgb <- ifelse(pred_xgb>0.5, 1, 0)
err <- mean(pred_xgb != test$Diagnosis)
err

