### Download Data ####
train <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.train"),header=FALSE)
test <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.test"),header=FALSE)
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
AdaB_heart <- adaboost(Diagnosis ~ . , data = train, min(optimal_m)) # 'min' for Occam's Razor (simplest model)
# Make predictions on test set
pred_heart_AdaB <- predict(AdaB_heart, test, type='response')
# Evaluate accuracy
accuracy_heart <- mean(pred_heart_AdaB$class==test$Diagnosis)
print(accuracy_heart)
# Generate the confusion matrix
conf_mat <- confusionMatrix(pred_heart_AdaB$class, as.factor(test$Diagnosis))

### SVM ####
library(e1071)
SVM_heart <- svm(Diagnosis ~ .,train)

# Make predictions on test set
pred_heart <- predict(SVM_heart, test, type='response')
pred_heart <- ifelse(pred_heart>0.5, 1, 0)
# Evaluate accuracy
accuracy_heart <- sum(pred_heart==test$Diagnosis) / length(pred_heart)
print(accuracy_heart)

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
print(accuracy_heart)

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
print(accuracy_heart)

### Logistic Regression ####
set.seed(1000)
GLM_heart <- glm(Diagnosis ~ ., family=binomial, train)

# Make predictions on test set
pred_heart <- predict(GLM_heart, test, type='response')
pred_heart <- ifelse(pred_heart>0.5, 1, 0)
# Evaluate accuracy
accuracy_heart <- sum(pred_heart==test$Diagnosis) / length(pred_heart)
print(accuracy_heart)

### KNN ####
library(class)
set.seed(1000)
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
print(accuracy_KNN)

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
print(accuracy_heart_stump)

### Random Guessing ####
set.seed(1000)
pred_rand <- runif(nrow(test), min = 0, max = 1)
pred_rand <- ifelse(pred_rand>0.5, 1, 0)
# Evaluate accuracy
accuracy_heart <- mean(pred_rand==test$Diagnosis)
print(accuracy_heart)

