#
# Chapter 9: Support Vector Machines
# Lab: Support Vector Machines
#

set.seed(1)

##################################
# Support Vector Classifier
##################################

# Generate random observations belonging to two classes
x <- matrix(rnorm(20*2), ncol=2)
y <- c(rep(-1,10), rep(1,10))

x[y==1,] <- x[y==1,] + 1
plot(x, col=(3-y))


# Encode response as a factor variable in order to perform classification
dat <- data.frame(x=x,y=as.factor(y))
library(e1071)

# Fit Support Vector Classifier
# Large cost means margins are narrow, fewer support vectors on/violating margin
svmfit <- svm(y~., data=dat, kernel="linear", cost=10, scale=FALSE)

# Plot and list support vectors
plot(svmfit, dat)
svmfit$index
summary(svmfit)

# Smaller 'cost', more support vectors b/c of wider margins
svmfit <- svm(y~., data=dat, kernel="linear", cost=0.1, scale=FALSE)
plot(svmfit, dat)
svmfit$index


# Use e1071's built in CV procedure (deafults to 10 fold CV) and test costs
set.seed(1)
tune.out <- tune(svm, y~., data=dat, kernel="linear",
                 ranges=list(cost=c(0.001,0.01,0.1,1,5,10,100)))
summary(tune.out) # cost=0.1 is best

# Get that model
bestmod <- tune.out$best.model
summary(bestmod)

# Generate more random observations for Test data
xtest <- matrix(rnorm(20*2), ncol=2)
ytest <- sample(c(-1,1), 20, rep=TRUE)
xtest[ytest==1,] <- xtest[ytest==1,] + 1
testdat <- data.frame(x=xtest, y=as.factor(ytest))

# Use optimal model to classify these test observations
ypred <- predict(bestmod, testdat)
table(predict=ypred, truth=testdat$y) # Scores 19/20, nice


# Simulate linear separability by manually separating data
x[y==1,] <- x[y==1,]+0.5
plot(x, col=(y+5)/2, pch=19)

# Data is now barely linear separable, fit support vector classifier
# Cost extremely high, allowing no misclassifications

dat <- data.frame(x=x, y=as.factor(y))
svmfit <- svm(y~., data=dat, kernel="linear", cost=1e5)
summary(svmfit)
plot(svmfit, dat)

# In comparison, smaller cost with larger margin is comparable
svmfit <- svm(y~., data=dat, kernel="linear", cost=1)
summary(svmfit)
plot(svmfit, dat)



##################################
# Support Vector Machine
##################################

set.seed(1)

# Generate data with non-linear class boundary
x <- matrix(rnorm(200*2), ncol=2)
x[1:100,] <- x[1:100,]+2
x[101:150,] <- x[101:150,]-2
y <- c(rep(1,150), rep(2,50))
dat <- data.frame(x=x, y=as.factor(y))

plot(x, col=y)

# Train radial svm on training data
train <- sample(200,100)
svmfit <- svm(y~., data=dat[train,], kernel="radial", gamma=1, cost=1)

plot(svmfit, dat[train,])
summary(svmfit)


# Use CV to determine optimal gamma (gamma=2, cost=1)
set.seed(1)
tune.out <- tune(svm, y~., data=dat[train,], kernel="radial",
                 ranges=list(cost=c(0.1,1,10,100,1000), gamma=c(0.5,1,2,3,4)))
summary(tune.out)

# And use to predict
table(true=dat[-train,"y"], pred=predict(tune.out$best.model, newdata=dat[-train,]))



##################################
# ROC Curves
##################################

library(ROCR)

# Quick function to plot vector w/ numerical score for each observation
rocplot <- function(pred, truth, ...) {
  predob <- prediction(pred, truth)
  perf <- performance(predob, "tpr", "fpr")
  plot(perf,...)
}

# Get the fitted values to predict which side of decision boundary each falls
svmfit.opt <- svm(y~., data=dat[train,], kernel="radial", 
    gamma=2, cost=1, decision.values=T)
fitted <- attributes(predict(svmfit.opt, dat[train,], 
    decision.values=TRUE))$decision.values

par(mfrow=c(1,2))
rocplot(fitted, dat[train,"y"], main="Training Data")


# Try increasing gamma, show results
svmfit.flex <- svm(y~., data=dat[train,], kernel="radial", 
    gamma=50, cost=1, decision.values=T)
fitted <- attributes(predict(svmfit.flex, dat[train,], 
    decision.values=TRUE))$decision.values

par(mfrow=c(1,2))
rocplot(fitted, dat[train,"y"], main="Training Data")


# Compare both on test data
fitted <- attributes(predict(svmfit.opt, dat[-train,], 
    decision.values=TRUE))$decision.values
rocplot(fitted, dat[-train,"y"], main="Test Data")
fitted <- attributes(predict(svmfit.flex, dat[-train,], 
    decision.values=TRUE))$decision.values
rocplot(fitted, dat[-train,"y"], add=T, col="red")



##################################
# Multiclass SVM
##################################

library(ISLR)

# Khan dataset has 2,308 predictors and only 63+20 observations
names(Khan)
table(Khan$ytrain)
table(Khan$ytest)

# Best try a linear kernel, more complex is unneccessary
dat <- data.frame(x=Khan$xtrain, y=as.factor(Khan$ytrain))
out <- svm(y~., data=dat, kernel="linear", cost=10)

# We get perfect TRAIN prediction as it is very easy for SVM to find
# fully separable hyperplanes when p > n

summary(out)
table(out$fitted, dat$y)

# Test (also near perfect prediction)
dat.te <- data.frame(x=Khan$xtest, y=as.factor(Khan$ytest))
pred.te <- predict(out, newdata=dat.te)
table(pred.te, dat.te$y)