#
# Chapter 6: Linear Model Selection and Regularization
# Lab 1: Subset Selection Methods
#

library(ISLR)
fix(Hitters)
names(Hitters)

# There are 59 missing salary stats
dim(Hitters)
sum(is.na(Hitters$Salary))

# Remove these observations entirely
Hitters <- na.omit(Hitters)
dim(Hitters)
sum(is.na(Hitters))


##################################
# Best Subset Selection
##################################

library(leaps)

# Uses RSS to determine best predictor combos for each 
# possible number of predictors (in this case 1 up to 8)
regfit.full <- regsubsets(Salary~., Hitters)

# See picks
summary(regfit.full)

# Find 19 subsets
regfit.full <- regsubsets(Salary~., Hitters, nvmax=19)
reg.summary <- summary(regfit.full)

# We can look at several measures given here
names(reg.summary)
reg.summary$rsq # Note this increases linearly as it should for train set


# Plot RSS and adjusted R^2
par(mfrow=c(2,2))
plot(reg.summary$rss, xlab="Number of Variables", ylab="RSS", type="l")
plot(reg.summary$adjr2, xlab="Number of Variables", ylab="Adjusted RSq", type="l")

# Add max point onto existing plot
which.max(reg.summary$adjr2) # 11
points(11, reg.summary$adjr2[11], col="red", cex=2, pch=20)

# Plot Cp and BIC
plot(reg.summary$cp, xlab="Number of Variables", ylab="Cp", type='l')
which.min(reg.summary$cp) # 10
points(10,reg.summary$cp[10], col="red", cex=2, pch=20)
which.min(reg.summary$bic) # 6
plot(reg.summary$bic, xlab="Number of Variables", ylab="BIC", type='l')
points(6,reg.summary$bic[6], col="red", cex=2, pch=20)

# regsubsets has built in plot fn
# Top row is optimal (by its corresponding scale), black boxes indicate selected
plot(regfit.full, scale="r2")
plot(regfit.full, scale="adjr2")
plot(regfit.full, scale="Cp")
plot(regfit.full, scale="bic")



##################################
# Fwd/Bwd Stepwise Selection
##################################

regfit.fwd <- regsubsets(Salary~., data=Hitters, nvmax=19, method="forward")
regfit.bwd <- regsubsets(Salary~., data=Hitters, nvmax=19, method="backward")
summary(regfit.fwd)
summary(regfit.bwd)



# Cross Validation Approach
set.seed(1)
train <- sample(c(TRUE,FALSE), nrow(Hitters), rep=TRUE)
test <- (!train)

regfit.best <- regsubsets(Salary~., data=Hitters[train,], nvmax=19)

# Build a fresh X matrix using only test
test.mat <- model.matrix(Salary~., data=Hitters[test,])

# Find MSE for each subset containing i most optimal predictors
val.errors <- rep(NA, 19)
for (i in 1:19) {
  coefi <- coef(regfit.best, id=i)
  pred <- test.mat[, names(coefi)]%*%coefi
  val.errors[i] <- mean((Hitters$Salary[test] - pred)^2) # Test MSE
}

which.min(val.errors) # Min is index (p=) 10
coef(regfit.best, 10)

# That kinda sucked, there is no predict() for regsubsets
# Here is one based on what we do above
predict.regsubsets <- function(object, newdata, id, ...) {
  
  form <- as.formula(object$call[[2]])
  mat <- model.matrix(form, newdata)
  coefi <- coef(object, id=id)
  xvars <- names(coefi)
  mat[, xvars]%*%coefi
}

# Now we know 10 is optimal, find best 10 from ENTIRE ORIGINAL SET
regfit.best <- regsubsets(Salary~., data=Hitters, nvmax=19)
coef(regfit.best, 10)



k <- 10
set.seed(1)

# Vector to store each observation to one of k=10 folds, matrix for results
folds <- sample(1:k, nrow(Hitters), replace=TRUE)
cv.errors <- matrix(NA, k, 19, dimnames=list(NULL, paste(1:19)))

# Perform CV
# For each fold in 10-fold CV...
for (j in 1:k) {
  
  best.fit <- regsubsets(Salary~., data=Hitters[folds!=j,], nvmax=19)
  
  # For each predictor...
  for (i in 1:19) {
    # Predict from test set eg. folds==j
    pred <- predict(best.fit, Hitters[folds==j,], id=i)
    cv.errors[j,i] <- mean((Hitters$Salary[folds==j] - pred)^2)
  }
}

# cv.errors is 10x19, where (i,j)th element is test MSE for ith CV
# fold for the best j-variable model

# apply() will average over all columns
# jth element is CV error for j-variable model
mean.cv.errors <- apply(cv.errors, 2, mean)
mean.cv.errors

par(mfrow=c(1,1))
plot(mean.cv.errors, type='b') # Best is now using 11 predictors


# Now get final, CV 11-variable model found using 
# subset selection on full dataset
reg.best <- regsubsets(Salary~., data=Hitters, nvmax=19)
coef(reg.best, 11)
