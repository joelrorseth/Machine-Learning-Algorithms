#
# Chapter 5: Resampling Methods
# Lab: Cross-Validation and the Bootstrap
#

# Note we obtain slightly different results for different random subsets
library(ISLR)
set.seed(1)

##################################
# Validation Set Approach
##################################

# Split observations into a random subset of 196 indicies
train=sample(392,196)

# Fit to train set only
lm.fit <- lm(mpg~horsepower, data=Auto, subset=train)
attach(Auto)

# Calculate Mean Squared Error (MSE) for TEST set
# Avg(For each Yi, (Pred. Yi - Actual Yi)^2)
mean((mpg-predict(lm.fit, Auto))[-train]^2)

# Find MSE for Lin Regression w/ cubic and quadratic regresions
lm.fit2 <- lm(mpg~poly(horsepower, 2), data=Auto, subset=train)
lm.fit3 <- lm(mpg~poly(horsepower, 3), data=Auto, subset=train)
mean((mpg-predict(lm.fit2, Auto))[-train]^2)
mean((mpg-predict(lm.fit3, Auto))[-train]^2)



##################################
# Leave-One-Out Cross Validation
##################################

# Since we dont say family="binomial", this is normal linear regression
glm.fit <- glm(mpg~horsepower, data=Auto)
lm.fit <- lm(mpg~horsepower, data=Auto)

coef(glm.fit) # Same results
coef(lm.fit)  # Same results


library(boot)
glm.fit <- glm(mpg~horsepower, data=Auto)

# cv.glm() produces a list with various cross-validation results
cv.err <- cv.glm(Auto, glm.fit)
# Avg MSE across all cross-validated sets (raw and bias corrected errors given)
cv.err$delta

# Use a for-loop to try i'th order polynomial functions
cv.error <- rep(0,5) # Create vector of 0's
for (i in 1:5) {
  glm.fit <- glm(mpg~poly(horsepower, i), data=Auto)
  cv.error[i] <- cv.glm(Auto, glm.fit)$delta[1]
}

# Increasing to higher order decreases test MSE
cv.error



##################################
# k-Fold Cross Validation
##################################

set.seed(17)
cv.error.10 <- rep(0,10)

# To get k-Folds, simply specify K in CV predictor
for (i in 1:10) {
  glm.fit <- glm(mpg~poly(horsepower, i), data=Auto)
  cv.error.10[i] <- cv.glm(Auto, glm.fit, K=10)$delta[1]
}

cv.error.10



##################################
# The Bootstrap
##################################

# Define the statistic of interest
alpha.fn <- function(data, index) {
  X <- data$X[index]
  Y <- data$Y[index]
  
  # Estimate for alpha (ISLR eqn. 5.7)
  return ((var(Y) - cov(X,Y)) / (var(X) + var(Y) - 2*cov(X,Y)))
}

alpha.fn(Portfolio, 1:100)

# Randomly select 100 observations w/ replacement
set.seed(1)
alpha.fn(Portfolio, sample(100, 100, replace=T))

# Bootstrap applys fn to find alpha 1000 times, determining std dev over all
boot(Portfolio, alpha.fn, R=1000)


# Return coefficients for lin regression fit on indices specified
boot.fn <- function(data, index) {
  return (coef(lm(mpg~horsepower, data=data, subset=index)))
}

boot.fn(Auto, 1:392)
set.seed(1)

# Bootstrap can create estimates for slope & intercept by randomly sampling
boot.fn(Auto, sample(392, 392, replace=T)) # Different
boot.fn(Auto, sample(392, 392, replace=T)) # Different

# 1. Compute std err of 1000 bootstrap estimates (likely more accurate)
# SE(B0)=0.86, SE(B1)=0.0074
boot(Auto, boot.fn, 1000)

# 2. Alternatively, compute std errors for coefficients using summary
summary(lm(mpg~horsepower, data=Auto))$coef


# Try finding std err estimates from fitting quadratic lin regression model
boot.fn <- function(data, index) {
  coefficients(lm(mpg~horsepower+I(horsepower^2), data=data, subset=index))
}

set.seed(1)
boot(Auto, boot.fn, 1000)
summary(lm(mpg~horsepower+I(horsepower^2), data=Auto))$coef