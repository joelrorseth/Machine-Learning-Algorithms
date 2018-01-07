#
# Chapter 6: Linear Model Selection and Regularization
# Lab 3: PCR and PLS Regression
#

# Split up for CV
set.seed(1)
train <- sample(1:nrow(x), nrow(x)/2)
test <- (-train)
y.test <- y[test]


##################################
# Principal Components Regression
##################################

library(pls)
set.seed(2)

# Principal Components Regression model, scale=T allows standardizing
pcr.fit <- pcr(Salary~., data=Hitters, scale=TRUE, validation="CV")

# PCR by default shows CV scores for all possible reduction to M components
# Note: Scores are Root MSEs, so square these to get MSE
summary(pcr.fit)

# Plot CV MSE vs # of components
validationplot(pcr.fit, val.type="MSEP") # Best is M=16


# Run CV
set.seed(1)
pcr.fit <- pcr(Salary~., data=Hitters, subset=train, scale=TRUE, validation="CV")
validationplot(pcr.fit, val.type="MSEP") # Now best is M=7

pcr.pred <- predict(pcr.fit, x[test,], ncomp=7)
mean((pcr.pred-y.test)^2) # Test MSE is 96556

# Fit on entire dataset
pcr.fit <- pcr(y~x, scale=TRUE, ncomp=7)
summary(pcr.fit)



##################################
# Partial Least Squares
##################################

library(pls)
set.seed(1)

# Make PLS model
pls.fit <- plsr(Salary~., data=Hitters, subset=train, scale=TRUE, validation="CV")
summary(pls.fit) # We can see from this that optimal M=2

pls.pred <- predict(pls.fit, x[test,], ncomp=2)
mean((pls.pred-y.test)^2) # Test MSE for M=2 run is 101417

# Fit PLS on entire dataset
# Note: PLS searches for directions to explain variance in predictor & response
pls.fit <- plsr(Salary~., data=Hitters, scale=TRUE, ncomp=2)
summary(pls.fit)