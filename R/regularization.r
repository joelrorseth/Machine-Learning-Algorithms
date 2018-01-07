#
# Chapter 6: Linear Model Selection and Regularization
# Lab 2: Ridge Regression and the Lasso
#

library(ISLR)
fix(Hitters)

# Remove missing these observations entirely
Hitters <- na.omit(Hitters)

# Make x -- automatically transforms qualitative vars into dummys also
x <- model.matrix(Salary~., Hitters)[,-1]
y <- Hitters$Salary


##################################
# Ridge Regression
##################################

library(glmnet)

# Create custom λ (lambda), 100 numbers over 10^10 to 10^-2
grid <- 10^seq(10, -2, length=100)

# Create model -- alpha=0 indicates ridge regression
ridge.mod <- glmnet(x, y, alpha=0, lambda=grid)

# Each λ has vector of ridge regression coefficients, thus 20x100
# Each ith column (of 100) is coefficient for that predictor on ith λ

dim(coef(ridge.mod))
ridge.mod$lambda[50] # 11498
coef(ridge.mod)[,50]

# L2 norm for λ=11498
sqrt(sum(coef(ridge.mod)[-1,50]^2))

# Smaller λ yields larger L2 norm
sqrt(sum(coef(ridge.mod)[-1,60]^2))

# We can use predict() to test on new λ(=50)
predict(ridge.mod, s=50, type="coefficients")[1:20,]


# Split up for CV
set.seed(1)
train <- sample(1:nrow(x), nrow(x)/2)
test <- (-train)
y.test <- y[test]

# Fit model on train set w/ λ=4
# newx is values for x which are used to make predictions
ridge.mod <- glmnet(x[train,], y[train], alpha=0, lambda=grid, thresh=1e-12)
ridge.pred <- predict(ridge.mod, s=4, newx=x[test,])

# Get test MSE
mean((ridge.pred-y.test)^2) # 101037

# Compare against least squares (λ=0)
ridge.pred <- predict(x=x[train,], y=y[train], ridge.mod, s=0, newx=x[test,], exact=T)
mean((ridge.pred-y.test)^2) # 114783

# Equivalently
lm(y~x, subset=train)
predict(x=x[train,], y=y[train], ridge.mod, s=0, exact=T, type="coefficients")[1:20,]



# Use CV to determine best λ
set.seed(1)
cv.out <- cv.glmnet(x[train,], y[train], alpha=0, nfolds=10)
plot(cv.out)
bestlam <- cv.out$lambda.min

bestlam # Best λ for smallest CV error is λ=212
ridge.pred <- predict(ridge.mod, s=bestlam, newx=x[test,])
mean((ridge.pred-y.test)^2) # MSE for λ=212 is 96016

# Now refit using optimal value λ=212 on entire dataset
# Note: Coefficients for best model are all non-zero
out <- glmnet(x, y, alpha=0)
predict(out, type="coefficients", s=bestlam)[1:20,]



##################################
# Lasso
##################################

# alpha=1 for Lasso
lasso.mod <- glmnet(x[train,], y[train], alpha=1, lambda=grid)

set.seed(1)
cv.out <- cv.glmnet(x[train,], y[train], alpha=1)
plot(cv.out)

# Find best lambda λ
bestlam <- cv.out$lambda.min # 16.78

lasso.pred <- predict(lasso.mod, s=bestlam, newx=x[test,])
mean((lasso.pred-y.test)^2) # Test MSE for λ=16.78 is 100743

# Refit on entire dataset
# Note: Lasso has set 12/19 coefficients to exactly 0!
out <- glmnet(x, y, alpha=1, lambda=grid)
lasso.coef <- predict(out, type="coefficients", s=bestlam)[1:20,]
lasso.coef
