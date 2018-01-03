#
# Chapter 3: Linear Regression
# Lab
#

library(MASS)
library(ISLR)

fix(Boston)
names(Boston)


##############################
# Linear Regression
##############################

# lm() fits a simple linear regression model:  lm(y~x, data)
# Here use medv as response, lstat as the single predictor
lm.fit = lm(medv~lstat, data=Boston)

# Output summary of the fit (second way more detailed)
lm.fit
summary(lm.fit)

# Use names() to find stored info
names(lm.fit)

# coef() is best way to extract coefficients
coef(lm.fit)

# Confidence interval for coef estimates
confint(lm.fit)

# Produce confidence / prediction intervals for some hypothetical lstat vals
predict(lm.fit, data.frame(lstat=c(5,10,15)), interval="confidence")
predict(lm.fit, data.frame(lstat=c(5,10,15)), interval="prediction")


# Plot this thing
# General: abline(<intercept>, <slope>)
attach(Boston)
plot(lstat, medv) # Y vs X
abline(lm.fit) # L.S. Regression line

# Misc formatting
#abline(lm.fit, lwd=3)
#abline(lm.fit, lwd=3, col="red")
#plot(lstat, medv, col="red")
#plot(lstat, medv, pch=20)
#plot(lstat, medv, pch="+")
#plot(1:20, 1:20, pch=1:20)


# Plot four graphs in one
par(mfrow=c(2,2))
plot(lm.fit)

# Plot residuals vs fit predictions
# Graph could indicate possible non-linearity
plot(predict(lm.fit), residuals(lm.fit))
plot(predict(lm.fit), rstudent(lm.fit))

# Use leverage stats for predictors
plot(hatvalues(lm.fit))
which.max(hatvalues(lm.fit)) # Largest leverage statistic



##############################
# Multiple Linear Regression
##############################

# General: lm(y~x1+x2+...xp, data)
# Reference all predictors using ~. shorthand
lm.fit <- lm(medv~., data=Boston)
#lm.fit <- lm(medv~lstat+age, data=Boston)
summary(lm.fit)

# Access different parts of summary object
summary(lm.fit)$r.sq
summary(lm.fit)$sigma

# VIF
library(car)
vif(lm.fit)

# Can also use shorthand to omit one or more predictors
lm.fit1 <- lm(medv~.-age, data=Boston)
# lm.fit1 <- update(lm.fit, ~.-age)
summary(lm.fit1)



##############################
# Interaction Terms
##############################

# To denote interaction terms between a and b, use a:b
# a*b includes a, b and a:b (interaction term) as predictors

summary(lm(medv~lstat*age, data=Boston))



##############################
# Non-linear transformations
##############################

# Use I() to to create higher order terms
# lstat^2 has p-score ~ 0, means it is significant
lm.fit2 <- lm(medv~lstat+I(lstat^2))
summary(lm.fit2)

# Compare linear vs non
lm.fit <- lm(medv~lstat)
anova(lm.fit, lm.fit2)

par(mfrow=c(2,2))
plot(lm.fit2)

# Create 5th order polynomial
lm.fit5 <- lm(medv~poly(lstat, 5))
summary(lm.fit5)



##############################
# Qualitative Predictors
##############################

fix(Carseats)
names(Carseats)

# R generates dummy variables automatically, will handle ShelveLoc itself
lm.fit <- lm(Sales~.+Income:Advertising+Price:Age, data=Carseats)
summary(lm.fit)

attach(Carseats)
contrasts(ShelveLoc)



LoadLibraries = function() {
  library(ISLR)
  library(MASS)
  print("Libraries have been loaded.")
}