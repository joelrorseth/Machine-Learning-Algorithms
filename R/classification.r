#
# Chapter 4: Classification
# Lab: Logistic Regression, LDA, QDA, KNN
#

library(ISLR)
names(Smarket)
dim(Smarket)
summary(Smarket)

# List pairwise correlations among predictors
cor(Smarket[, -9])
#cor(Smarket)  # This wont work, "Direction" is qualitative

# Plot observed Volumes -- volume is increasing with time
attach(Smarket)
plot(Volume)


##################################
# Logistic Regression
##################################

# Binomial type Generalized Linear Model (GLM) is the logistic regression classifier
glm.fits <- glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume, data=Smarket, family=binomial)
summary(glm.fits)

# Display coefficients
coef(glm.fits)
summary(glm.fits)$coef

# Isolate p-values
summary(glm.fits)$coef[,4]

# R is treating "Up" as Y=1
contrasts(Direction)

# Output prediction probabilities in form P(Y=1|X) for first ten
# This is the type=response probability that sample is "Up"
glm.probs <- predict(glm.fits, type="response")
glm.probs[1:10]

# Transform probabilities into vector of response labels, using threshold
# Create vector of 1250 "Down"'s, map "Up"'s to glm.probs indicies
glm.pred=rep("Down", 1250)
glm.pred[glm.probs > 0.5] <- "Up"

# Confusion matrix -- This gives training error rate essentially
table(glm.pred, Direction)
mean(glm.pred == Direction)


# Try cross-validation
train <- (Year < 2005) # Boolean vector
Smarket.2005 <- Smarket[!train,] # Submatrix of Smarket with only 2005 samples
dim(Smarket.2005)
Direction.2005 <- Direction[!train]

# Predict using 2001-2004 sample subset, predict using 2005 only
glm.fits <- glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume, data=Smarket, family=binomial, subset=train)
glm.probs <- predict(glm.fits, Smarket.2005, type="response")

# Map Up to indicies where % > 0.5
glm.pred <- rep("Down", 252)
glm.pred[glm.probs > 0.5] <- "Up"

table(glm.pred, Direction.2005)
mean(glm.pred == Direction.2005)
mean(glm.pred == Direction.2005) # Test error rate


# Refit using only Lag1 and Lag2 (a little bit better)
glm.fits <- glm(Direction~Lag1+Lag2, data=Smarket, family=binomial, subset=train)
glm.probs <- predict(glm.fits, Smarket.2005, type="response")
glm.pred <- rep("Down", 252)
glm.pred[glm.probs > 0.5] <- "Up"

table(glm.pred, Direction.2005)
mean(glm.pred == Direction.2005)

# Can also predict for days which have particular Lag1 or Lag2 values
predict(glm.fits, newdata=data.frame(Lag1=c(1.2,1.5), Lag2=c(1.1,-0.8)), type="response")



##################################
# Linear Discriminant Analysis
##################################

library(MASS)
lda.fit <- lda(Direction~Lag1+Lag2, data=Smarket, subset=train)
lda.fit

# Get LDA predicted classes as a vector
lda.pred <- predict(lda.fit, Smarket.2005)
lda.class <- lda.pred$class

# Results are nearly identical to logistic regression here
table(lda.class, Direction.2005)
mean(lda.class == Direction.2005)

# How many times was LDA 90% sure about an increase happening?
sum(lda.pred$posterior[,1] > 0.9)



##################################
# Quadtratic Discriminant Analysis
##################################

# Note here, qda no longer has coefficients displayed b/c it is quadratic
qda.fit <- qda(Direction~Lag1+Lag2, data=Smarket, subset=train)
qda.fit

qda.class <- predict(qda.fit, Smarket.2005)$class
table(qda.class, Direction.2005)
mean(qda.class == Direction.2005)



##################################
# K-Nearest Neighbors
##################################

library(class) # for knn()

train.X <- cbind(Lag1, Lag2)[train,]
test.X <- cbind(Lag1, Lag2)[!train,]
train.Direction <- Direction[train]

# knn(): Train predictors, Test predictors, Train response labels, k
set.seed(1)
knn.pred <- knn(train.X, test.X, train.Direction, k=3)

table(knn.pred, Direction.2005)
mean(knn.pred == Direction.2005)



# New data set -- Insurance (yes or no buy)

dim(Caravan)
attach(Caravan)
summary(Purchase)

# Standardize dataset so mean is 0, std dev is 1
# Remember, std dev = sqrt(var)
standardized.X <- scale(Caravan[,-86])
var(Caravan[,1])
var(Caravan[,2])
var(standardized.X[,1])
var(standardized.X[,2])

test <- 1:1000
train.X <- standardized.X[-test,]
test.X <- standardized.X[test,]
train.Y <- Purchase[-test]
test.Y <- Purchase[test]
set.seed(1)

knn.pred <- knn(train.X, test.X, train.Y, k=1)
mean(test.Y != knn.pred)