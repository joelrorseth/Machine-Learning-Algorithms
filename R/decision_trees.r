#
# Chapter 8: Tree-Based Methods
# Lab: Decision Trees
#

library(tree)
library(ISLR)

attach(Carseats)
High <- ifelse(Sales <= 8, "No", "Yes") # Create bool vector
Carseats <- data.frame(Carseats, High) # Merge 'High' w/ Carseats frame


##################################
# Classification Trees
##################################

# Will predict High using all predictors except Sales
tree.carseats <- tree(High~.-Sales, Carseats)
summary(tree.carseats)

# Easy to visualize
plot(tree.carseats)
text(tree.carseats, pretty=0)

# See info about each branch, split criterion, deviance, predictions etc
tree.carseats


# (Validation Set) Test Error
set.seed(2)
train <- sample(1:nrow(Carseats), 200)

Carseats.test <- Carseats[-train,]
High.test <- High[-train]
tree.carseats <- tree(High~.-Sales, Carseats, subset=train)
tree.pred <- predict(tree.carseats, Carseats.test, type="class")

table(tree.pred, High.test)
(86+57)/200 # 71% correctly predicted


# Use Cross Validation to determine optimal tree complexity (pruning)
# Use classif error rate to guide pruning process (FUN) instead of dev
set.seed(3)
cv.carseats <- cv.tree(tree.carseats, FUN=prune.misclass)

names(cv.carseats)
cv.carseats

# Plot error rate as a function of size and k (alpha in (8.4))
par(mfrow=c(1,2))
plot(cv.carseats$size, cv.carseats$dev, type="b")
plot(cv.carseats$k, cv.carseats$dev, type="b")

# Prune tree to get the optimal 9-node tree discovered above
prune.carseats <- prune.misclass(tree.carseats, best=9)
plot(prune.carseats)
text(prune.carseats, pretty=0)

tree.pred <- predict(prune.carseats, Carseats.test, type="class")
table(tree.pred, High.test)
(94+60)/200 # 77% correctly predicted


# You can specify a specific substree size / # terminal nodes
prune.carseats <- prune.misclass(tree.carseats, best=15)
plot(prune.carseats)
text(prune.carseats, pretty=0)
tree.pred <- predict(prune.carseats, Carseats.test, type="class")
table(tree.pred, High.test)
(85+62)/200 # 74%



##################################
# Regression Trees
##################################

library(MASS)
set.seed(1)

# Fit regression tree to training data
train <-sample(1:nrow(Boston), nrow(Boston)/2)
tree.boston <- tree(medv~., Boston, subset=train)

summary(tree.boston)
plot(tree.boston)
text(tree.boston, pretty=0)

# Test using CV to see if pruning will help 
# Using all predictors (8) turns out to be optimal
cv.boston <- cv.tree(tree.boston)
plot(cv.boston$size, cv.boston$dev, type='b')

# Even tho using all is best, this is how you could prune
prune.boston <- prune.tree(tree.boston, best=5)
plot(prune.boston)
text(prune.boston, pretty=0)

# Use unpruned tree to make pred
yhat <- predict(tree.boston, newdata=Boston[-train,])
boston.test <- Boston[-train, "medv"]
plot(yhat, boston.test)
abline(0,1)
mean((yhat-boston.test)^2) # Test MSE = 25.05



##################################
# Bagging / Random Forests
##################################

library(randomForest)
set.seed(1)

# Perform Bagging, which is simply random forest w/ m=p (=13)
bag.boston <- randomForest(medv~., data=Boston, subset=train, 
                           mtry=13, importance=TRUE)
bag.boston

# Test it
yhat.bag <- predict(bag.boston, newdata=Boston[-train,])
plot(yhat.bag, boston.test)
abline(0,1)
mean((yhat.bag-boston.test)^2) # Test MSE is 13.16 (much better)

# Increase number of trees grown
bag.boston <- randomForest(medv~., data=Boston, subset=train, 
                           mtry=13, ntree=25)
yhat.bag <- predict(bag.boston, newdata=Boston[-train,])
mean((yhat.bag-boston.test)^2) # Test MSE is 13.46



# Create Random Forest (defaults to p/3 for reg. or sqrt(p) for clas.)
set.seed(1)
rf.boston <- randomForest(medv~., data=Boston, subset=train,
                          mtry=6, importance=TRUE)
yhat.rf <- predict(rf.boston, newdata=Boston[-train,])
mean((yhat.rf-boston.test)^2) # Test MSE is 11.31 (Better)

# For RF, we can see importance of each variable / predictor!
importance(rf.boston)
varImpPlot(rf.boston)



##################################
# Boosting
##################################

library(gbm)
set.seed(1)

boost.boston <- gbm(medv~., data=Boston[train,], distribution="gaussian",
                    n.trees=5000, interaction.depth=4)
summary(boost.boston)

# Visualize effect of these two (most important) vars on response
# after integrating out the other variables
par(mfrow=c(1,2))
plot(boost.boston, i="rm")
plot(boost.boston, i="lstat")

# Use mode to predict
yhat.boost <- predict(boost.boston, newdata=Boston[-train,], n.trees=5000)
mean((yhat.boost-boston.test)^2) # Test MSE = 11.8

# Can also try it with different values of lambda (shrinkage param)
boost.boston <- gbm(medv~., data=Boston[train,], distribution="gaussian",
                    n.trees=5000, interaction.depth=4, shrinkage=0.2,
                    verbose=F)
yhat.boost <- predict(boost.boston, newdata=Boston[-train,], n.trees=5000)
mean((yhat.boost-boston.test)^2) # Test MSE is 11.5