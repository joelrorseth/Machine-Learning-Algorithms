#
# Chapter 10: Unsupervised Learning
# Lab 1: Principal Components Analysis
#

# Each observation is named in this example
states <- row.names(USArrests)
states
names(USArrests)

# Check mean and variance of each variable (predictor)
apply(USArrests, 2, mean) # 2 indicates apply to each row
apply(USArrests, 2, var)

# We must scale these b/c Assault vastly outweighs other vars
# Standardize vars to mean 0 and std dev 1

# This PCA centers vars to mean 0, scale=T standardizes std dev to 1
pr.out <- prcomp(USArrests, scale=TRUE)

names(pr.out)
pr.out$center # Mean
pr.out$scale  # Std Dev

# Principal component loadings -- min(n-1,p) yields 4 components
# Mult X by this to get principal component scores...
pr.out$rotation

# ...Alternatively, each kth col in 'x' is the kth prin. comp. score vector
pr.out$x


biplot(pr.out, scale=0)

# Equivalently (via a sign change)...
pr.out$rotation <- -pr.out$rotation
pr.out$x <- -pr.out$x
biplot(pr.out, scale=0)

# Standard deviation of each principal component
pr.out$sdev

# Variance explained by each principal component
pr.var <- pr.out$sdev^2
pr.var

# Proportion of variance explained by each principal component
pve <- pr.var/sum(pr.var)
pve

# Plot PVE and cumulative PVE
plot(pve, xlab="Principal Component", ylab="Proportion of Variance Explained",
     ylim=c(0,1), type='b')
plot(cumsum(pve), xlab="Principal Component", 
     ylab="Cumulative Proportion of Variance Explained", ylim=c(0,1), type='b')