#
# Chapter 10: Unsupervised Learning
# Lab 2: Clustering
#

set.seed(2)

##################################
# K-means
##################################

# Manually adjust simulated data to fall into 2 distinct clusters
x <- matrix(rnorm(50*2), ncol=2)
x[1:25,1] <- x[1:25,1]+3
x[1:25,2] <- x[1:25,2]-4

# Perform K-means clustering with K=2, using 20 random start sets
km.out <- kmeans(x, 2, nstart=20)

# See cluster assignments per observation and plot
km.out$cluster
plot(x, col=(km.out$cluster+1), main="K-Means Clustering Results with K=2",
     xlab="", pch=20, cex=2)

set.seed(3)

# kmeans() will pick the best of all 20 runs, demonstrate improvement here
km.out <- kmeans(x, 3, nstart=1)
km.out$tot.withinss  # Within-cluster sum of squares
km.out$withinss      # Individual within-cluster sum of squares
km.out <- kmeans(x, 3, nstart=20)
km.out$tot.withinss  # Naturally smaller (better)



##################################
# Hierarchical Clustering
##################################

# Hierarchical Clustering -- Cluster observations using complete linkage
# Note: dist() computes 50x50 matrix of inter-observation Euclidean dist

hc.complete <- hclust(dist(x), method="complete")
hc.average <- hclust(dist(x), method="average")
hc.single <- hclust(dist(x), method="single")

# Plot dendograms
par(mfrow=c(1,3))
plot(hc.complete, main="Complete Linkage", xlab="", sub="", cex=0.9)
plot(hc.average, main="Average Linkage", xlab="", sub="", cex=0.9)
plot(hc.single, main="Single Linkage", xlab="", sub="", cex=0.9)

# Can actually determine cluster labels for each observation
cutree(hc.complete, 2)
cutree(hc.average, 2)
cutree(hc.single, 2) # Sucks for 2 desired clusters

# Scaled version
xsc <- scale(x)
plot(hclust(dist(xsc), method="complete"), 
     main="Hierarchical Clustering with Scaled Features")


# Cluster 3-dim dataset using correlation-based distance
x <- matrix(rnorm(30*3), ncol=3)
dd <- as.dist(1-cor(t(x)))
plot(hclust(dd, method="complete"), xlab="", sub="",
     main="Complete Linkage with Correlation-Based Distance")