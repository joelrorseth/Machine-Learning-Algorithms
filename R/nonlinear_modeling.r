#
# Chapter 7: Moving Beyond Linearity
# Lab: Non-Linear Modeling
#

library(ISLR)
attach(Wage)


##################################
# Poly Regression & Step Functions
##################################

# Fit linear model using deg. 4 age polynomial
# Each column yielded from poly() is linear combo of age,age^2,age^3,age^4
fit <- lm(wage~poly(age,4), data=Wage)
coef(summary(fit))

# Setting raw=T, poly() yeilds age,age^2...etc directly, not lin combination
fit2 <- lm(wage~poly(age,4,raw=T), data=Wage)
coef(summary(fit2))

# Create polynomial basis functions on the fly
fit2a <- lm(wage~age+I(age^2)+I(age^3)+I(age^3)+I(age^4), data=Wage)
coef(summary(fit2a))

# Alternatively...
fit2b <- lm(wage~cbind(age,age^2,age^3,age^4), data=Wage)



# Create grid for ages at which we want predictions
# Fit linear model to 'age'
agelims <- range(age) # 18 80
age.grid <- seq(from=agelims[1], to=agelims[2]) # 18 19 ... 79 80

# Predict, using fit, the wage for ages 18-80
preds <- predict(fit, newdata=list(age=age.grid), se=TRUE)

# Get std errors (points) to plot
se.bands <- cbind(preds$fit + 2*preds$se.fit, preds$fit - 2*preds$se.fit)

# Plot age vs wage, linear fitted line of 63 values, SE's for said values
par(mfrow=c(1,2), mar=c(4.5,4.5,1,1), oma=c(0,0,4,0))
plot(age, wage, xlim=agelims, cex=0.5, col="darkgrey")
title("Degree-4 Polynomial", outer=T)
lines(age.grid, preds$fit, lwd=2, col="blue")
matlines(age.grid, se.bands, lwd=1, col="blue", lty=3)


# Create deg. 1 thru 5 polynomial regression models to test
fit.1 <- lm(wage~age, data=Wage)
fit.2 <- lm(wage~poly(age,2), data=Wage)
fit.3 <- lm(wage~poly(age,3), data=Wage)
fit.4 <- lm(wage~poly(age,4), data=Wage)
fit.5 <- lm(wage~poly(age,5), data=Wage)

# Perform analysis of variance -- sequentially compare increasingly complex models
# Test null hyp: Model M1 is sufficient to explain data vs more complex models
# Here M3 & M4 share the optimal p-val of 0.05, so cubic or quadratic is best

anova(fit.1,fit.2,fit.3,fit.4,fit.5)

# Equivalently... (because poly() creates orthogonal polynomials)
coef(summary(fit.5))

# ANOVA is better tho, can compare non-orthogonal polynomial based models
fit.1 <- lm(wage~education+age, data=Wage)
fit.2 <- lm(wage~education+poly(age,2), data=Wage)
fit.3 <- lm(wage~education+poly(age,3), data=Wage)
anova(fit.1,fit.2,fit.3)



# Predict if someone makes >$250k/yr using polynomial logistic regression
fit <- glm(I(wage>250)~poly(age,4), data=Wage, family=binomial)
preds <- predict(fit, newdata=list(age=age.grid), se=T)

# Manually obtain confidence intervals for Pr(Y=1|X) by transforming the logit
pfit <- exp(preds$fit)/(1+exp(preds$fit))
se.bands.logit <- cbind(preds$fit + 2*preds$se.fit, preds$fit - 2*preds$se.fit)
se.bands <- exp(se.bands.logit)/(1+exp(se.bands.logit))

# Equivalently... (except this would yield negative probabilities)
# preds <- predict(fit, newdata=list(age=age.grid), type="response", se=T)

plot(age, I(wage>250), xlim=agelims, type="n", ylim=c(0,.2))
points(jitter(age), I((wage>250)/5), cex=0.5, pch="|", col="darkgrey")
lines(age.grid, pfit, lwd=2, col="blue")
matlines(age.grid, se.bands, lwd=1, col="blue", lty=3)



# Fit a step function
table(cut(age,4))
fit <- lm(wage~cut(age,4), data=Wage)
coef(summary(fit))



##################################
# Splines
##################################

library(splines)

# Fit a regression spline (defaults to cubic spline)
fit <- lm(wage~bs(age, knots=c(25,40,60)), data=Wage)
pred <- predict(fit, newdata=list(age=age.grid), se=T)

# bs() generates matrix of basis functions for splines w/ specified knots
# Spline w/ 6 basis fn's is produced

plot(age, wage, col="gray")
lines(age.grid, pred$fit, lwd=2)
lines(age.grid, pred$fit + 2*pred$se, lty="dashed")
lines(age.grid, pred$fit - 2*pred$se, lty="dashed")

dim(bs(age, knots=c(25,40,60)))
dim(bs(age, df=6)) # Produce knots at uniform quantiles
attr(bs(age,df=6), "knots")


# Fit a natural spline w/ 4 degrees of freedom
fit2 <- lm(wage~ns(age,df=4), data=Wage)
pred2 <- predict(fit2, newdata=list(age=age.grid), se=T)
lines(age.grid, pred2$fit, col="red", lwd=2)


# Fit smoothing spline
plot(age, wage, xlim=agelims, cex=0.5, col="darkgrey")
title("Smoothing Spline")
fit <- smooth.spline(age, wage, df=16)
fit2 <- smooth.spline(age, wage, cv=TRUE)
fit2$df # CV determines optimal df is 6.8

lines(fit, col="red", lwd=2)
lines(fit2, col="blue", lwd=2)
legend("topright", legend=c("16 DF", "6.8 DF"), col=c("red","blue"), lty=1, lwd=2, cex=0.8)


# Perform local regression using spans of 0.2 and 0.5
# Note how higher span is more generalized, thus smoother

plot(age, wage, xlim=agelims, cex=0.5, col="darkgrey")
title("Local Regression")
fit <- loess(wage~age, span=0.2, data=Wage)
fit2 <- loess(wage~age, span=0.5, data=Wage)
lines(age.grid, predict(fit, data.frame(age=age.grid)), col="red", lwd=2)
lines(age.grid, predict(fit2, data.frame(age=age.grid)), col="blue", lwd=2)
legend("topright", legend=c("Span=0.2", "Span=0.5"), col=c("red","blue"), lty=1, lwd=2, cex=0.8)



##################################
# Generalized Additive Models
##################################

# Use GAM to predict wage using natural spline (ns())
gam1 <- lm(wage~ns(year,4)+ns(age,5)+education, data=Wage)

library(gam)

# Use s() to generally fit GAM using smoothing splines
# Here, we specify year has 4 df, age has 5 df etc.
gam.m3 <- gam(wage~s(year,4)+s(age,5)+education, data=Wage)
par(mfrow=c(1,3))
plot(gam.m3, se=TRUE, col="blue")

# Or...
plot.gam(gam1, se=TRUE, col="red")

# Compare against some other variations -- M2 is preffered
gam.m1 <- gam(wage~s(age,5)+education, data=Wage)
gam.m2 <- gam(wage~year+s(age,5)+education, data=Wage)
anova(gam.m1, gam.m2, gam.m3, test="F")

summary(gam.m3)


# Make predictions
preds <- predict(gam.m2, newdata=Wage)

# Use local regression fits as GAM building blocks
gam.lo <- gam(wage~s(year,df=4)+lo(age,span=0.7)+education, data=Wage)
plot.gam(gam.lo, se=TRUE, col="green")

# Fit 2 term model, 1st term is an interaction between year & age
gam.lo.i <- gam(wage~lo(year,age,span=0.5)+education, data=Wage)

library(akima)
plot(gam.lo.i)

# Fit logistic regression GAM, use I() to construct binary response var
gam.lr <- gam(I(wage>250)~year+s(age,df=5)+education, family=binomial, data=Wage)
par(mfrow=c(1,3))
plot(gam.lr, se=T, col="green")
table(education, I(wage>250))

# Nobody in < HS Grad earns >250k/year, eliminate this and refit
gam.lr.s <- gam(I(wage>250)~year+s(age,df=5)+education, family=binomial, data=Wage,
                subset=(education!="1. < HS Grad"))
plot(gam.lr.s, se=T, col="green")