Case Study: Diabetes Prediction
================

## Bayesian Data Analysis

## Paula Chiaramonte

## Pima Indians Diabetes Database

This [*Pima Indians Diabetes
Dataset*](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
is originally from the National Institute of Diabetes and Digestive and
Kidney Diseases. The objective of the dataset is to diagnostically
predict whether or not a patient has diabetes, based on certain
diagnostic measurements included in the dataset.

### Content

The datasets consists of several medical predictor variables and one
target variable, Outcome. Predictor variables includes the number of
pregnancies the patient has had, their BMI, insulin level, age, and so
on.

## Objective

Obtain a Bayesian inference of Diabetes using the Glucose values and
implement a Logistic Bayesian Model for calculating the probability of a
patient having Diabetes as a function of the different diagnostic
measurements of each person in the dataset.

![Diabetes](https://res.cloudinary.com/grohealth/image/upload/c_fill,f_auto,fl_lossy,h_650,q_auto,w_1085/v1581695681/DCUK/Content/causes-of-diabetes.png)

## Data Exploration of the Data

``` r
data = read.csv("diabetes.csv")
attach(data)
head(data)
```

    ##   Pregnancies Glucose BloodPressure SkinThickness Insulin  BMI
    ## 1           6     148            72            35       0 33.6
    ## 2           1      85            66            29       0 26.6
    ## 3           8     183            64             0       0 23.3
    ## 4           1      89            66            23      94 28.1
    ## 5           0     137            40            35     168 43.1
    ## 6           5     116            74             0       0 25.6
    ##   DiabetesPedigreeFunction Age Outcome
    ## 1                    0.627  50       1
    ## 2                    0.351  31       0
    ## 3                    0.672  32       1
    ## 4                    0.167  21       0
    ## 5                    2.288  33       1
    ## 6                    0.201  30       0

``` r
summary(data)
```

    ##   Pregnancies        Glucose      BloodPressure    SkinThickness  
    ##  Min.   : 0.000   Min.   :  0.0   Min.   :  0.00   Min.   : 0.00  
    ##  1st Qu.: 1.000   1st Qu.: 99.0   1st Qu.: 62.00   1st Qu.: 0.00  
    ##  Median : 3.000   Median :117.0   Median : 72.00   Median :23.00  
    ##  Mean   : 3.845   Mean   :120.9   Mean   : 69.11   Mean   :20.54  
    ##  3rd Qu.: 6.000   3rd Qu.:140.2   3rd Qu.: 80.00   3rd Qu.:32.00  
    ##  Max.   :17.000   Max.   :199.0   Max.   :122.00   Max.   :99.00  
    ##     Insulin           BMI        DiabetesPedigreeFunction      Age       
    ##  Min.   :  0.0   Min.   : 0.00   Min.   :0.0780           Min.   :21.00  
    ##  1st Qu.:  0.0   1st Qu.:27.30   1st Qu.:0.2437           1st Qu.:24.00  
    ##  Median : 30.5   Median :32.00   Median :0.3725           Median :29.00  
    ##  Mean   : 79.8   Mean   :31.99   Mean   :0.4719           Mean   :33.24  
    ##  3rd Qu.:127.2   3rd Qu.:36.60   3rd Qu.:0.6262           3rd Qu.:41.00  
    ##  Max.   :846.0   Max.   :67.10   Max.   :2.4200           Max.   :81.00  
    ##     Outcome     
    ##  Min.   :0.000  
    ##  1st Qu.:0.000  
    ##  Median :0.000  
    ##  Mean   :0.349  
    ##  3rd Qu.:1.000  
    ##  Max.   :1.000

The dataset contains:

  - `Pregnancies`: shows the number of times the patient had been
    pregnant

  - `Glucose`: shows the concentration of Glucose after 2 hours of oral
    glucose concentration in the glucose tolerance test (mg/dL)

  - `BloodPressure`: Diastolic blood pressure (mm Hg)

  - `SkinThickness`: Triceps skin fold thickness (mm)

  - `Insulin`: 2-Hour serum insulin (mu U/ml)

  - `BMI`: Body mass index

  - `DiabetesPedigreeFunction`: The diabetes Pedigree Function

  - `Age`: Age in years

  - `Outcome`: 1 if has diabetes and 0 if not

<!-- end list -->

``` r
X = data[,-9]
y = as.factor(data$Outcome)
y.bin = ifelse(data$Outcome == 1, 1, 0)

VARS = names(X)
VARS
```

    ## [1] "Pregnancies"              "Glucose"                 
    ## [3] "BloodPressure"            "SkinThickness"           
    ## [5] "Insulin"                  "BMI"                     
    ## [7] "DiabetesPedigreeFunction" "Age"

We can plot the distribution of all of the variables

``` r
par(mfrow=c(4,2))
par(mar = rep(2, 4))
for( i in 1:8){
  hist(X[,i], main = colnames(X)[i],xlab =     colnames(X)[i], col = 'blue')
}
```

![](DiabetesCaseStudy_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

## Bayesian Normal Model

As a first approach, we can observe that the Glucose kind of follows a
similar distribution to a Normal distribution

``` r
hist(Glucose, breaks = 20)
```

![](DiabetesCaseStudy_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

But if we separate the Glucose of the people with Diabetes and the
people with no diabetes, we can observe that the Glucose of people with
no Diabetes follows a Normal Distribution, and is kind of separable from
the Glucose distribution of diabetes patients

``` r
par(mfrow=c(1,1))

hist(Glucose[y==0], col=rgb(0,0,1,1/4), xlim=c(0,max(Glucose)), ylim=c(0,120), breaks = 20)
hist(Glucose[y==1], col=rgb(1,0,0,1/4), add=T, breaks=20)
```

![](DiabetesCaseStudy_files/figure-gfm/figures-side-1.png)<!-- -->

We know that Glucose is used to detect diabetic people, so we can use
this knowledge to create a Bayesian Model.

With the assumption that the Glucose follows a Normal Distribution, we
can obtain a predictive distribution of the Glucose levels of a non
diabetic person, in order to evaluate extreme values of Glucose to
detect anomalies, that can be classified as people with possible
Diabetes.

### Bayesian Prediction of Glucose levels of non-diabetics patients

We will use a logarithm transformation in order to create a more Normal
distribution:

``` r
hist(log(Glucose[y==0]), breaks=20, main="Histogram of Log(Glucose) Non-Diabetic People")
```

![](DiabetesCaseStudy_files/figure-gfm/unnamed-chunk-6-1.png)<!-- --> We
will assume that the Glucose, $X$, follows a lognormal distribution:
$$X\mid\mu, \tau\sim \text{Lognormal} \left(\mu,\frac{1}{\tau}\right)$$
and we assume the conjugate prior distribution for the normal
distribution, that is, a normal-gamma prior distribution:
$$\mu\mid\tau\sim \text{Normal}\left(m,\frac{1}{c\tau}\right)$$
$$\tau\sim\text{Gamma}\left(\frac{a}{2},\frac{b}{2}\right)$$

We set the prior parameters.We can use a non informative prior:

``` r
m=0; c=0.01; a=0.01; b=0.01;
```
<p>

Given the Glucose $X_i$ for i in $1, ..., n$, the
posterior is equal to:
$$\mu \mid \tau,X \sim \text{Normal} (m^*,\frac{1}{c^*\tau})
$$\tau \mid \text{data} \sim \text{Gamma}(\frac{a^*}{2},\frac{b^*}{2})$$
where, 
    $$m^{\ast}   =\frac{c m+n\bar{x}}{c+n},\] \[c^{\ast}=c+n,$$
    $$a^{\ast}   =a+n,$$
    $$b^{\ast}   =b+\left(  n-1\right)  s^{2}+\frac{c n}{c+n}(m-\bar{x})^2$$

Calculate the observed data values:

``` r
log.Gluc = log(Glucose[y==0][Glucose[y==0] > 0])
n=length(log.Gluc)
mean.gluc=mean(log.Gluc)
var.gluc=var(log.Gluc)
```

Compute the posterior values:

``` r
m.ast=(c*m+n*mean.gluc)/(c+n)
c.ast=c+n
a.ast=a+n
b.ast=b+(n-1)*var.gluc+c*n*(m-mean.gluc)^2/(c+n)
```

We can compare the prior and the posterior joint distribution for the
normal paremeters

``` r
library(nclbayes)
mu=seq(4.5,5,len=1000)
tau=seq(17,25,len=1000)
NGacontour(mu,tau,m,c,a,b)
NGacontour(mu,tau,m.ast,c.ast,a.ast,b.ast,add=TRUE,col="red")
legend("topright",c("prior","posterior"),lty=c(1,1),col=c("black","red"))
```

![](DiabetesCaseStudy_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

We obtained the join posterior distribution of the Gaussian parameters
is:
$$\mu\mid\tau,\text{data}\sim\text{Normal}\left( 4.68,\frac{1}{497.01\tau}\right)$$

$$\tau\mid\text{data}\sim\text{Gamma}\left(\frac{ 497.01}{2},\frac{24.754}{2}\right)$$

We can use the Monte Carlo sampling, we may obtain a sample of size
M=10000 from this join distribution:

``` r
M=10000
tau.post=rgamma(M,shape=a.ast/2,rate=b.ast/2)
mu.post=rnorm(M,mean=m.ast,sd=sqrt(1/(c.ast*tau.post)))
plot(mu.post,tau.post,col="darkgrey")
NGacontour(mu,tau,m.ast,c.ast,a.ast,b.ast,col="red",add=T)
```

![](DiabetesCaseStudy_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

With this sample from the joint posterior distribution we can obtain the
95% credible interval for $\mu$:

``` r
quantile(mu.post,c(0.025,0.975))
```

    ##     2.5%    97.5% 
    ## 4.662257 4.701377

And also a 95% credible interval for \(\tau\):

``` r
quantile(tau.post,c(0.025,0.975))
```

    ##     2.5%    97.5% 
    ## 17.62077 22.63358

### Predictive Probabilities

Supposing we want to estimate the predictive probability that a future
value of a Glucose $X_{n+1}$ is larger than a value $Y_i$ of
glucose, given the observed Glucose values:

$$\Pr\left(X_{n+1}\gt log(Y_i)\mid \text{data}\right)$$

If we assume a lognormal-gamma for ($\mu, \tau$), the predictive
density is ascaled, shifted Student distribution:

$$\log(X_{n+1})\mid\text{data}\sim\text{SS-Student}\left(m^*,\frac{(c^*+1)b^*}{c^*a^*},a^*\right)$$

We can obtain an approximation of the predictive density and compare it
with the observed data of log-Glucose:

``` r
hist(log.Gluc,freq=F, breaks=20)
x.axis=seq(0,200,by=0.01)
pred.mean=m.ast
scale=(c.ast+1)*b.ast/(c.ast*a.ast)
lines(x.axis,dt((x.axis-pred.mean)/sqrt(scale),a.ast)/sqrt(scale))
```

![](DiabetesCaseStudy_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->
The distribution samples really good the observed data

We can plot it with the observed data of the log-Glucose of Diabetic
people:

``` r
hist(log.Gluc,freq=F, breaks=20)
hist(log(Glucose[y==1][Glucose[y==1] > 0]), freq=F, breaks=20, add=T, col=rgb(1,0,0,1/4))
x.axis=seq(0,200,by=0.01)
pred.mean=m.ast
scale=(c.ast+1)*b.ast/(c.ast*a.ast)
lines(x.axis,dt((x.axis-pred.mean)/sqrt(scale),a.ast)/sqrt(scale))
```

![](DiabetesCaseStudy_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

We can compare the empirical cds with the predictive cdf:

``` r
plot(ecdf(log.Gluc))
plot(ecdf(log(Glucose[y==1][Glucose[y==1] > 0])), add=T, col='red')
lines(x.axis,pt((x.axis-pred.mean)/sqrt(scale),a.ast), col='blue')
```

![](DiabetesCaseStudy_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

We can obtain the predictive probability that a non-diabetic person has
a Glucose level higher than 160:

$$\Pr(X_{n+1}\gt;log(160)\mid\mathbf{x})=\Pr\left(t_{a^*}\gt;\frac{4-m^*}{\sqrt{\frac{(c^*+1)b^*}{c^*a^*}}} \right)$$

``` r
1-pt((log(160)-pred.mean)/sqrt(scale),a.ast)
```

    ## [1] 0.03938783

Is really low, which means that is strange for a Non-diabetic person to
have a Glucose level (in the tolerance test) higher than 160, which can
be an extreme values (which can be related to a Diabetes value).

We can obtain a sample of predictive glucose values, by applying the
exponent of the posterior mean and posterior standard deviation of the
lognormal distribution:

``` r
M=10000
glu.pred=rnorm(M,mean=mu.post, sd=1/sqrt(tau.post))
hist(glu.pred, breaks=20 , main="Predictive distribution of log-Glucose")
```

![](DiabetesCaseStudy_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

We can obtain the predictive distributions, with the real values of the
glucose, by applying the exponent:

``` r
M=10000
glu.pred_exp=exp(rnorm(M,mean=mu.post, sd=1/sqrt(tau.post)))
hist(glu.pred_exp, breaks=20 , main="Predictive distribution of Glucose")
```

![](DiabetesCaseStudy_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->

In order to obtain the real values of the Glucose, and not the
logarithm, we apply the exponent:

We can observe that the mean value of the Glucose for Non-Diabetic
people is:

``` r
exp(mean(glu.pred))
```

    ## [1] 107.8523

We can interpretated the predictive intervals as the values that are
inside the range of normal values for Non-diabetic people, which means
that if the glucose is higher than those values, then the patient may
have diabetes.

If we choose a predictive interval of 5%, the normal values of the
glucose are between:

``` r
exp(quantile(glu.pred, c(0.025, 0.975)))
```

    ##      2.5%     97.5% 
    ##  69.87283 166.38286

If we choose a right confidence interval of 5% and 10%, the normal
values of glucose are between:

``` r
exp(quantile(glu.pred, c(0.95)))
```

    ##      95% 
    ## 155.9562

``` r
exp(quantile(glu.pred, c(0.9)))
```

    ##      90% 
    ## 143.7913

And the predictive probability of a non diabetic patient having a
glucose value higher than 160 is:

``` r
mean(glu.pred > log(160))
```

    ## [1] 0.0365

![Glucose
Chart](https://www.cdc.gov/diabetes/images/basics/CDC_Diabetes_Social_Ad_Concept_A2_Facebook.png)

We can compare with the medical values, and observe that the outliers,
or not significant values in the Predictive Distribution correspond to
values of the “Glucose Tolerance Levels” to not normal values, but
values related to pre-diabetes or diabetes. So we can say that our
inference makes sense.

## Bayesian Logistic Regression

We are going to plot the boxplots for each variable in order to observe
how are different values for Diabetic people and Non-diabetic people

``` r
par(mfrow=c(2,2))
par(mar = rep(2, 4))
par(mfrow = c(2,2))
for( i in 1:8){
  boxplot(X[,i]~y, main = names(X)[i])
}
```

![](DiabetesCaseStudy_files/figure-gfm/unnamed-chunk-24-1.png)<!-- -->![](DiabetesCaseStudy_files/figure-gfm/unnamed-chunk-24-2.png)<!-- -->

We can observe that when a person has higher Glucose levels it is more
associated with Diabetes. So using this variable we can create a first
approach of an Uni-variate Bayesian Logistic Regression of the
probability of failure, explained by the Glucose levels of a persona

### Uni-variate Bayesian Logistic Regression

We assume a logistic regression for this problem:

We set $y_i =1$ to a person $i$ with Diabetes and $y_i =0$ if a
person $i$ doesn’t have Diabetes:
$$y_i\mid p_i\sim\text{Bernoulli}(p_i)$$ where $p_i$ denotes the
diabetes probability of the person \(i\), such that,
$$\log\frac{p_i}{1-p_i}= \alpha+\beta x_i$$

where $x_i$ is the Glucose level of the person $i$

We assume proper but non informative priors for the parameters:
$$\alpha\sim \text{Normal}(0,1000)$$ $$\beta\sim \text{Normal}(0,1000)$$
We can make use of MCMC methods to obtain a sample from the posterior of
$(\alpha,\beta)$

We can achieve this using the MCMCpack that contains the MCMC algorithm
for obtaining a Bayesian Logistic Regression:

``` r
library(MCMCpack)
```

    ## Warning: package 'MCMCpack' was built under R version 4.0.5

    ## Loading required package: coda

    ## Warning: package 'coda' was built under R version 4.0.5

    ## Loading required package: MASS

    ## ##
    ## ## Markov Chain Monte Carlo Package (MCMCpack)

    ## ## Copyright (C) 2003-2022 Andrew D. Martin, Kevin M. Quinn, and Jong Hee Park

    ## ##
    ## ## Support provided by the U.S. National Science Foundation

    ## ## (Grants SES-0350646 and SES-0350613)
    ## ##

``` r
logit.mcmc <- MCMClogit(y.bin~Glucose,burnin=5000, mcmc=20000, thin = 10,data=data)
```

The logit.mcmc model implements the MCMC algorithm for 11000 iterations,
in which the first 1000 iteratiosn are used as burn-in iterations and
the other 10000 iterations are keep.

We can plot the trace of convergence and the posterior densities of the
intercept \(\alpha\) and the slope \(\beta\):

``` r
plot(logit.mcmc)
```

![](DiabetesCaseStudy_files/figure-gfm/unnamed-chunk-26-1.png)<!-- -->

There is no autocorrelation:

``` r
par(mfrow=c(1,2))
par(mar = rep(1, 4))
acf(logit.mcmc[,1], main='ACF of Intercept')
acf(logit.mcmc[,2], main='ACF of Glucose Coeff')
```

![](DiabetesCaseStudy_files/figure-gfm/unnamed-chunk-27-1.png)<!-- -->

``` r
summary(logit.mcmc)
```

    ## 
    ## Iterations = 5001:24991
    ## Thinning interval = 10 
    ## Number of chains = 1 
    ## Sample size per chain = 2000 
    ## 
    ## 1. Empirical mean and standard deviation for each variable,
    ##    plus standard error of the mean:
    ## 
    ##                 Mean       SD  Naive SE Time-series SE
    ## (Intercept) -5.38937 0.419673 9.384e-03      0.0103414
    ## Glucose      0.03819 0.003232 7.227e-05      0.0000802
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##                 2.5%      25%      50%      75%    97.5%
    ## (Intercept) -6.23301 -5.66876 -5.36851 -5.10551 -4.57348
    ## Glucose      0.03185  0.03605  0.03804  0.04034  0.04447

We can compare it with a classical Logistic Regression

``` r
fit = glm(y~Glucose,family = "binomial")
summary(fit)
```

    ## 
    ## Call:
    ## glm(formula = y ~ Glucose, family = "binomial")
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.1096  -0.7837  -0.5365   0.8566   3.2726  
    ## 
    ## Coefficients:
    ##              Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept) -5.350080   0.420827  -12.71   <2e-16 ***
    ## Glucose      0.037873   0.003252   11.65   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 993.48  on 767  degrees of freedom
    ## Residual deviance: 808.72  on 766  degrees of freedom
    ## AIC: 812.72
    ## 
    ## Number of Fisher Scoring iterations: 4

The results are very similar.

But the advantage of a Bayesian Logistic Regression is that we can
obtain a posterior distribution of the probability of a person having
diabetes. For example, we can obtain the posterior distribution of the
probability of Diabetes of someone with glucose level of 170 mg/dL

The trace and the density of probability looks good:

``` r
library(boot)
```

    ## 
    ## Attaching package: 'boot'

    ## The following object is masked _by_ '.GlobalEnv':
    ## 
    ##     tau

``` r
glucose_level = 170
diabetes.prob=inv.logit(logit.mcmc[,1]+glucose_level*logit.mcmc[,2])
plot(diabetes.prob)
```

![](DiabetesCaseStudy_files/figure-gfm/unnamed-chunk-30-1.png)<!-- -->

We can obtain a summary of the probability of a person having diabetes
with a glucose level of 170 mg/dL. With the Bayesian Logistic Regression
we can obtain the quantiles, that can be used for evaluate the veracity
or significance of the probability.

``` r
summary(diabetes.prob)
```

    ## 
    ## Iterations = 5001:24991
    ## Thinning interval = 10 
    ## Number of chains = 1 
    ## Sample size per chain = 2000 
    ## 
    ## 1. Empirical mean and standard deviation for each variable,
    ##    plus standard error of the mean:
    ## 
    ##           Mean             SD       Naive SE Time-series SE 
    ##      0.7494503      0.0307307      0.0006872      0.0007751 
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##   2.5%    25%    50%    75%  97.5% 
    ## 0.6886 0.7295 0.7505 0.7706 0.8057

And we can plot the probability distribution, for a better visualization
of the results:

``` r
hist(diabetes.prob,freq=F, xlim = c(0,1))
```

![](DiabetesCaseStudy_files/figure-gfm/unnamed-chunk-32-1.png)<!-- -->

And we can obtain the mean probability, the median probability and the
quantiles for the probability

``` r
mean(diabetes.prob)
```

    ## [1] 0.7494503

``` r
median(diabetes.prob)
```

    ## [1] 0.7504718

``` r
quantile(diabetes.prob,c(0.025,0.975))
```

    ##      2.5%     97.5% 
    ## 0.6885549 0.8057433

``` r
y.pred=rbinom(1000,size=1,prob=diabetes.prob)

plot(density(diabetes.prob), main = "Diabetes Probability of Glucose = 170")
```

![](DiabetesCaseStudy_files/figure-gfm/unnamed-chunk-34-1.png)<!-- -->

### Multi-variate Logistic Regression

We assume a logistic regression for this problem:

We set $y_i =1$ to a person \(i\) with Diabetes and $y_i =0$ if a
person $i$ doesn’t have Diabetes:

$$y_i\mid p_i\sim\text{Bernoulli}(p_i)$$ where $p_i$ denotes the
diabetes probability of the person \(i\), such that,
$$\log\frac{p_i}{1-p_i}= \alpha+X_i\beta$$ where $X_i$ is the
variables levels of a person (Pregnancies, Glucose, Blood Pressure, BMI,
Diabetes Pedigree Function, Skin Thickness, Insulin and Age)$i$

We assume proper but non informative priors for the parameters:
\[\alpha\sim \text{Normal}(0,1000)\]
\[\beta_i\sim \text{Normal}(0,1000)\]

``` r
logit.mcmc <- MCMClogit(y.bin~Pregnancies + Glucose + BloodPressure + BMI + DiabetesPedigreeFunction +SkinThickness + Insulin + Age, 
                        burnin=2000, mcmc=15000, thin =20, data=X)
```

We can evaluate the trace of the convergence:

``` r
par(mar = rep(2, 4))
par(mfrow = c(4,2))
plot(logit.mcmc)
```

![](DiabetesCaseStudy_files/figure-gfm/unnamed-chunk-36-1.png)<!-- -->![](DiabetesCaseStudy_files/figure-gfm/unnamed-chunk-36-2.png)<!-- -->![](DiabetesCaseStudy_files/figure-gfm/unnamed-chunk-36-3.png)<!-- -->
We can plot the autocorrelations, and observe that there is no
autocorrelation in any of the coefficients

``` r
par(mfrow=c(4,2))
par(mar = rep(2, 4))
for(i in 1:8){
  acf(logit.mcmc[,i])
}
```

![](DiabetesCaseStudy_files/figure-gfm/unnamed-chunk-37-1.png)<!-- -->

``` r
summary(logit.mcmc)
```

    ## 
    ## Iterations = 2001:16981
    ## Thinning interval = 20 
    ## Number of chains = 1 
    ## Sample size per chain = 750 
    ## 
    ## 1. Empirical mean and standard deviation for each variable,
    ##    plus standard error of the mean:
    ## 
    ##                                Mean        SD  Naive SE Time-series SE
    ## (Intercept)              -8.5747320 0.7677167 2.803e-02      0.0407173
    ## Pregnancies               0.1245477 0.0314623 1.149e-03      0.0015427
    ## Glucose                   0.0356545 0.0037729 1.378e-04      0.0001942
    ## BloodPressure            -0.0132368 0.0052418 1.914e-04      0.0002978
    ## BMI                       0.0928938 0.0157157 5.739e-04      0.0007336
    ## DiabetesPedigreeFunction  0.9664493 0.3092234 1.129e-02      0.0166744
    ## SkinThickness             0.0001905 0.0068280 2.493e-04      0.0003578
    ## Insulin                  -0.0011499 0.0009399 3.432e-05      0.0000549
    ## Age                       0.0143198 0.0098967 3.614e-04      0.0004912
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##                                2.5%       25%        50%       75%      97.5%
    ## (Intercept)              -10.048641 -9.107238 -8.5759392 -8.032756 -7.1444427
    ## Pregnancies                0.064340  0.103693  0.1243840  0.146503  0.1837644
    ## Glucose                    0.028841  0.032979  0.0356486  0.038236  0.0431591
    ## BloodPressure             -0.023403 -0.016992 -0.0130876 -0.009677 -0.0027044
    ## BMI                        0.063128  0.081782  0.0928675  0.103839  0.1244017
    ## DiabetesPedigreeFunction   0.383069  0.764159  0.9618602  1.168481  1.5461657
    ## SkinThickness             -0.013331 -0.004230  0.0003047  0.004929  0.0132726
    ## Insulin                   -0.002904 -0.001775 -0.0011600 -0.000555  0.0006867
    ## Age                       -0.005109  0.007546  0.0144816  0.020424  0.0337660

We can observe that the coefficients for the variables `SkinThickness`,
`Insulin` and `Age` are not significant, so we can delete them.

We create another model with the variables that are significant:

``` r
logit.mcmc <- MCMClogit(y.bin~Glucose + BloodPressure + 
                          BMI + DiabetesPedigreeFunction,
                        burnin=2000, mcmc=15000, thin =20, data=data)

summary(logit.mcmc)
```

    ## 
    ## Iterations = 2001:16981
    ## Thinning interval = 20 
    ## Number of chains = 1 
    ## Sample size per chain = 750 
    ## 
    ## 1. Empirical mean and standard deviation for each variable,
    ##    plus standard error of the mean:
    ## 
    ##                               Mean       SD  Naive SE Time-series SE
    ## (Intercept)              -7.508366 0.644855 0.0235468      0.0268107
    ## Glucose                   0.035687 0.003208 0.0001172      0.0001172
    ## BloodPressure            -0.007219 0.004974 0.0001816      0.0002045
    ## BMI                       0.077156 0.013561 0.0004952      0.0005348
    ## DiabetesPedigreeFunction  0.846151 0.277776 0.0101429      0.0112384
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##                              2.5%      25%       50%       75%     97.5%
    ## (Intercept)              -8.77844 -7.92815 -7.458505 -7.115980 -6.252596
    ## Glucose                   0.02923  0.03349  0.035630  0.037880  0.041972
    ## BloodPressure            -0.01644 -0.01037 -0.007341 -0.004078  0.003128
    ## BMI                       0.05121  0.06857  0.076041  0.085656  0.106714
    ## DiabetesPedigreeFunction  0.30570  0.65614  0.852480  1.023637  1.383308

We can compare it with a Classical Logistic Regression

``` r
fit = glm(y~Pregnancies + Glucose + BloodPressure + BMI + DiabetesPedigreeFunction,family = "binomial")
summary(fit)
```

    ## 
    ## Call:
    ## glm(formula = y ~ Pregnancies + Glucose + BloodPressure + BMI + 
    ##     DiabetesPedigreeFunction, family = "binomial")
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.7931  -0.7362  -0.4188   0.7251   2.9555  
    ## 
    ## Coefficients:
    ##                           Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)              -7.954952   0.675823 -11.771  < 2e-16 ***
    ## Pregnancies               0.153492   0.027835   5.514  3.5e-08 ***
    ## Glucose                   0.034658   0.003394  10.213  < 2e-16 ***
    ## BloodPressure            -0.012007   0.005031  -2.387  0.01700 *  
    ## BMI                       0.084832   0.014125   6.006  1.9e-09 ***
    ## DiabetesPedigreeFunction  0.910628   0.294027   3.097  0.00195 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 993.48  on 767  degrees of freedom
    ## Residual deviance: 728.56  on 762  degrees of freedom
    ## AIC: 740.56
    ## 
    ## Number of Fisher Scoring iterations: 5

We plot the trace of convergence and the autocorrelations, and observe
that there is no autocorrelation in the coefficients

``` r
par(mar = rep(2, 4))
par(mfrow = c(4,2))
plot(logit.mcmc)
```

![](DiabetesCaseStudy_files/figure-gfm/unnamed-chunk-41-1.png)<!-- -->![](DiabetesCaseStudy_files/figure-gfm/unnamed-chunk-41-2.png)<!-- -->

``` r
# par(mfrow=c(5,1))
# # par(mar = rep(1,5))
# par(mfrow=c(5,1))
par(mar = rep(2, 4))
par(mfrow = c(4,2))
for(i in 1:5){
  acf(logit.mcmc[,i])
}
```

![](DiabetesCaseStudy_files/figure-gfm/unnamed-chunk-42-1.png)<!-- -->

### Predictive Probability:

We can test the model, by obtaining the probability of diabetes of a
patient with the following values (which corresponds to a patient with
No Diabetes)

``` r
idx = 700
rw = data[idx,]
rw
```

    ##     Pregnancies Glucose BloodPressure SkinThickness Insulin  BMI
    ## 700           4     118            70             0       0 44.5
    ##     DiabetesPedigreeFunction Age Outcome
    ## 700                    0.904  26       0

``` r
# data[700, ]
idx = 700
rw = data[idx,]
new_data = c(rw$Glucose, rw$BloodPressure, 
             rw$BMI, rw$DiabetesPedigreeFunction)

pred_logit = logit.mcmc[,1] + new_data[1]*logit.mcmc[,2] + 
  new_data[2]*logit.mcmc[,3] + new_data[3]*logit.mcmc[,4] + 
  new_data[4]*logit.mcmc[,5]
diabetes.prob=inv.logit(pred_logit)
hist(diabetes.prob,freq=F, xlim = c(0,1))
```

![](DiabetesCaseStudy_files/figure-gfm/unnamed-chunk-44-1.png)<!-- -->

``` r
mean(diabetes.prob)
```

    ## [1] 0.5965871

``` r
median(diabetes.prob)
```

    ## [1] 0.5982196

``` r
quantile(diabetes.prob,c(0.025,0.975))
```

    ##      2.5%     97.5% 
    ## 0.4901855 0.6934838

A patient with the following values (which corresponds to a patient with
Diabetes)

``` r
idx = 133
rw = data[idx,]
rw
```

    ##     Pregnancies Glucose BloodPressure SkinThickness Insulin  BMI
    ## 133           3     170            64            37     225 34.5
    ##     DiabetesPedigreeFunction Age Outcome
    ## 133                    0.356  30       1

``` r
# data[133, ]
idx = 133
rw = data[idx,]
new_data = c(rw$Glucose, rw$BloodPressure, 
             rw$BMI, rw$DiabetesPedigreeFunction)
pred_logit = logit.mcmc[,1] + new_data[1]*logit.mcmc[,2] + 
  new_data[2]*logit.mcmc[,3] + new_data[3]*logit.mcmc[,4] + 
  new_data[4]*logit.mcmc[,5]
diabetes.prob=inv.logit(pred_logit)
hist(diabetes.prob,freq=F, xlim = c(0,1))
```

![](DiabetesCaseStudy_files/figure-gfm/unnamed-chunk-46-1.png)<!-- -->

``` r
mean(diabetes.prob)
```

    ## [1] 0.7411335

``` r
median(diabetes.prob)
```

    ## [1] 0.7411201

``` r
quantile(diabetes.prob,c(0.025,0.975))
```

    ##      2.5%     97.5% 
    ## 0.6761970 0.8049049

And a patient with the following values(Which corresponds to a patient
with no Diabetes)

``` r
idx = 11
rw = data[idx,]
rw
```

    ##    Pregnancies Glucose BloodPressure SkinThickness Insulin  BMI
    ## 11           4     110            92             0       0 37.6
    ##    DiabetesPedigreeFunction Age Outcome
    ## 11                    0.191  30       0

``` r
# data[133, ]
idx = 11
rw = data[idx,]
new_data = c(rw$Glucose, rw$BloodPressure, 
             rw$BMI, rw$DiabetesPedigreeFunction)
pred_logit = logit.mcmc[,1] + new_data[1]*logit.mcmc[,2] + 
  new_data[2]*logit.mcmc[,3] + new_data[3]*logit.mcmc[,4] + 
  new_data[4]*logit.mcmc[,5]
diabetes.prob=inv.logit(pred_logit)
hist(diabetes.prob,freq=F, xlim = c(0,1))
```

![](DiabetesCaseStudy_files/figure-gfm/unnamed-chunk-48-1.png)<!-- -->

``` r
mean(diabetes.prob)
```

    ## [1] 0.2357422

``` r
median(diabetes.prob)
```

    ## [1] 0.2329497

``` r
quantile(diabetes.prob,c(0.025,0.975))
```

    ##      2.5%     97.5% 
    ## 0.1778035 0.3040728

# Conclusion

  - We were able to obtain a Bayesian Distribution of the Glucose levels
    of non Diabetic people and a Bayesian Predictive Distribution of the
    Glucose levels of non diabetic people, that can be use for
    evaluating the extremes values of Glucose levels.
  - We were able to implement a univariate Bayesian Logistic Regression,
    using as feature the Glucose levels, in order to predict the
    probability of a person having Diabetes, which allowed us to obtain
    several statistics measures of the distribution probability of a
    single person having Diabetes.
  - With the Bayesian Logistic Regression we also implemented a
    multivariate Bayesian Logistic Regression with the significant
    measures, which allowed us to predict the probability of a person
    having Diabetes, and obtain several statistics measures of the
    distribution probability of a single person having Diabetes.
