setwd("/Users/alisonzhang/Desktop/2018 Spring/STAT 686/Final Project")
dat = read.csv("Financial Distress.csv")


library(dplyr)
library(tidyr)
########## Data Exploration ##########

## Financial Distress <= -0.5, then financially distressed 

## Outlier Detection & Solution 
plot(dat$Financial.Distress)
  # There is an obvious outlier whose value is over 120 
greatest_outlier = subset(dat, 
                          Company == subset(dat, Financial.Distress > 120)$Company)
  # Other Financial Distress values for the company seem normal. 
  # I decide to remove this only outlier first
set.seed(666)
dat1 = subset(dat, Financial.Distress < 120)
quantile(dat1$Financial.Distress, c(.025, .975))
plot(density(dat1$Financial.Distress))
abline(v = median(dat1$Financial.Distress), col = 'red')
abline(v = -0.768085, col = 'blue')
abline(v = 5.248100, col = 'blue')
  # Only one outlier is removed from the dataset. 
  # More outliers will be removed if the prediction power is poor


## Identify bankruptcy 
distressed = subset(dat1, Financial.Distress <= -0.5)
distressed_comp = distressed$Company
non_distressed_comp = unique(dat1$Company)[!unique(dat1$Company) 
                                           %in% distressed_comp]
######################################



library(caret)
########## Data Exploration ##########

## Processing Categorical Variables 
# Feature x80 is categorical
# Company identity is categorical 
#     Since conforming Company variable into dummy my create an extremely sparse
#     dataset, I decide to remove Company. It will be used if the prediction power 
#     is poor 
dat1$x80 = as.factor(dat1$x80)
x80 = dat1$x80
x80 = model.matrix( ~ x80 - 1)
dat1$Company = as.factor(dat1$Company)
Company = dat1$Company
Company = model.matrix( ~ Company - 1)

dat1$x80 = NULL
dat2 = cbind(dat1, x80)


##Centering & Scaling predictors
y = dat1$Financial.Distress

dat2$Financial.Distress = NULL
dat2$Company = NULL

x = scale(dat2, center = TRUE, scale = TRUE)


##Data Splitting 
# 80% as training set 
# 20% as testing set
# from distressed companies and non-distressed companies 

# How to work with sampling and centering & scaling?!?!?!?!?

set.seed(666)
train1 = sample(distressed_comp, floor(length(distressed_comp)*.8))
train2 = sample(non_distressed_comp, floor(length(non_distressed_comp)*.8))
train_comp = sort(c(train1, train2))
test_comp = unique(dat1$Company)[!unique(dat1$Company) 
                            %in% train_comp]

train_x = x[which(dat1$Company %in% train_comp), ]
train_y = y[which(dat1$Company %in% train_comp)]
test_x = x[which(dat1$Company %in% test_comp), ]
test_y = y[which(dat1$Company %in% test_comp)]
######################################

