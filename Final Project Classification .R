setwd("/Users/alisonzhang/Desktop/2018 Spring/STAT 686/Final Project")
dat = read.csv("Financial Distress.csv")


library(dplyr)
library(tidyr)
########## Data Exploration 1 ##########

## Financial Distress <= -0.5, then financially distressed 

## Outlier Detection & Solution 
plot(dat1$Financial.Distress, 
     xlab = 'ID',
     ylab = 'Financial Distress Score', 
     main = 'Financial Distress Plot')
# There is an obvious outlier whose value is over 120 
greatest_outlier = subset(dat, 
                          Company == subset(dat, Financial.Distress > 120)$Company)
# Other Financial Distress values for the company seem normal. 
# I decide to remove this only outlier first
set.seed(666)
dat1 = subset(dat, Financial.Distress < 120)
quantile(dat1$Financial.Distress, c(.025, .975))
plot(density(dat1$Financial.Distress), main = 'Density Plot with Median & 95% CI')
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

dim(distressed)
######################################



library(caret)
########## Data Exploration 2 ##########

## Processing Categorical Variables 
# Feature x80 is categorical
# Company identity is categorical 
#     Since conforming Company variable into dummy my create an extremely sparse
#     dataset, I decide to remove Company. It will be used if the prediction power 
#     is poor 

#dat1$x80 = as.factor(dat1$x80)
x80 = dat1$x80
#x80 = model.matrix( ~ x80 - 1)
#dat1$Company = as.factor(dat1$Company)
#Company = dat1$Company
#Company = model.matrix( ~ Company - 1)

dat1$x80 = NULL
dat2 = cbind(dat1, x80)

# What if we disregard the categorical variable? 
# To do this, comment out the code in the first block


##Centering & Scaling predictors
y = dat1$Financial.Distress

classy = c()
for(i in y){
  classy = append(classy, ifelse(i > -0.5, 1, 0))
}

dat2$Financial.Distress = NULL
dat2$Company = NULL

x = scale(dat2, center = TRUE, scale = TRUE)


## Data Splitting 
# 80% as training set 
# 20% as testing set
# from distressed companies and non-distressed companies 

set.seed(666)
train1 = sample(distressed_comp, floor(length(distressed_comp)*.8))
train2 = sample(non_distressed_comp, floor(length(non_distressed_comp)*.8))
train_comp = sort(c(train1, train2))
test_comp = unique(dat1$Company)[!unique(dat1$Company) 
                                 %in% train_comp]

train_x = x[which(dat1$Company %in% train_comp), ]
train_cy = as.factor(as.character(classy[which(dat1$Company %in% train_comp)]))

test_x = x[which(dat1$Company %in% test_comp), ]
test_cy = classy[which(dat1$Company %in% test_comp)]

train = cbind(as.data.frame(train_x), train_cy)


library(DMwR)
library(ROSE)
########## Resampling ##########
set.seed(666)
# smote_dat1 = SMOTE(train_cy ~., data = train, k = 10, perc.over = 500, perc.under= 700)
rose = ROSE(train_cy ~., data = train, p = 0.186)$data  
smote_dat1 = rose
table(smote_dat1$train_cy)

  # 700, 300: 864, 2268   38.10%
  # 800, 200: 972, 1728   56.25%
  # 700, 400: 864, 3024   28.57%
  # 700, 500: 864, 3780   22.86%
  # 500, 700: 648, 3780   17.14%

train_cy = smote_dat1$train_cy
smote_dat1$train_cy = NULL
train_x = smote_dat1








require(caret)
require(fastAdaboost)
##### Boosted classification trees
ada_train = train(train_x, train_cy, method = 'ada', 
                  metric = "Accuracy", 
                  trControl = trainControl(
                    method = "repeatedcv",
                    number = 5,
                    repeats = 5))
ada_cm_train = confusionMatrix(as.factor((as.character(train_cy))), 
                as.factor(as.character(predict(ada_train, newdata = train_x))))
ada_cm_test = confusionMatrix(as.factor((as.character(test_cy))), 
                as.factor(as.character(predict(ada_train, newdata = test_x))))
    # training sensitivity: 0.9342
    # testing sensitivity: 0.39286 



require(mboost)
require(plyr)
require(import)
##### Boosted Generalized Additive Model
gamboost_train = train(train_x, train_cy, method = 'gamboost', 
                       metric = "Accuracy", 
                       trControl = trainControl(
                         method = "repeatedcv",
                         number = 5,
                         repeats = 5))
gamboost_cm_train = confusionMatrix(as.factor((as.character(train_cy))), 
                                    as.factor(as.character(predict(gamboost_train, newdata = train_x))))
gamboost_cm_test = confusionMatrix(as.factor((as.character(test_cy))), 
                                   as.factor(as.character(predict(gamboost_train, newdata = test_x))))
  # training sensitivity: 82.95%
  # testing sensitivity:  39.68%


require(plyr)
require(mboost)
##### Boosted Generalized Linear Model 
glmboost_train = train(train_x, train_cy, method = 'glmboost', 
                       metric = "Accuracy", 
                       trControl = trainControl(
                         method = "repeatedcv",
                         number = 5,
                         repeats = 5))
glmboost_cm_train = confusionMatrix(as.factor((as.character(train_cy))), 
                                    as.factor(as.character(predict(glmboost_train, newdata = train_x))))
glmboost_cm_test = confusionMatrix(as.factor((as.character(test_cy))), 
                                   as.factor(as.character(predict(glmboost_train, newdata = test_x))))
# training sensitivity: 81.08%
# testing sensitivity:  39.06%


require(C50)
require(plyr)
##### Cost-Sensitive C5.0
c50_train = train(train_x, train_cy, method = 'C5.0Cost', 
                       metric = "Accuracy", 
                       trControl = trainControl(
                         method = "repeatedcv",
                         number = 5,
                         repeats = 5))
c50_cm_train = confusionMatrix(as.factor((as.character(train_cy))), 
                                    as.factor(as.character(predict(c50_train, newdata = train_x))))
c50_cm_test = confusionMatrix(as.factor((as.character(test_cy))), 
                                   as.factor(as.character(predict(c50_train, newdata = test_x))))
  # training sensitivity: 1
  # testing sensitivity:  32.69%

c50_cm_train


require(rpart)
require(plyr)
##### Cost-Sensitive CART
rpartcost_train = train(train_x, train_cy, method = 'rpartCost', 
                  metric = "Accuracy", 
                  trControl = trainControl(
                    method = "repeatedcv",
                    number = 5,
                    repeats = 5))
rpartcost_cm_train = confusionMatrix(as.factor((as.character(train_cy))), 
                               as.factor(as.character(predict(rpartcost_train, newdata = train_x))))
rpartcost_cm_test = confusionMatrix(as.factor((as.character(test_cy))), 
                              as.factor(as.character(predict(rpartcost_train, newdata = test_x))))
# training sensitivity: 73.27%
# testing sensitivity:  26.53%


require(rpart)
require(plyr)
##### Cost-Sensitive CART
rpartcost_train = train(train_x, train_cy, method = 'rpartCost', 
                        metric = "Accuracy", 
                        trControl = trainControl(
                          method = "repeatedcv",
                          number = 5,
                          repeats = 5))
rpartcost_cm_train = confusionMatrix(as.factor((as.character(train_cy))), 
                                     as.factor(as.character(predict(rpartcost_train, newdata = train_x))))
rpartcost_cm_test = confusionMatrix(as.factor((as.character(test_cy))), 
                                    as.factor(as.character(predict(rpartcost_train, newdata = test_x))))
  # training sensitivity: 73.27%
  # testing sensitivity:  26.53%



require(xgboost)
require(plyr)
require(caret)
##### eXtreme Gradient Boosting 
xg_train = train(train_x, train_cy, method = 'xgbTree', 
                        metric = "Accuracy", 
                        trControl = trainControl(
                          method = "repeatedcv",
                          number = 5,
                          repeats = 5))
xg_cm_train = confusionMatrix(as.factor((as.character(train_cy))), 
                                     as.factor(as.character(predict(xg_train, newdata = train_x))))
xg_cm_test = confusionMatrix(as.factor((as.character(test_cy))), 
                                    as.factor(as.character(predict(xg_train, newdata = test_x))))
# training sensitivity: 1
# testing sensitivity:  46.15%

## 1, 0.35088
## 1, 0.48571
## 1, 0.52941
## 1, 0.43750

## ROSE: 1, 0.5 

## ROSE: 



require(mda)
require(earth)
##### Flexible Discriminant Analysis 
fda_train = train(train_x, train_cy, method = 'fda', 
                 metric = "Accuracy", 
                 trControl = trainControl(
                   method = "repeatedcv",
                   number = 5,
                   repeats = 5))
fda_cm_train = confusionMatrix(as.factor((as.character(train_cy))), 
                              as.factor(as.character(predict(fda_train, newdata = train_x))))
fda_cm_test = confusionMatrix(as.factor((as.character(test_cy))), 
                             as.factor(as.character(predict(fda_train, newdata = test_x))))
# training sensitivity: 81.04%
# testing sensitivity:  33.78%



require(nnet)
##### Model Averaged Neural Network 
avnnet_train = train(train_x, train_cy, method = 'avNNet', 
                  metric = "Accuracy", 
                  trControl = trainControl(
                    method = "repeatedcv",
                    number = 5,
                    repeats = 5))
avnnet_cm_train = confusionMatrix(as.factor((as.character(train_cy))), 
                               as.factor(as.character(predict(avnnet_train, newdata = train_x))))
avnnet_cm_test = confusionMatrix(as.factor((as.character(test_cy))), 
                              as.factor(as.character(predict(avnnet_train, newdata = test_x))))
# training sensitivity: 95.86%
# testing sensitivity:  38.30%


require(nnet)
##### Multivariate Adaptive Regression Spline 
earth_train = train(train_x, train_cy, method = 'earth', 
                     metric = "Accuracy", 
                     trControl = trainControl(
                       method = "repeatedcv",
                       number = 5,
                       repeats = 5))
earth_cm_train = confusionMatrix(as.factor((as.character(train_cy))), 
                                  as.factor(as.character(predict(earth_train, newdata = train_x))))
earth_cm_test = confusionMatrix(as.factor((as.character(test_cy))), 
                                 as.factor(as.character(predict(earth_train, newdata = test_x))))
# training sensitivity: 82.31%
# testing sensitivity:  37.31%



require(nnet)
##### Neural Network 
nnet_train = train(train_x, train_cy, method = 'nnet', 
                    metric = "Accuracy", 
                    trControl = trainControl(
                      method = "repeatedcv",
                      number = 5,
                      repeats = 5))
nnet_cm_train = confusionMatrix(as.factor((as.character(train_cy))), 
                                 as.factor(as.character(predict(nnet_train, newdata = train_x))))
nnet_cm_test = confusionMatrix(as.factor((as.character(test_cy))), 
                                as.factor(as.character(predict(nnet_train, newdata = test_x))))
# training sensitivity: 95.54%
# testing sensitivity:  33.93%


require(plsRglm)
##### Multilayer Perceptron Network with Weight Decay 
plsRglm_train = train(train_x, train_cy, method = 'plsRglm', 
                   metric = "Accuracy", 
                   trControl = trainControl(
                     method = "repeatedcv",
                     number = 5,
                     repeats = 5))
plsRglm_cm_train = confusionMatrix(as.factor((as.character(train_cy))), 
                                as.factor(as.character(predict(plsRglm_train, newdata = train_x))))
plsRglm_cm_test = confusionMatrix(as.factor((as.character(test_cy))), 
                               as.factor(as.character(predict(plsRglm_train, newdata = test_x))))
  # training sensitivity: 80.10%
  # testing sensitivity:  31.82%



require(rotationForest)
##### Rotation Forest 
rof_train = train(train_x, train_cy, method = 'rotationForest', 
                      metric = "Accuracy", 
                      trControl = trainControl(
                        method = "repeatedcv",
                        number = 5,
                        repeats = 5))
rof_cm_train = confusionMatrix(as.factor((as.character(train_cy))), 
                                   as.factor(as.character(predict(rof_train, newdata = train_x))))
rof_cm_test = confusionMatrix(as.factor((as.character(test_cy))), 
                                  as.factor(as.character(predict(rof_train, newdata = test_x))))
# training sensitivity: 89.98%
# testing sensitivity:  32.39%














