setwd("/Users/alisonzhang/Desktop/2018 Spring/STAT 686/Final Project")
dat = read.csv("Financial Distress.csv")


library(dplyr)
library(tidyr)
########## Data Exploration 1 ##########

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

dim(distressed)
dim()
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
train_y = y[which(dat1$Company %in% train_comp)]
train_cy = classy[which(dat1$Company %in% train_comp)]

test_x = x[which(dat1$Company %in% test_comp), ]
test_y = y[which(dat1$Company %in% test_comp)]
test_cy = classy[which(dat1$Company %in% test_comp)]
######################################



############# PCA ###############
pca_check = prcomp(train_x)
summary(pca_check)



############# Resampling ###############
# If regression does not work and we need to use classification directly
# Resample the data to create more "bankrupt" data 




library(caret)
library(glmnet)
library(plus)
########## Linear Regression Method doesn't seem to work in this setting 

# Ridge Regression 
ridgecv = cv.glmnet(train_x, train_y, 
                 nfold = 10, alpha = 0)
ridge = glmnet(train_x, train_y, alpha = 0, lambda = ridgecv$lambda.1se)
ridgetest = glmnet(test_x, test_y, alpha = 0, lambda = ridgecv$lambda.1se)
predict(ridge, type = 'coef')
ridgetrainresult = cbind(train_y, predict(ridge, train_x))
ridgetestresult = cbind(test_y, predict(ridge, test_x))
ridgetrain_mse = sum((ridgetrainresult[, 1] - 
                        ridgetrainresult[, 2])^2)/length(ridgetrainresult[, 1]) 
ridgetest_mse = sum((ridgetestresult[, 1] - 
                       ridgetestresult[, 2])^2)/length(ridgetestresult[, 1]) 
    # 2.178635
    # 2.15488 

# Elastic Net 
eecv = cv.glmnet(train_x, train_y, 
                    nfold = 10, alpha = 0.5)
ee = glmnet(train_x, train_y, alpha = 0.5, lambda = eecv$lambda.1se)
eetest = glmnet(test_x, test_y, alpha = 0.5, lambda = eecv$lambda.1se)
predict(ee, type = 'coef')
eetrainresult = cbind(train_y, predict(ee, train_x))
eetestresult = cbind(test_y, predict(eetest, test_x))
eetrain_mse = sum((eetrainresult[, 1] - 
                     eetrainresult[, 2])^2)/length(eetrainresult[, 1]) 
eetest_mse =  sum((eetestresult[, 1] - 
                     eetestresult[, 2])^2)/length(eetestresult[, 1]) 
  # training error: 2.61947
  # testing error:  2.325715

# L1 regularization seems to work better 




################ L1 Regularization ################

### Top1 (testing MSE):  Boosted Generalized Linear Model 1.186139

require(lars)
######### Least Angle Regression Faster and more accurate 
la_train = train(train_x, train_y, method = 'lars', 
                 metric = "RMSE", 
                 trControl = trainControl(
                   method = "repeatedcv",
                   number = 5,
                   repeats = 5))
la_test = predict(la_train, newdata = test_x)
la_train_mse = sum((train_y - 
                      predict(la_train, newdata = train_x))^2)/length(train_y) 
la_test_mse = sum((test_y-la_test)^2)/length(la_test) 

  # training mse: 1.309547
  # testing mse: 1.486019


require(rqPen)
########## Non-convex penalized quantile regression
rqnc_train = train(train_x, train_y, method = 'rqnc', 
                  metric = "RMSE", 
                  trControl = trainControl(
                    method = "repeatedcv",
                    number = 5,
                    repeats = 5))
rqnc_train_mse = sum((train_y - 
                       predict(rqnc_train, newdata = train_x))^2)/length(train_y) 
rqnc_test_mse = sum((test_y - 
                      predict(rqnc_train, newdata = test_x))^2)/length(test_y) 

  # training: 1.834793
  # testing: 1.673703


require(rqPen)
########## Quantile Regression with Lasso Penalty 
rqlasso_train = train(train_x, train_y, method = 'rqlasso', 
                  metric = "RMSE", 
                  trControl = trainControl(
                    method = "repeatedcv",
                    number = 5,
                    repeats = 5))
rqlasso_train_mse = sum((train_y - 
                       predict(rqlasso_train, newdata = train_x))^2)/length(train_y) 
rqlasso_test_mse = sum((test_y - 
                      predict(rqlasso_train, newdata = test_x))^2)/length(test_y) 
  # training: 2.040764
  # testing:  1.833237


require(relaxo)
require(plyr)
########## Relaxed Lasso
rela_train = train(train_x, train_y, method = 'relaxo', 
                      metric = "RMSE", 
                      trControl = trainControl(
                        method = "repeatedcv",
                        number = 5,
                        repeats = 5))
rela_train_mse = sum((train_y - 
                           predict(rela_train, newdata = train_x))^2)/length(train_y) 
rela_test_mse = sum((test_y - 
                          predict(rela_train, newdata = test_x))^2)/length(test_y) 
  # training: 2.423962
  # testing:  2.414913

require(spikeslab)
########## Spike and Slab Regression 
spikes_train = train(train_x, train_y, method = 'spikeslab', 
                     metric = "RMSE", 
                     trControl = trainControl(
                       method = "repeatedcv",
                       number = 5,
                       repeats = 5))
spikes_train_mse = sum((train_y - 
                          predict(spikes_train, newdata = train_x))^2)/length(train_y) 
spikes_test_mse = sum((test_y - 
                         predict(spikes_train, newdata = test_x))^2)/length(test_y) 

  # training: 1.32556
  # testing:  1.456363

require(nnet)
## Accespts Case Weights 
########## Projection Pursuit Regression  Overfitting involved here 
ppr_train = train(train_x, train_y, method = 'ppr',
                     metric = "RMSE", 
                     trControl = trainControl(
                       method = "repeatedcv",
                       number = 5,
                       repeats = 5))
ppr_train_mse = sum((train_y - 
                          predict(ppr_train, newdata = train_x))^2)/length(train_y) 
ppr_test_mse = sum((test_y - 
                         predict(ppr_train, newdata = test_x))^2)/length(test_y) 

  # training: 1.092568
  # testing:  1.343419


require(MASS)


require(mboost)
require(party)
require(plyr)
########## Boosted tree
bt_train = train(train_x, train_y, method = 'blackboost',
                  metric = "RMSE", 
                  trControl = trainControl(
                    method = "repeatedcv",
                    number = 5,
                    repeats = 5))
bt_train_mse = sum((train_y - 
                       predict(bt_train, newdata = train_x))^2)/length(train_y) 
bt_test_mse = sum((test_y - 
                      predict(bt_train, newdata = test_x))^2)/length(test_y)

  # training: 0.9887555
  # testing:  1.292693

require(plyr)
########## Boosted Generalized Additive Model


require(rpart)
######### CART
cart_train = train(train_x, train_y, method = 'rpart',
                 metric = "RMSE", 
                 trControl = trainControl(
                   method = "repeatedcv",
                   number = 5,
                   repeats = 5))
cart_train_mse = sum((train_y - 
                      predict(cart_train, newdata = train_x))^2)/length(train_y) 
cart_test_mse = sum((test_y - 
                     predict(cart_train, newdata = test_x))^2)/length(test_y)

   # training: 1.503354
   # testing:  1.583972



require(caret)
require(plyr)
require(e1071)
########## Bagged CART
bcart_train = train(train_x, train_y, method = 'treebag',
                   metric = "RMSE", 
                   trControl = trainControl(
                     method = "repeatedcv",
                     number = 5,
                     repeats = 5))
bcart_train_mse = sum((train_y - 
                        predict(bcart_train, newdata = train_x))^2)/length(train_y) 
bcart_test_mse = sum((test_y - 
                       predict(bcart_train, newdata = test_x))^2)/length(test_y)

check = cbind(test_y, predict(bcart_train, newdata = test_x))

  # training: 0.989467
  # testing:  1.231283

check = cbind(test_y, predict(bcart_train, newdata = test_x))


require(mboost)
require(plyr)
require(import)
######### Boosted Generalized Additive Model  
gamboost_train = train(train_x, train_y, method = 'gamboost',
                    metric = "RMSE", 
                    trControl = trainControl(
                      method = "repeatedcv",
                      number = 5,
                      repeats = 5))
gamboost_train_mse = sum((train_y - 
                         predict(gamboost_train, newdata = train_x))^2)/length(train_y) 
gamboost_test_mse = sum((test_y - 
                        predict(gamboost_train, newdata = test_x))^2)/length(test_y)

  # training: 1.186139
  # testing: 1.186139

require(mboost)
require(plyr)
########## Boosted Generalized Linear Model
glmboost_train = train(train_x, train_y, method = 'glmboost',
                       metric = "RMSE", 
                       trControl = trainControl(
                         method = "repeatedcv",
                         number = 5,
                         repeats = 5))
glmboost_train_mse = sum((train_y - 
                            predict(glmboost_train, newdata = train_x))^2)/length(train_y) 
glmboost_test_mse = sum((test_y - 
                           predict(glmboost_train, newdata = test_x))^2)/length(test_y)

  # training: 1.427396
  # testing: 1.398082


require(gbm)
require(plyr)
########## Stochastic Gradient Boosting 
gbm_train = train(train_x, train_y, method = 'gbm',
                  metric = "RMSE", 
                  trControl = trainControl(
                    method = "repeatedcv",
                    number = 5,
                    repeats = 5))
gbm_train_mse = sum((train_y - 
                        predict(gbm_train, newdata = train_x))^2)/length(train_y) 
gbm_test_mse = sum((test_y - 
                       predict(gbm_train, newdata = test_x))^2)/length(test_y)
  # training: 0.8036005
  # testing: 1.28199


require(brnn)
######### Bayesian Regularized Neural Netowkrs
brnn_train = train(train_x, train_y, method = 'brnn',
                  metric = "RMSE", 
                  trControl = trainControl(
                    method = "repeatedcv",
                    number = 5,
                    repeats = 5))
brnn_train_mse = sum((train_y - 
                       predict(brnn_train, newdata = train_x))^2)/length(train_y) 
brnn_test_mse = sum((test_y - 
                      predict(brnn_train, newdata = test_x))^2)/length(test_y)
  # training: 1.128469
  # testing: 1.207897


require(LiblineaR)
########## L2 Regularized SVM with Linear Kernel
svmlinear3_train = train(train_x, train_y, method = 'svmLinear3',
                   metric = "RMSE", 
                   trControl = trainControl(
                     method = "repeatedcv",
                     number = 5,
                     repeats = 5))
svmlinear3_train_mse = sum((train_y - 
                        predict(svmlinear3_train, newdata = train_x))^2)/length(train_y) 
svmlinear3_test_mse = sum((test_y - 
                       predict(svmlinear3_train, newdata = test_x))^2)/length(test_y)

  # training: 1.314903
  # testing: 1.470469

require(kernlab)
########## SVM with Polynomial Kernel
svmpoly_train = train(train_x, train_y, method = 'svmPoly',
                         metric = "RMSE", 
                         trControl = trainControl(
                           method = "repeatedcv",
                           number = 5,
                           repeats = 5))
svmpoly_train_mse = sum((train_y - 
                              predict(svmpoly_train, newdata = train_x))^2)/length(train_y) 
svmpoly_test_mse = sum((test_y - 
                             predict(svmpoly_train, newdata = test_x))^2)/length(test_y)

  # training: 1.702041
  # testing:  1.530558

require(kernlab)
########## SVM with RBF
svmrbf_train = train(train_x, train_y, method = 'svmRadial',
                      metric = "RMSE", 
                      trControl = trainControl(
                        method = "repeatedcv",
                        number = 5,
                        repeats = 5))
svmrbf_train_mse = sum((train_y - 
                           predict(svmrbf_train, newdata = train_x))^2)/length(train_y) 
svmrbf_test_mse = sum((test_y - 
                          predict(svmrbf_train, newdata = test_x))^2)/length(test_y)

  # training: 1.030741
  # testing: 2.557623








