library(ranger)
library(xgboost)
library(torch)
library(iml)
library(cito)
library(glmnet)
library(glmnetUtils)
set.seed(42)
Sys.setenv(OMP_NUM_THREADS=3)

source("code/AME.R")
source("code/Scenarios.R")

torch::torch_set_num_threads(3L)


get_R = function(pred, test) {
  return(cor(test[,1], pred)**2)
}

train_test_brt = function(train, test,test2, MCE = FALSE, nrounds = 100) {
  brt = xgboost::xgboost(data=xgboost::xgb.DMatrix(as.matrix(train)[,-1,drop=FALSE],
                                                    label = (as.matrix(train)[,1, drop=FALSE])),
                          nrounds = nrounds,
                          objective="reg:squarederror", nthread = 1, verbose = 0)
  if(MCE) {
    ME = diag(marginalEffects(brt, as.matrix(train[,-1, drop = FALSE]), interactions = FALSE)$mean)
    return(list(MCE = ME, R2 = get_R(predict(brt, newdata = as.matrix(test[,-1,drop=FALSE])), test)))
  } else {
    return(list(R2out=get_R(predict(brt, newdata = as.matrix(test[,-1,drop=FALSE])), test), R2in=get_R(predict(brt, newdata = as.matrix(test2[,-1,drop=FALSE])), test2)))
  }
}


train_test_rf = function(train, test,test2, MCE = FALSE) {
  rf = ranger(LungC ~., data = data.frame(train), num.trees = 100L, num.threads = 3L)
  if(MCE) {
    ME = diag(marginalEffects(rf, as.matrix(train), interactions = FALSE)$mean)
    return(list(MCE = ME, R2 = get_R(predict(rf, data = test)$predictions, test)))
  } else { return(list(R2out=get_R(predict(rf, data = test)$predictions, test), R2in=get_R(predict(rf, data = test2)$predictions, test2))) }
}

train_test_cito = function(train, test, test2, MCE = FALSE) {
  dnn = cito::dnn(LungC~., data = as.data.frame(train), 
                  hidden = rep(50L, 3),
                  verbose = FALSE, 
                  plot=FALSE, lambda = 0.00, alpha = 1., 
                  epochs = 500,
                  lr = 0.01,
                  device = "cuda",
                  lr_scheduler = config_lr_scheduler("reduce_on_plateau", factor = 0.80, patience = 7), early_stopping = 20)
  dnn$use_model_epoch = length(dnn$weights)
  
  if(MCE) {
    ME = diag(marginalEffects(dnn, interactions = FALSE)$mean)
    return(list(MCE = ME, R2 = get_R(predict(dnn, newdata = test)[,1], test)))
  } else {return(list(R2out=get_R(predict(dnn, newdata = test)[,1], test), R2in=get_R(predict(dnn, newdata = test2)[,1], test2))) }
}


train_test_lm = function(train, test, test2, MCE = FALSE) {
  lm = lm(LungC~., data = as.data.frame(train))
  if(MCE) {
    ME = diag(marginalEffects(lm, interactions = FALSE)$mean)
    return(list(MCE = ME, R2 = get_R(predict(lm, newdata = test), test)))
  } else {return(list(R2out=get_R(predict(lm, newdata = test), test), R2in=get_R(predict(lm, newdata = test2), test2))) }
}


train_test_glmnet= function(train, test,test2, MCE = FALSE) {
  glmnet = cva.glmnet(as.matrix(train[,-1,drop=FALSE]), train[,1], alpha = 0.2)
  if(MCE) {
  ME = diag(marginalEffects(glmnet, data = as.matrix(train[,-1,drop=FALSE], interactions = FALSE, max_indices = 5), alpha = 0.2)$mean)
  return(list(MCE = ME, R2 = get_R(predict(glmnet, newx = as.matrix(test[,-1,drop=FALSE]), alpha = 0.2)[,1], test)))
  } else {return(list(R2out=get_R(predict(glmnet, newx = as.matrix(test[,-1,drop=FALSE]), alpha = 0.2)[,1], test), R2in=get_R(predict(glmnet, newx = as.matrix(test2[,-1,drop=FALSE]), alpha = 0.2)[,1], test2) ))}
}



simulate_SDM = function(effs = c(1.0, 1.0, -1.5, 1.0, 1.0, 1.0, 1.0), n = 5000) {
  Stress = rnorm(n, sd = 0.3)
  SleepD = effs[2]*Stress + rnorm(n, sd = 0.3)
  Smoking = effs[1]*Stress + effs[5]*SleepD + rnorm(n, sd = 0.3)
  LungC = effs[3]*Smoking + effs[4]*SleepD + rnorm(n, sd = 0.3)
  LungV = effs[6]*LungC + effs[7]*Smoking + rnorm(n, sd = 0.3)
  return(data.frame(LungC = LungC, Smoking = Smoking, SleepD = SleepD, LungV = LungV, mvtnorm::rmvnorm(n, sigma = diag(0.3, 10))))
}


samples = 200L
result_list = vector("list", samples)
cl = parallel::makeCluster(50L)
nodes = unlist(parallel::clusterEvalQ(cl, paste(Sys.info()[['nodename']], Sys.getpid(), sep='-')))
parallel::clusterExport(cl, varlist = ls(envir = .GlobalEnv))
parallel::clusterEvalQ(cl, {library(ranger);library(xgboost);library(cito);library(glmnet);library(glmnetUtils);Sys.setenv(OMP_NUM_THREADS=3);library(tidyverse)})
result_list = 
  parallel::parLapply(cl, 1:samples, function(i) {
    
    # who am I
    myself = paste(Sys.info()[['nodename']], Sys.getpid(), sep='-')
    dist = cbind(nodes,0:3)
    dev = as.integer(as.numeric(dist[which(dist[,1] %in% myself, arr.ind = TRUE), 2]))
    
    
    Sys.setenv(CUDA_VISIBLE_DEVICES=dev)

    # data = simulate_SDM(   c(3.0,  3.0, 2.5, -1.2, 0.0, 1.0, 1.0), n = 1000)
    # train = data[1:500, ]
    # test1 = simulate_SDM(   c(0.0,  0.0, 2.5, -1.2, 0.0, 0.0, 0.0), n = 500)
    # test2 = data[-(1:500), ]
    data = simulate_SDM(   c(2.0,  -2.0, 2, -1.5, 0.0, -1, -1.5), n = 1000)
    train = data[1:500, ]
    test1 = simulate_SDM(   c(0.0,  0.0, 2, -1.5, 0.0, 0.0, 0.0), n = 500)
    test2 = data[-(1:500), ]
    
    data = simulate_SDM(   c(-3.0,  -2.0, 2, -1, 0.0, -1.5, -.5), n = 1000)
    train = data[1:500, ]
    test1 = simulate_SDM(   c(0.0,  0.0, 2, -1, 0.0, 0.0, 0.0), n = 500)
    test2 = data[-(1:500), ]
    
    
    
    result = matrix(NA, 3, 4)
    result[1, ]= c(train_test_brt(train[,-4], test1[,-4], test2[,-4]) %>% unlist, 
                   train_test_brt(train[,], test1[,], test2[,]) %>% unlist) 
    result[2, ]=  c(train_test_rf(train[,-4], test1[,-4], test2[,-4]) %>% unlist, 
                    train_test_rf(train[,], test1[,], test2[,]) %>% unlist) 
    result
    result[3, ]=  c(train_test_cito(train[,-4], test1[,-4], test2[,-4]) %>% unlist,
                    train_test_cito(train[,], test1[,], test2[,]) %>% unlist)
    colnames(result) = c("R2out","R2in", "R2out", "R2in" )
    return(result)
    
  })
abind::abind(result_list, along = 0L)
apply(abind::abind(result_list, along = 0L), 2:3, mean)

saveRDS(abind::abind(result_list, along = 0L), "results/results_case_study.RDS")



sapply(1:2000, function(N) {

data = simulate_SDM(   c(-3.0,  -2.0, 2, -1, 0.0, -1.5, -.5), n = 1000)
train = data[1:500, ]
test1 = simulate_SDM(   c(0.0,  0.0, 2, -1, 0.0, 0.0, 0.0), n = 500)
test2 = data[-(1:500), ]



result = matrix(NA, 1, 4)
result[1, ]= c(train_test_brt(train[,-4], test1[,-4], test2[,-4], nrounds = N) %>% unlist, 
               train_test_brt(train[,], test1[,], test2[,], nrounds = N) %>% unlist) 

})
