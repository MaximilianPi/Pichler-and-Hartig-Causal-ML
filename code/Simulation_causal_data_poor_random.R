library(ranger)
library(xgboost)
library(torch)
library(iml)
library(cito)
library(glmnet)
library(glmnetUtils)
library(MASS)
set.seed(42)

Sys.setenv(OMP_NUM_THREADS=5)

source("code/AME.R")
source("code/Scenarios.R")

get_result = function(sim ) {
  samples = 100L
  result_list = vector("list", samples)
  
  cl = parallel::makeCluster(50L)
  parallel::clusterExport(cl, varlist = ls(envir = .GlobalEnv))
  parallel::clusterEvalQ(cl, {library(ranger);library(xgboost);library(cito);library(glmnet);library(glmnetUtils);Sys.setenv(OMP_NUM_THREADS=3);torch::torch_set_num_threads(1L);torch::torch_set_num_interop_threads(1L)
})
  result_list = 
    parallel::parLapply(cl, 1:samples, function(i) {
      #library(torch)
      print(i)
      Sigma = cov2cor(rWishart(1, 100, diag(1.0, 100))[,,1])
      
      simulation = sim(Sigma)
      data = simulation[[1]]
      true_effs = simulation[[2]]
      
      ind = nrow(data)/2
      
      train = data[1:ind,]
      test = data[(ind+1):nrow(data),]
      
      result = vector("list", 8L)
      
      rmse = function(y, y_hat) sqrt(mean((y-y_hat)**2))
      
      ## LM
      m = lm(Y~., data = data.frame(train))
      eff = diag(marginalEffects(m, interactions=FALSE)$mean)
      pred = predict(m, newdata = data.frame(test))
      result[[1]] = list(eff, rmse(test[,1], pred), true_effs)
      
      
      ## RF
      m = ranger(Y ~., data = data.frame(train), num.trees = 100L,num.threads = 3L)
      eff = diag(marginalEffects(m, data = data.frame(train), interactions=FALSE)$mean)
      pred = predict(m, data = data.frame(test))$predictions
      result[[2]] = list(eff, rmse(test[,1], pred), true_effs)
      
      ## BRT
      m = xgboost::xgboost(data=xgboost::xgb.DMatrix(as.matrix(train)[,-1],
                                                     label = (as.matrix(train)[,1, drop=FALSE])),
                           nrounds = 140L,
                           objective="reg:squarederror", nthread = 1, verbose = 0)
      
      eff = diag(marginalEffects(m, data = data.frame(train)[,-1], interactions=FALSE)$mean)
      pred = predict(m, newdata = xgboost::xgb.DMatrix(test[,-1]))
      result[[3]] = list(eff, rmse(test[,1], pred), true_effs)
      
      
      ## DNN
      m = cito::dnn(Y~., data = as.data.frame(train), 
                    activation = rep("selu", 2),
                    hidden = rep(20, 2),
                    batchsize = 50, 
                    verbose = FALSE, 
                    plot=FALSE, lambda = 0.00, alpha = 1., 
                    epochs = 300,
                    lr = 0.05,
                    lr_scheduler = config_lr_scheduler("reduce_on_plateau", factor = 0.90, patience = 4), early_stopping = 6)
      eff = diag(marginalEffects(m, interactions = FALSE)$mean)
      pred = predict(m, newdata = data.frame(test))
      result[[4]] = list(eff, rmse(test[,1], pred), true_effs)
      
      
      ## DNN with dropout
      
      m = cito::dnn(Y~., data = as.data.frame(train), 
                    activation = rep("selu", 2),
                    hidden = rep(20, 2),
                    verbose = FALSE, 
                    batchsize = 50, 
                    dropout = 0.15,
                    lr = 0.05,
                    plot=FALSE, lambda = 0.00, alpha = 1., 
                    epochs = 300,
                    lr_scheduler = config_lr_scheduler("reduce_on_plateau", factor = 0.90, patience = 4), early_stopping = 6)
      eff = diag(marginalEffects(m, interactions = FALSE)$mean)
      pred = predict(m, newdata = data.frame(test))
      result[[5]] = list(eff, rmse(test[,1], pred), true_effs)
      
      
      
      ## L1
      m = cva.glmnet(train[,-1], train[,1], alpha = 1.0)
      eff = diag(marginalEffects(m, data = train[,-1],alpha = 1.0, interactions = FALSE)$mean)
      pred = predict(m, newx = test[,-1], alpha = 1.0)
      result[[6]] = list(eff, rmse(test[,1], pred), true_effs)
      
      
      ## L2 
      m = cva.glmnet(train[,-1], train[,1], alpha = 0.0)
      eff = diag(marginalEffects(m, data = train[,-1],alpha = 0.0, interactions = FALSE)$mean)
      pred = predict(m, newx = test[,-1], alpha = 0.0)
      result[[7]] = list(eff, rmse(test[,1], pred), true_effs)
      
      
      ## L1 + L2 
      m = cva.glmnet(train[,-1], train[,1], alpha = 0.2)
      eff = diag(marginalEffects(m, data = train[,-1],alpha = 0.2, interactions = FALSE)$mean)
      pred = predict(m, newx = test[,-1], alpha = 0.2)
      result[[8]] = list(eff, rmse(test[,1], pred), true_effs)
      
      
      return(result)
    })
  parallel::stopCluster(cl)
  return(result_list)
}
sim = function(Sigma) {
  
  n_effs = rpois(1, 20)
  effs = sample(c(-1, 1), n_effs, replace = TRUE) * rexp(n_effs, 2)
  effs = c(effs, rep(0, 100-length(effs)))
  return(list(simulate(r = Sigma , effs = effs, n = 100*2), true_effs = effs))
}


results = get_result(sim)
saveRDS(results, "results/data_poor_small_random.RDS")


sim = function(Sigma) {
  
  n_effs = rpois(1, 20)
  effs = sample(c(-1, 1), n_effs, replace = TRUE) * rexp(n_effs, 2)
  effs = c(effs, rep(0, 100-length(effs)))
  return(list(simulate(r = Sigma , effs = effs, n = 600*2), true_effs = effs))
}


results = get_result(sim)
saveRDS(results, "results/data_poor_mid_random.RDS")


sim = function(Sigma) {
  
  n_effs = rpois(1, 20)
  effs = sample(c(-1, 1), n_effs, replace = TRUE) * rexp(n_effs, 2)
  effs = c(effs, rep(0, 100-length(effs)))
  return(list(simulate(r = Sigma , effs = effs, n = 2000*2), true_effs = effs))
}


results = get_result(sim)
saveRDS(results, "results/data_poor_big_random.RDS")

