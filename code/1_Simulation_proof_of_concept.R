library(ranger)
library(xgboost)
library(torch)
library(iml)
library(cito)
library(glmnet)
library(glmnetUtils)

Sys.setenv(OMP_NUM_THREADS=3)

source("code/AME.R")
source("code/Scenarios.R")

torch::torch_set_num_threads(3L)

get_result = function(sim ) {
  samples = 100L
  result_list = vector("list", samples)
  
  cl = parallel::makeCluster(50L)
  parallel::clusterExport(cl, varlist = ls(envir = .GlobalEnv))
  parallel::clusterEvalQ(cl, {library(ranger);library(xgboost);library(cito);library(glmnet);library(glmnetUtils);Sys.setenv(OMP_NUM_THREADS=3)
})
  result_list = 
    parallel::parLapply(cl, 1:samples, function(i) {
      #library(torch)
      data = sim()
      
      train = data[1:1000,]
      test = data[1001:2000,]
      
      result = vector("list", 8L)
      
      rmse = function(y, y_hat) sqrt(mean((y-y_hat)**2))
      
      ## LM
      m = lm(Y~., data = data.frame(train))
      eff = diag(marginalEffects(m)$mean)
      pred = predict(m, newdata = data.frame(test))
      result[[1]] = list(eff, rmse(test[,1], pred))
      
      
      ## RF
      m = ranger(Y ~., data = data.frame(train), 
                 num.threads = 3L)
      eff = diag(marginalEffects(m, data = data.frame(train))$mean)
      pred = predict(m, data = data.frame(test))$predictions
      result[[2]] = list(eff, rmse(test[,1], pred))

      ## BRT
      m = xgboost::xgboost(data=xgboost::xgb.DMatrix(as.matrix(train)[,-1],
                                                     label = (as.matrix(train)[,1, drop=FALSE])),
                           nrounds = 100,
                           lambda = 0,
                           objective="reg:squarederror", nthread = 1, verbose = 0)
      (eff = diag(marginalEffects(m, data = data.frame(train)[,-1])$mean))
      pred = predict(m, newdata = xgboost::xgb.DMatrix(test[,-1]))
      result[[3]] = list(eff, rmse(test[,1], pred))

      ## DNN
      m = cito::dnn(Y~., data = as.data.frame(train), 
                    hidden = rep(50L, 3),
                    verbose = FALSE, 
                    plot=FALSE, lambda = 0.00, alpha = 1., 
                    epochs = 500,
                    lr = 0.01,
                    lr_scheduler = config_lr_scheduler("reduce_on_plateau", factor = 0.80, patience = 7), early_stopping = 20)
      eff = diag(marginalEffects(m)$mean)
      pred = predict(m, newdata = data.frame(test))
      result[[4]] = list(eff, rmse(test[,1], pred))
      
      
      ## DNN with dropout
      
      m = cito::dnn(Y~., data = as.data.frame(train), 
                    hidden = rep(50L, 3),
                    batchsize = 50, 
                    verbose = FALSE, 
                    plot=FALSE, lambda = 0.00, alpha = 1., 
                    epochs = 500,
                    lr = 0.01,
                    lr_scheduler = config_lr_scheduler("reduce_on_plateau", factor = 0.80, patience = 7), early_stopping = 20)
      eff = diag(marginalEffects(m)$mean)
      pred = predict(m, newdata = data.frame(test))
      result[[5]] = list(eff, rmse(test[,1], pred))
      
      
      
      ## L1
      m = cva.glmnet(train[,-1], train[,1], alpha = 1.0)
      eff = diag(marginalEffects(m, data = train[,-1],alpha = 1.0)$mean)
      pred = predict(m, newx = test[,-1], alpha = 1.0)
      result[[6]] = list(eff, rmse(test[,1], pred))
      
      
      ## L2 
      m = cva.glmnet(train[,-1], train[,1], alpha = 0.0)
      eff = diag(marginalEffects(m, data = train[,-1],alpha = 0.0)$mean)
      pred = predict(m, newx = test[,-1], alpha = 0.0)
      result[[7]] = list(eff, rmse(test[,1], pred))
      
      
      ## L1 + L2 
      m = cva.glmnet(train[,-1], train[,1], alpha = 0.2)
      eff = diag(marginalEffects(m, data = train[,-1],alpha = 0.2)$mean)
      pred = predict(m, newx = test[,-1], alpha = 0.2)
      result[[8]] = list(eff, rmse(test[,1], pred))
      
    
      return(result)
    })
  parallel::stopCluster(cl)
  return(result_list)
}

sim = function() simulate(r = 0.9, effs = c(1, -0.5, 0, 0, 1),n = 2000)
results = get_result(sim)
saveRDS(results, "results/confounder_unequal.RDS")

sim = function() simulate(r = 0.9, effs = c(1, 0.5, 0, 0, 1),n = 2000)
results = get_result(sim)
saveRDS(results, "results/confounder.RDS")



sim = function() simulate(r = 0.9, effs = c(1, 0, 0, 0, 1),n = 2000)
results = get_result(sim)
saveRDS(results, "results/collinearity_0.90.RDS")

sim = function() simulate(r = 0.5, effs = c(1, 0, 0, 0, 1),n = 2000)
results = get_result(sim)
saveRDS(results, "results/collinearity_0.5.RDS")

sim = function() simulate(r = 0.99, effs = c(1, 0, 0, 0, 1),n = 2000)
results = get_result(sim)
saveRDS(results, "results/collinearity_0.99.RDS")

sim = function() simulate(effs = c(1.0, 0.0, 1.0, 0.0, 1.0), r = 0.0,n = 2000)
results = get_result(sim)
saveRDS(results, "results/effects.RDS")

sim = function() simulate( effs = c(0.0, 0.0, 0.0, 0.0, 0.0), r = 0.0,n = 2000)
results = get_result(sim)
saveRDS(results, "results/no_effects.RDS")

