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
  
  cl = parallel::makeCluster(10L)
  parallel::clusterExport(cl, varlist = ls(envir = .GlobalEnv))
  parallel::clusterEvalQ(cl, {library(ranger);library(xgboost);library(cito);library(glmnet);library(glmnetUtils);Sys.setenv(OMP_NUM_THREADS=3)
})
  result_list = 
    parallel::parLapply(cl, 1:samples, function(i) {
      #library(torch)
      data = sim()
      result = vector("list", 7L)
      
      result[[1]] = diag(marginalEffects(lm(Y~., data = data.frame(data)))$mean)
      
      result[[2]] = diag(marginalEffects(ranger(Y ~., data = data.frame(data), num.trees = 100L, num.threads = 3L), data = data.frame(data))$mean)
      
      result[[3]] = diag(marginalEffects(xgboost::xgboost(data=xgboost::xgb.DMatrix(as.matrix(data)[,-1],
                                                                                    label = (as.matrix(data)[,1, drop=FALSE])),
                                                          nrounds = 140L,
                                                          objective="reg:squarederror", nthread = 1, verbose = 0), data = data.frame(data)[,-1])$mean)
      
      result[[4]] = diag(marginalEffects(cito::dnn(Y~., data = as.data.frame(data), 
                                                   activation = rep("relu", 3),
                                                   hidden = rep(20L, 3),
                                                   verbose = FALSE, 
                                                   batchsize = 100L, 
                                                   plot=FALSE, lambda = 0.00, alpha = 1.))$mean)
      # L1
      result[[5]] = diag(marginalEffects(cva.glmnet(data[,-1], data[,1], alpha = 1.0), data = data[,-1],alpha = 1.0)$mean)
      
      # L2 
      result[[6]] = diag(marginalEffects(cva.glmnet(data[,-1], data[,1], alpha = 0.0), data = data[,-1],alpha = 0.0)$mean)
      
      # L1 + L2 
      result[[7]] = diag(marginalEffects(cva.glmnet(data[,-1], data[,1], alpha = 0.2), data = data[,-1],alpha = 0.2)$mean)
      
    
      return(result)
    })
  parallel::stopCluster(cl)
  return(result_list)
}

sim = function() simulate(r = 0.9, effs = c(1, 1, 0, 0, 1))
results = get_result(sim)
saveRDS(results, "results/confounder.RDS")

sim = function() simulate(r = 0.9, effs = c(-1, 0.5, 0, 0, 1))
results = get_result(sim)
saveRDS(results, "results/confounder_unequal.RDS")

sim = function() simulate(r = 0.9, effs = c(1, 0, 0, 0, 1))
results = get_result(sim)
saveRDS(results, "results/collinearity_0.90.RDS")

sim = function() simulate(r = 0.5, effs = c(1, 0, 0, 0, 1))
results = get_result(sim)
saveRDS(results, "results/collinearity_0.5.RDS")

sim = function() simulate(r = 0.99, effs = c(1, 0, 0, 0, 1))
results = get_result(sim)
saveRDS(results, "results/collinearity_0.99.RDS")

sim = function() simulate(effs = c(1.0, 0.5, 1.0, 0.0, 1.0), r = 0.0)
results = get_result(sim)
saveRDS(results, "results/effects.RDS")

sim = function() simulate( effs = c(0.0, 0.0, 0.0, 0.0, 0.0), r = 0.0)
results = get_result(sim)
saveRDS(results, "results/no_effects.RDS")

