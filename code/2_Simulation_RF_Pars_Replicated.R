library(ranger)
library(xgboost)
library(torch)
library(iml)
library(cito)
library(glmnet)
library(glmnetUtils)
library(MASS)
set.seed(2)

Sys.setenv(OMP_NUM_THREADS=5)

source("code/AME.R")
source("code/Scenarios.R")



set.seed(42)
NN = 1000
mtry = runif(NN, 0, 1)
min.node.size = sample(2:70, NN, replace = TRUE)
max.depth = sample(2:50, NN, replace = TRUE)
regularization.factor = runif(NN, 0, 1)

pars = data.frame(mtry, min.node.size, max.depth, regularization.factor)
pars$eff_1 = NA
pars$eff_2 = NA
pars$pred_bias = NA
pars$pred_var = NA
pars$pred_mse = NA

get_result = function(data ) {
  cl = parallel::makeCluster(20L)
  parallel::clusterExport(cl, varlist = ls(envir = .GlobalEnv))
  parallel::clusterEvalQ(cl, {library(ranger);library(xgboost);library(cito);library(glmnet);library(glmnetUtils);Sys.setenv(OMP_NUM_THREADS=1L);torch::torch_set_num_threads(1L);torch::torch_set_num_interop_threads(1L)
  })
  result_list = 
    parallel::parLapply(cl, 1:nrow(pars), function(i) {
      #library(torch)
      
      
      parameter = pars[i,]
      
      try({
        
        ind = nrow(data)/2
        
        train = data[1:ind,]
        test = data[(ind+1):nrow(data),]
        
        mse = function(y, y_hat) (mean((y-y_hat)**2))
        
        ## RF
        m = ranger(Y ~., data = data.frame(train), 
                   mtry = max(1, floor(parameter$mtry*ncol(train))),
                   min.node.size = parameter$min.node.size,
                   max.depth = parameter$max.depth,
                   regularization.factor = parameter$regularization.factor,
                   num.trees = 100L,num.threads = 3L)
        eff = diag(marginalEffects(m, data = data.frame(train), interactions=FALSE, max_indices = 2)$mean)
        predictions = predict(m, data = data.frame(test))$predictions
        
        parameter$eff_1 = eff[1]
        parameter$eff_2 = eff[2]
        parameter$pred_bias = mean(test[,1]) - mean(predictions)
        parameter$pred_var = var(predictions)
        parameter$pred_mse = mse(test[,1], predictions)
      
      }, silent = FALSE)
      
      return(parameter)
    })
  parallel::stopCluster(cl)
  return(result_list)
}

results = lapply(1:20, function(replicate) {
  print(replicate)
  N_pred = 100
  Sigma = trialr::rlkjcorr(1, N_pred, 2)
  effs = c(1, seq(0, 1, length.out = 99))
  sim = function(Sigma) {
    return(
      simulate(r = Sigma ,
               effs = effs,
               n = 100*2))
  }
  data = sim(Sigma)
  res = get_result(data)
  return(do.call(rbind, res))
})


saveRDS(results, file = "results/RF_pars_100_100_replicate.RDS")



results = lapply(1:20, function(replicate) {
  print(replicate)
  N_pred = 100
  Sigma = trialr::rlkjcorr(1, N_pred, 2)
  effs = c(1, seq(0, 1, length.out = 99))
  sim = function(Sigma) {
    return(
      simulate(r = Sigma ,
               effs = effs,
               n = 600*2))
  }
  data = sim(Sigma)
  res = get_result(data)
  return(do.call(rbind, res))
})


saveRDS(results, file = "results/RF_pars_600_100_replicate.RDS")



results = lapply(1:20, function(replicate) {
  print(replicate)
  N_pred = 100
  Sigma = trialr::rlkjcorr(1, N_pred, 2)
  effs = c(1, seq(0, 1, length.out = 99))
  sim = function(Sigma) {
    return(
      simulate(r = Sigma ,
               effs = effs,
               n = 2000*2))
  }
  data = sim(Sigma)
  res = get_result(data)
  return(do.call(rbind, res))
})


saveRDS(results, file = "results/RF_pars_2000_100_replicate.RDS")



# N_pred = 100
# sim = function(Sigma) {
#   return(
#     simulate(r = Sigma ,
#              effs = effs, 
#              n = 600*2)) 
# }
# system.time({results = get_result(sim)})
# saveRDS(results, file = "results/RF_pars_600_100.RDS")
# 
# 
# 
# N_pred = 100
# sim = function(Sigma) {
#   return(
#     simulate(r = Sigma ,
#              effs = effs,
#              n = 2000*2))
# }
# system.time({results = get_result(sim)})
# saveRDS(results, file = "results/RF_pars_2000_100.RDS")
# 
# 
# 
# N_pred = 100
# sim = function(Sigma) {
#   return(
#     simulate(r = Sigma ,
#              effs = effs,
#              n = 1000*2))
# }
# system.time({results = get_result(sim)})
# saveRDS(results, file = "results/RF_pars_1000_100.RDS")
# 
# effs = c(1, seq(0, 1, length.out = 9))
# 
# N_pred = 10
# sim = function(Sigma) {
#   return(
#     simulate(r = Sigma ,
#              effs = effs, 
#              n = 100*2)) 
# }
# system.time({results = get_result(sim)})
# saveRDS(results, file = "results/RF_pars_100_10.RDS")
# 
# 
# N_pred = 10
# sim = function(Sigma) {
#   return(
#     simulate(r = Sigma ,
#              effs = effs,
#              n = 1000*2))
# }
# system.time({results = get_result(sim)})
# saveRDS(results, file = "results/RF_pars_1000_10.RDS")
