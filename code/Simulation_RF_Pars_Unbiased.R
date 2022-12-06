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



set.seed(42)
NN = 3000
mtry = sample(2:99, NN, replace = TRUE)
min.node.size = sample(2:70, NN, replace = TRUE)
max.depth = sample(2:50, NN, replace = TRUE)
regularization.factor = runif(NN, 0, 1)

pars = data.frame(mtry, min.node.size, max.depth, regularization.factor)
pars$bias_1 = NA
pars$bias_5 = NA
pars$bias_0 = NA
pars$rmse = NA

get_result = function(sim ) {
  cl = parallel::makeCluster(50L)
  parallel::clusterExport(cl, varlist = ls(envir = .GlobalEnv))
  parallel::clusterEvalQ(cl, {library(ranger);library(xgboost);library(cito);library(glmnet);library(glmnetUtils);Sys.setenv(OMP_NUM_THREADS=1L);torch::torch_set_num_threads(1L);torch::torch_set_num_interop_threads(1L)
  })
  result_list = 
    parallel::parLapply(cl, 1:nrow(pars), function(i) {
      #library(torch)
      
      
      parameter = pars[i,]
      
      try({
        res= 
          sapply(1:1, function(j) {
          
          Sigma = cov2cor(rWishart(1, N_pred, diag(1.0, N_pred))[,,1])
          data = sim(Sigma)
          
          ind = nrow(data)/2
          
          train = data[1:ind,]
          test = data[(ind+1):nrow(data),]
          
          rmse = function(y, y_hat) sqrt(mean((y-y_hat)**2))
          
          ## RF
          m = ranger(Y ~., data = data.frame(train), 
                     mtry = parameter$mtry,
                     min.node.size = parameter$min.node.size,
                     max.depth = parameter$max.depth,
                     regularization.factor = parameter$regularization.factor,
                     num.trees = 100L,num.threads = 3L)
          eff = diag(marginalEffects(m, data = data.frame(train), interactions=FALSE)$mean)[c(1,2, 5)]
          pred = predict(m, data = data.frame(test))$predictions
  
          bias = eff - c(1, 0, 1)
          
          return(c(bias,  rmse(test[,1], pred)))
          
          })
        parameter$bias_1 = res[1,1]
        parameter$bias_5 = res[3,1]
        parameter$bias_0 = res[2,1]
        parameter$rmse = res[4, 1]
      
      }, silent = FALSE)
      
      return(parameter)
    })
  parallel::stopCluster(cl)
  return(result_list)
}

N_pred = 100
sim = function(Sigma) {
  return(
    simulate(r = Sigma ,
             effs = c(1.0, 0.0, 0.0, 0.0, 1, rep(0, 95)),
             n = 100*2))
}
system.time({results = get_result(sim)})
saveRDS(results, file = "results/RF_pars_100_100.RDS")


N_pred = 10
sim = function(Sigma) {
  return(
    simulate(r = Sigma ,
             effs = c(1.0, 0.0, 0.0, 0.0, 1, rep(0, 5)),
             n = 100*2))
}
system.time({results = get_result(sim)})
saveRDS(results, file = "results/RF_pars_100_10.RDS")



N_pred = 100
sim = function(Sigma) {
  return(
    simulate(r = Sigma ,
             effs = c(1.0, 0.0, 0.0, 0.0, 1, rep(0, 95)), 
             n = 1000*2)) 
}
system.time({results = get_result(sim)})
saveRDS(results, file = "results/RF_pars_1000_100.RDS")



N_pred = 10
sim = function(Sigma) {
  return(
    simulate(r = Sigma ,
             effs = c(1.0, 0.0, 0.0, 0.0, 1, rep(0, 5)),
             n = 1000*2))
}
system.time({results = get_result(sim)})
saveRDS(results, file = "results/RF_pars_1000_10.RDS")

