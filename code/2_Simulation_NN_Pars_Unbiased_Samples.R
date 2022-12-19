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
NN = 300
activations = sample(c("relu", "leaky_relu", "tanh", "selu", "elu", "celu", "gelu"), size = NN, replace = TRUE)
sgd = runif(NN, 0, 1)
depth = 3 #sample(1:20, NN, replace = TRUE)
width = 50 #ceiling( rexp(NN, rate = 0.05))+5
dropout = runif(NN, 0, 0.3)
alpha = runif(NN, 0, 1.0)
lambda = runif(NN, 0.005, 0.4)**2
pars = data.frame(activations, sgd, depth, width, dropout, alpha, lambda)
pars$bias_1 = NA
pars$bias_5 = NA
pars$bias_0 = NA
pars$rmse = NA

get_result = function(sim ) {
  cl = parallel::makeCluster(20L)
  parallel::clusterExport(cl, varlist = ls(envir = .GlobalEnv))
  parallel::clusterEvalQ(cl, {library(ranger);library(xgboost);library(cito);library(glmnet);library(glmnetUtils);Sys.setenv(OMP_NUM_THREADS=1L);torch::torch_set_num_threads(1L);torch::torch_set_num_interop_threads(1L)
  })
  result_list = 
    parallel::parLapply(cl, 1:nrow(pars), function(i) {
      #library(torch)
      
      
      parameter = pars[i,]
      
      try({
        res= 
          sapply(1:10, function(j) {
          
          Sigma = cov2cor(rWishart(1, N_pred, diag(1.0, N_pred))[,,1])
          data = sim(Sigma)
          
          ind = nrow(data)/2
          
          train = data[1:ind,]
          test = data[(ind+1):nrow(data),]
          
          rmse = function(y, y_hat) sqrt(mean((y-y_hat)**2))
          
          ## DNN
          m = cito::dnn(Y~., data = as.data.frame(train), 
                        activation = rep(parameter$activations, parameter$depth),
                        hidden = rep(parameter$width, parameter$depth),
                        verbose = TRUE, 
                        epochs = 300,
                        lambda = parameter$lambda,
                        alpha = parameter$alpha,
                        batchsize = max(1, floor(nrow(train)*parameter$sgd)), 
                        lr = 0.05,
                        plot=FALSE, 
                        lr_scheduler = config_lr_scheduler("reduce_on_plateau", factor = 0.90, patience = 4), early_stopping = 8)
          eff = diag(marginalEffects(m, interactions = FALSE)$mean)[c(1,2, 5)]
          bias = eff - c(1, 0, 1)
          
          pred = predict(m, newdata = data.frame(test))
          
          return(c(bias,  rmse(test[,1], pred)))
          
          })
        parameter$bias_1 = mean(res[1,])
        parameter$bias_5 = mean(res[3,])
        parameter$bias_0 = mean(res[2,])
        parameter$rmse =   mean(res[4, ])
        #parameter$var = apply(res, 1, var)[1]
      
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
saveRDS(results, file = "results/NN_pars_100_100_SS.RDS")



N_pred = 100
sim = function(Sigma) {
  return(
    simulate(r = Sigma ,
             effs = c(1.0, 0.0, 0.0, 0.0, 1, rep(0, 95)), 
             n = 600*2)) 
}
system.time({results = get_result(sim)})
saveRDS(results, file = "results/NN_pars_600_100_SS.RDS")



N_pred = 100
sim = function(Sigma) {
  return(
    simulate(r = Sigma ,
             effs = c(1.0, 0.0, 0.0, 0.0, 1, rep(0, 95)),
             n = 2000*2))
}
system.time({results = get_result(sim)})
saveRDS(results, file = "results/NN_pars_2000_100_SS.RDS")


N_pred = 10
sim = function(Sigma) {
  return(
    simulate(r = Sigma ,
             effs = c(1.0, 0.0, 0.0, 0.0, 1, rep(0, 5)),
             n = 100*2))
}
system.time({results = get_result(sim)})
saveRDS(results, file = "results/NN_pars_100_10_SS.RDS")


N_pred = 100
sim = function(Sigma) {
  return(
    simulate(r = Sigma ,
             effs = c(1.0, 0.0, 0.0, 0.0, 1, rep(0, 95)),
             n = 1000*2))
}
system.time({results = get_result(sim)})
saveRDS(results, file = "results/NN_pars_1000_100_SS.RDS")



N_pred = 10
sim = function(Sigma) {
  return(
    simulate(r = Sigma ,
             effs = c(1.0, 0.0, 0.0, 0.0, 1, rep(0, 5)),
             n = 1000*2))
}
system.time({results = get_result(sim)})
saveRDS(results, file = "results/NN_pars_1000_10_SS.RDS")




