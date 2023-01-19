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
activations = sample(c("relu", "leaky_relu", "tanh", "selu", "elu", "celu", "gelu"), size = NN, replace = TRUE)
sgd = runif(NN, 0, 1)
depth = sample(1:8, NN, replace = TRUE)
width = sample(2:50, NN, replace = TRUE)
dropout = 0 # runif(NN, 0, 0.3)
alpha = runif(NN, 0, 1.0)
lambda = runif(NN, 0.005, 0.4)**2
pars = data.frame(activations, sgd, depth, width, dropout, alpha, lambda)

pars$eff_1 = NA
pars$eff_2 = NA
pars$pred_bias = NA
pars$pred_var = NA
pars$pred_mse = NA

get_result = function(data ) {
  cl = parallel::makeCluster(50L)
  nodes = unlist(parallel::clusterEvalQ(cl, paste(Sys.info()[['nodename']], Sys.getpid(), sep='-')))
  parallel::clusterExport(cl, varlist = ls(envir = .GlobalEnv))
  parallel::clusterEvalQ(cl, {library(ranger);library(xgboost);library(cito);library(glmnet);library(glmnetUtils);Sys.setenv(OMP_NUM_THREADS=1L);torch::torch_set_num_threads(1L);torch::torch_set_num_interop_threads(1L)
  })
  result_list = 
    parallel::parLapplyLB(cl, 1:nrow(pars), function(i) {
      #library(torch)
      
      # who am I
      myself = paste(Sys.info()[['nodename']], Sys.getpid(), sep='-')
      dist = cbind(nodes,0:3)
      dev = as.integer(as.numeric(dist[which(dist[,1] %in% myself, arr.ind = TRUE), 2]))
      
      
      Sys.setenv(CUDA_VISIBLE_DEVICES=dev)
      device = "cuda"
      
      parameter = pars[i,]
      
      
      try({
        
        ind = nrow(data)/2
        train = data[1:ind,]
        test = data[(ind+1):nrow(data),]
        mse = function(y, y_hat) (mean((y-y_hat)**2))
        
        ## DNN
        
        m = 
          cito::dnn(Y~., data = as.data.frame(train), 
                    activation = rep(parameter$activations, parameter$depth),
                    hidden = rep(parameter$width, parameter$depth),
                    verbose = FALSE, 
                    epochs = 500,
                    #validation = 0.2,
                    lambda = parameter$lambda,
                    alpha = parameter$alpha,
                    batchsize = max(1, floor(nrow(train)*parameter$sgd)), 
                    lr = 0.01,
                    plot=FALSE, 
                    device = device,
                    lr_scheduler = config_lr_scheduler("reduce_on_plateau", factor = 0.80, patience = 7), early_stopping = 20)
        m$use_model_epoch = length(m$weights)
        eff = diag(marginalEffects(m, interactions = FALSE, max_indices = 2, device = device)$mean)[c(1,2)]
        predictions = predict(m, newdata = data.frame(test), device = device)
        
        parameter$eff_1 = eff[1]
        parameter$eff_2 = eff[2]
        parameter$pred_bias = mean(test[,1]) - mean(predictions)
        parameter$pred_var = var(predictions)
        parameter$pred_mse = mse(test[,1], predictions)
      
      }, silent = FALSE )
      
      return(parameter)
    })
  parallel::stopCluster(cl)
  return(result_list)
}

set.seed(42)
results = lapply(1:20, function(replicate) {
  print(replicate)
  print(Sys.time())
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


saveRDS(results, file = "results/NN_pars_100_100_replicate.RDS")


# set.seed(42)
# results = lapply(1:20, function(replicate) {
#   print(replicate)
#   print(Sys.time())
#   N_pred = 100
#   Sigma = trialr::rlkjcorr(1, N_pred, 2)
#   effs = c(1, seq(0, 1, length.out = 99))
#   sim = function(Sigma) {
#     return(
#       simulate(r = Sigma ,
#                effs = effs,
#                n = 600*2))
#   }
#   data = sim(Sigma)
#   res = get_result(data)
#   return(do.call(rbind, res))
# })
# 
# saveRDS(results, file = "results/NN_pars_600_100_replicate.RDS")

set.seed(42)
results = lapply(1:20, function(replicate) {
  print(replicate)
  print(Sys.time())
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

saveRDS(results, file = "results/NN_pars_2000_100_replicate.RDS")
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
