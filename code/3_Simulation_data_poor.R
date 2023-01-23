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

get_result = function(sim) {
  samples = 1000L
  result_list = vector("list", samples)
  cl = parallel::makeCluster(25L)
  nodes = unlist(parallel::clusterEvalQ(cl, paste(Sys.info()[['nodename']], Sys.getpid(), sep='-')))
  
  parallel::clusterExport(cl, varlist = ls(envir = .GlobalEnv))
  parallel::clusterEvalQ(cl, {library(ranger);library(xgboost);library(cito);library(glmnet);library(glmnetUtils);Sys.setenv(OMP_NUM_THREADS=3);torch::torch_set_num_threads(1L);torch::torch_set_num_interop_threads(1L)
  })
  result_list = 
    parallel::parLapplyLB(cl, 1:samples, function(i) {
      #library(torch)
      
      
      # who am I
      myself = paste(Sys.info()[['nodename']], Sys.getpid(), sep='-')
      dist = cbind(nodes,0:3)
      dev = as.integer(as.numeric(dist[which(dist[,1] %in% myself, arr.ind = TRUE), 2]))
      
      
      Sys.setenv(CUDA_VISIBLE_DEVICES=dev)
      device = "cuda"
      
      print(i)
      
      Sigma = trialr::rlkjcorr(1, N_pred, 2)
      
      data = sim(Sigma)
      ind = nrow(data)/2
      
      train = data[1:ind,]
      test = data[(ind+1):nrow(data),]
      
      result = vector("list", 5L)
      
      mse = function(y, y_hat) sqrt(mean((y-y_hat)**2))
      
      ## LM
      m = lm(Y~., data = data.frame(train))
      eff = diag(marginalEffects(m, interactions=FALSE, max_indices = 2)$mean)
      pred = predict(m, newdata = data.frame(test))
      result[[1]] = c(eff, mse(test[,1], pred), mean(test[,1]) - mean(pred), var(pred))
      
      
      ## RF
      m = ranger(Y ~.,
                 data = data.frame(train),
                 max.depth = RF_max.depth,
                 min.node.size = RF_min.node.size,
                 mtry = max(1, floor(RF_mtry*ncol(train))),
                 num.trees = 100L,
                 num.threads = 3L)
      eff = diag(marginalEffects(m, data = data.frame(train), interactions=FALSE, max_indices = 2)$mean)
      pred = predict(m, data = data.frame(test))$predictions
      result[[2]] = c(eff, mse(test[,1], pred), mean(test[,1]) - mean(pred), var(pred))
      
      ## BRT
      m = xgboost::xgboost(data=xgboost::xgb.DMatrix(as.matrix(train)[,-1],
                                                     label = (as.matrix(train)[,1, drop=FALSE])),
                           nrounds = BRT_max_tree,
                           eta = BRT_eta,
                           max_depth = BRT_max_depth,
                           subsample = BRT_subsample,
                           lambda = BRT_lambda,
                           objective="reg:squarederror", nthread = 1, verbose = 0)
      
      eff = diag(marginalEffects(m, data = data.frame(train)[,-1], interactions=FALSE, max_indices = 2)$mean)
      pred = predict(m, newdata = xgboost::xgb.DMatrix(test[,-1]))
      result[[3]] = c(eff, mse(test[,1], pred), mean(test[,1]) - mean(pred), var(pred))
      
      
      ## DNN
      try({
        m = cito::dnn(Y~., data = as.data.frame(train),
                      activation = rep(NN_activations, NN_depth),
                      hidden = rep(NN_width, NN_depth),
                      batchsize = max(1, floor(NN_sgd*nrow(train))),
                      verbose = FALSE,
                      plot=FALSE,
                      lambda = NN_lambda,
                      alpha = NN_alpha,
                      epochs = 500,
                      device = device,
                      lr = 0.01,
                      lr_scheduler = config_lr_scheduler("reduce_on_plateau", factor = 0.80, patience = 7), early_stopping = 20)
        m$use_model_epoch = length(m$weights)
        eff = diag(marginalEffects(m, interactions = FALSE, max_indices = 2, device = device)$mean)
        pred = predict(m, newdata = data.frame(test), device = device)
        result[[4]] = c(eff, mse(test[,1], pred), mean(test[,1]) - mean(pred), var(pred))
      }, silent = TRUE)
      
      
      ## Elastic-net tuned
      m = glmnet(train[,-1], train[,1], alpha = EN_alpha, lambda = EN_lambda)
      eff = diag(marginalEffects(m, data = train[,-1],alpha = EN_alpha, lambda = EN_lambda, interactions = FALSE, max_indices = 2)$mean)
      pred = predict(m, newx = test[,-1])
      result[[5]] = c(eff, mse(test[,1], pred), mean(test[,1]) - mean(pred), var(pred))
      
      
      return(result)
    })
  parallel::stopCluster(cl)
  return(result_list)
}

##### Predictive RMSE #######


N_pred = 100
effs = c(1, seq(0, 1, length.out = 99))

source("code/hyper-parameter/MSE_hyper_param_config_50_100.R")
sim = function(Sigma) {
  return(
    simulate(r = Sigma ,
             effs = effs,
             n = 50*2))
}


results = get_result(sim)

saveRDS(results, "results/data_poor_small_MSE.RDS")

source("code/hyper-parameter/MSE_hyper_param_config_100_100.R")
sim = function(Sigma) {
  return(
    simulate(r = Sigma ,
             effs = effs,
             n = 100*2))
}


results = get_result(sim)

saveRDS(results, "results/data_poor_mid_MSE.RDS")


source("code/hyper-parameter/MSE_hyper_param_config_600_100.R")
sim = function(Sigma) {
  return(
    simulate(r = Sigma ,
             effs = effs,
             n = 600*2))
}

results = get_result(sim)
saveRDS(results, "results/data_poor_big_MSE.RDS")



##### MSE effect #######


source("code/hyper-parameter/BIAS_hyper_param_config_50_100.R")
sim = function(Sigma) {
  return(
    simulate(r = Sigma ,
             effs = effs,
             n = 50*2))
}


results = get_result(sim)
saveRDS(results, "results/data_poor_small_BIAS.RDS")


source("code/hyper-parameter/BIAS_hyper_param_config_100_100.R")
sim = function(Sigma) {
  return(
    simulate(r = Sigma ,
             effs = effs,
             n = 100*2))
}


results = get_result(sim)
saveRDS(results, "results/data_poor_mid_BIAS.RDS")


source("code/hyper-parameter/BIAS_hyper_param_config_600_100.R")
sim = function(Sigma) {
  return(
    simulate(r = Sigma ,
             effs = effs,
             n = 600*2))
}

results = get_result(sim)
saveRDS(results, "results/data_poor_big_BIAS.RDS")

