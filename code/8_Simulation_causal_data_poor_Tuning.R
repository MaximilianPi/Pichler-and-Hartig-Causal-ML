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
source("code/tuners.R")

get_result = function(sim ) {
  samples = 20L
  result_list = vector("list", samples)
  
  cl = parallel::makeCluster(20L)
  parallel::clusterExport(cl, varlist = ls(envir = .GlobalEnv))
  parallel::clusterEvalQ(cl, {library(ranger);library(xgboost);library(cito);library(glmnet);library(glmnetUtils);Sys.setenv(OMP_NUM_THREADS=3);torch::torch_set_num_threads(1L);torch::torch_set_num_interop_threads(1L)
})
  result_list = 
    parallel::parLapply(cl, 1:samples, function(i) {
      #library(torch)
      
      
      
      dev = sample.int(4, 1)-1
      Sys.setenv(CUDA_VISIBLE_DEVICES=dev)
      device = "cpu"
      
      
      print(i)
      Sigma = cov2cor(rWishart(1, 100, diag(1.0, 100))[,,1])
      
      data = sim(Sigma)
      
      ind = nrow(data)/2
      
      train = data[1:ind,]
      test = data[(ind+1):nrow(data),]
      
      result = vector("list", 8L)
      
      mse = function(y, y_hat) (mean((y-y_hat)**2))
      
      ## LM
      # m = lm(Y~., data = data.frame(train))
      # eff = diag(marginalEffects(m, interactions=FALSE, max_indices = 5)$mean)
      # pred = predict(m, newdata = data.frame(test))
      # result[[1]] = list(eff, rmse(test[,1], pred))
      # 
      # 
      # ## RF
      # m = ranger(Y ~., 
      #            data = data.frame(train), 
      #            max.depth = RF_max.depth,
      #            min.node.size = RF_min.node.size,
      #            mtry = max(1, floor(RF_mtry*ncol(train))),
      #            num.trees = 100L,
      #            num.threads = 3L)
      # eff = diag(marginalEffects(m, data = data.frame(train), interactions=FALSE, max_indices = 5)$mean)
      # pred = predict(m, data = data.frame(test))$predictions
      # result[[2]] = list(eff, rmse(test[,1], pred))
      # 
      # ## BRT
      # m = xgboost::xgboost(data=xgboost::xgb.DMatrix(as.matrix(train)[,-1],
      #                                                label = (as.matrix(train)[,1, drop=FALSE])),
      #                      nrounds = BRT_max_tree,
      #                      eta = BRT_eta,
      #                      max_depth = BRT_max_depth,
      #                      subsample = BRT_subsample, 
      #                      lambda = BRT_lambda,
      #                      objective="reg:squarederror", nthread = 1, verbose = 0)
      # 
      # eff = diag(marginalEffects(m, data = data.frame(train)[,-1], interactions=FALSE, max_indices = 5)$mean)
      # pred = predict(m, newdata = xgboost::xgb.DMatrix(test[,-1]))
      # result[[3]] = list(eff, rmse(test[,1], pred))
      
      
      ## DNN
      
      results = DNN_tuner(data = train, hyper_samples = 30L, true_eff = c(1, 0, 0, 0, 1), cv = 5, device = "cuda")
      
      
      eff_eff = diag(marginalEffects(results$models$m_eff, interactions = FALSE, max_indices = 5, device = device, data = test)$mean)
      eff_pred= diag(marginalEffects(results$models$m_pred, interactions = FALSE, max_indices = 5, device = device, data = test)$mean)
      
      pred_eff = predict(results$models$m_eff, newdata = data.frame(test), device = device)
      pred_pred = predict(results$models$m_pred, newdata = data.frame(test), device = device)
      
      result[[1]] = list( eff_eff, eff_pred, mse(test[,1], pred_eff), mse(test[,1], pred_pred) )
      
      # 
      # 
      # result[[5]] = list(eff, rmse(test[,1], pred))
      # 
      # 
      # ## Elastic-net tuned
      # m = glmnet(train[,-1], train[,1], alpha = EN_alpha, lambda = EN_lambda)
      # eff = diag(marginalEffects(m, data = train[,-1],alpha = EN_alpha, lambda = EN_lambda, interactions = FALSE, max_indices = 5)$mean)
      # pred = predict(m, newx = test[,-1])
      # result[[8]] = list(eff, rmse(test[,1], pred))
      # 
      # result[[6]] = result[[7]] =  list(eff, rmse(test[,1], pred))
      

      return(result)
    })
  parallel::stopCluster(cl)
  return(result_list)
}

##### Predictive RMSE #######

sim = function(Sigma) {
  return(
    simulate(r = Sigma ,
    effs = c(1.0, 0.0, 0.0, 0.0, 1, rep(0, 95)), 
    n = 100*2)) 
}


results = get_result(sim)
saveRDS(results, "results/data_poor_small_Tuner.RDS")


sim = function(Sigma) {
  return(
    simulate(r = Sigma ,
             effs = c(1.0, 0.0, 0.0, 0.0, 1, rep(0, 95)),
             n = 600*2))
}


results = get_result(sim)
saveRDS(results, "results/data_poor_mid_Tuner.RDS")


sim = function(Sigma) {
  return(
    simulate(r = Sigma ,
             effs = c(1.0, 0.0, 0.0, 0.0, 1, rep(0, 95)),
             n = 2000*2))
}

results = get_result(sim)
saveRDS(results, "results/data_poor_big_Tuner.RDS")



##### MSE effect #######
# 
# source("code/hyper-parameter/BIAS_hyper_param_config_100_100.R")
# sim = function(Sigma) {
#   return(
#     simulate(r = Sigma ,
#              effs = c(1.0, 0.0, 0.0, 0.0, 1, rep(0, 95)), 
#              n = 100*2)) 
# }
# 
# 
# results = get_result(sim)
# saveRDS(results, "results/data_poor_small_BIAS.RDS")
# 
# 
# source("code/hyper-parameter/BIAS_hyper_param_config_600_100.R")
# sim = function(Sigma) {
#   return(
#     simulate(r = Sigma ,
#              effs = c(1.0, 0.0, 0.0, 0.0, 1, rep(0, 95)),
#              n = 600*2))
# }
# 
# 
# results = get_result(sim)
# saveRDS(results, "results/data_poor_mid_BIAS.RDS")
# 
# 
# source("code/hyper-parameter/BIAS_hyper_param_config_2000_100.R")
# sim = function(Sigma) {
#   return(
#     simulate(r = Sigma ,
#              effs = c(1.0, 0.0, 0.0, 0.0, 1, rep(0, 95)),
#              n = 2000*2))
# }
# 
# results = get_result(sim)
# saveRDS(results, "results/data_poor_big_BIAS.RDS")

