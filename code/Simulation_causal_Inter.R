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

torch::torch_set_num_threads(5L)

get_result = function(sim ) {
  samples = 100L
  result_list = vector("list", samples)
  
  cl = parallel::makeCluster(10L)
  parallel::clusterExport(cl, varlist = ls(envir = .GlobalEnv))
  parallel::clusterEvalQ(cl, {library(ranger);library(xgboost);library(cito);library(glmnet);library(glmnetUtils);Sys.setenv(OMP_NUM_THREADS=3);torch::torch_set_num_threads(5L)
})
  result_list = 
    parallel::parLapply(cl, 1:samples, function(i) {
      #library(torch)
      print(i)
      Sigma = cov2cor(rWishart(1, 29, diag(1.0, 29))[,,1])
      
      data = sim(Sigma)
      result = vector("list", 6L)
      
      if(nrow(data) > 400) { result[[1]] = (marginalEffects(lm(Y~.^2, data = data.frame(data)))$mean) }
      
      result[[2]] = (marginalEffects(ranger(Y ~., data = data.frame(data), num.trees = 100L, num.threads = 3L), data = data.frame(data))$mean)
      
      result[[3]] = (marginalEffects(xgboost::xgboost(data=xgboost::xgb.DMatrix(as.matrix(data)[,-1],
                                                                                    label = (as.matrix(data)[,1, drop=FALSE])),
                                                          nrounds = 140L,
                                                          objective="reg:squarederror", nthread = 1, verbose = 0), data = data.frame(data)[,-1])$mean)
      
      # P = data[,-1][,c(1, 5)]%*%diag(result[[3]])[c(1, 5)] + (data[,-1][,1] * data[,-1][,2] )  * result[[3]][1, 2]+ (data[,-1][,3] * data[,-1][,4] )  * result[[3]][3, 4]
      # #P = predict(lm(Y~.^2, data = data.frame(data)))
      # plot( data[,1], P)
      # abline(1, 1)
      # summary(lm(P~data[,1]))
      # 
      # (sqrt((data[,1] - mean(P))**2 ))
      # mean( (mean(P) - P)**2  )
      # 
      # bias = (mean(P) - mean(data[,1]))**2
      # var = (var(P - mean(P)) )[1,1]
      # mean( ( P - data[,1])^2)
      bs = as.integer(ifelse(nrow(data) <400, 25, 75))
      result[[4]] = (marginalEffects(cito::dnn(Y~., data = as.data.frame(data), 
                                               activation = rep("relu", 6),
                                               hidden = rep(50L, 6),
                                               verbose = TRUE, 
                                               batchsize = bs, 
                                               epochs = 100L,
                                               shuffle = TRUE,
                                               loss = "mse",
                                               plot=FALSE, 
                                               lambda = 0.001, alpha = 1., 
                                               lr_scheduler = config_lr_scheduler("reduce_on_plateau", factor = 0.90, patience = 5)))$mean)
      # L1
      result[[5]] = 
        marginalEffects(cva.glmnet(model.matrix(~.^2, data = data.frame(data[,-1])), data[,1], alpha = 1.0), data = data[,-1], alpha = 1.0, formula = function(d) model.matrix(~.^2, data = data.frame(d)))$mean

      # L2 
      result[[6]] = 
        marginalEffects(cva.glmnet(model.matrix(~.^2, data = data.frame(data[,-1])), data[,1], alpha = 0.0), data = data[,-1], alpha = 0.0, formula = function(d) model.matrix(~.^2, data = data.frame(d)))$mean
      
      # L1 + L2 
      result[[7]] = 
        marginalEffects(cva.glmnet(model.matrix(~.^2, data = data.frame(data[,-1])), data[,1], alpha = 0.2), data = data[,-1], alpha = 0.2, formula = function(d) model.matrix(~.^2, data = data.frame(d)))$mean
      
      return(result)
    })
  parallel::stopCluster(cl)
  return(result_list)
}
sim = function(Sigma) {
  return(
    simulate(r = Sigma ,
    effs = c(1.0, 0.0, 0.0, 0.0, 1, rep(0, 24)), 
    inter = c(1.0, 1.0), 
    n = 300)) # 439
}


results = get_result(sim)
saveRDS(results, "results/interactions_small.RDS")


sim = function(Sigma) {
  return(
    simulate(r = Sigma ,
             effs = c(1.0, 0.0, 0.0, 0.0, 1, rep(0, 24)), 
             inter = c(1.0, 1.0), 
             n = 2000)) # 439
}


results = get_result(sim)
saveRDS(results, "results/interactions_big.RDS")



