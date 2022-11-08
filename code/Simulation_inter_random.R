library(ranger)
library(xgboost)
library(torch)
library(iml)
library(cito)
library(glmnet)
library(glmnetUtils)
library(MASS)
set.seed(42)
n_main = 15
n_inter = 30

effs_true = sample(c(rep(1, n_main), rep(0, 29-n_main)))
inter_true = sample(c(rep(1, n_inter), rep(0, 406-n_inter)))

Sys.setenv(OMP_NUM_THREADS=5)

source("code/AME.R")


simulate = function(r = 0.95, 
                    n = 1000, 
                    effs = rep(0, 5),
                    inter = NULL) {
  p = length(effs)
  if(!is.matrix(r)) {
    Sigma = diag(1.0, p)
    for(i in 1:(length(r)-1)) {
      Sigma[i, i+1]  = r[i]
      Sigma[i+1, i]  = r[i]
    }
    
  } else { Sigma = r}
  
  X = MASS::mvrnorm(n, mu = rep(0, p), Sigma = Sigma, empirical = TRUE)
  true_effs = diag(0, length(effs))
  diag(true_effs) = effs
  
  if(!is.null(inter)) {
    
    XX = model.matrix(~-1 + .^2, data = data.frame(X))
    Y = XX %*% (c(effs, inter)) + rnorm(n, sd = 0.3)
    
    counter = 1
    for(i in 1:(length(effs) - 1)) {
      for(j in (i+1):29) {
        true_effs[i, j] = true_effs[i,j] = inter[counter]
        counter = counter + 1
        print(c(i, j))
      }
    }
    
  } else {
    Y = X%*%effs + rnorm(n, sd = 0.3)
  }
  
  colnames(X) = paste0("X", 1:ncol(X))
  data = cbind(Y, X)
  colnames(data)[1] = "Y" 
  return(list(data = data, true_effs = true_effs, inter = inter, effs = effs))
}

torch::torch_set_num_threads(5L)

get_result = function(sim ) {
  samples = 100L
  result_list = vector("list", samples)
  
  cl = parallel::makeCluster(10L)
  parallel::clusterExport(cl, varlist = ls(envir = .GlobalEnv))
  parallel::clusterEvalQ(cl, {library(ranger);library(xgboost);library(cito);library(glmnet);library(glmnetUtils);Sys.setenv(OMP_NUM_THREADS=5);torch::torch_set_num_threads(5L)
})
  result_list = 
    parallel::parLapply(cl, 1:samples, function(i) {
      #library(torch)
      extract_bias = function(pred, simulation) {
        true_effs = simulation$effs
        true_inter = simulation$inter
        effs = diag(pred)
        inter = vector(mode = "numeric", 406)
        counter = 1
        for(i in 1:28) {
          for(j in (i+1):29) {
            inter[counter] = pred[i, j] 
            counter = counter + 1
          }
        }
        return(
            list(effs = effs, inter = inter, true_effs = true_effs, true_inter = true_inter)
        )
      }
      
      
      
      Sigma = cov2cor(rWishart(1, 29, diag(1.0, 29))[,,1])
      
      simulation = sim(Sigma)
      data = simulation$data
      result = vector("list", 7L)
      
      result[[1]] = extract_bias(marginalEffects(lm(Y~.^2, data = data.frame(data)))$mean, simulation )
      
      result[[2]] = extract_bias(marginalEffects(ranger(Y ~., data = data.frame(data), num.trees = 100L, num.threads = 3L), data = data.frame(data))$mean, simulation)
      
      result[[3]] = extract_bias(marginalEffects(xgboost::xgboost(data=xgboost::xgb.DMatrix(as.matrix(data)[,-1],
                                                                                    label = (as.matrix(data)[,1, drop=FALSE])),
                                                          nrounds = 140L,
                                                          objective="reg:squarederror", nthread = 1, verbose = 0), data = data.frame(data)[,-1])$mean, simulation)
      

    
      bs = as.integer(ifelse(nrow(data) <400, 25, 75))
      result[[4]] = extract_bias(marginalEffects(cito::dnn(Y~., data = as.data.frame(data), 
                                               activation = rep("relu", 6),
                                               hidden = rep(50L, 6),
                                               verbose = FALSE, 
                                               batchsize = bs, 
                                               epochs = 100L,
                                               shuffle = TRUE,
                                               loss = "mse",
                                               plot=FALSE, 
                                               lambda = 0.000, alpha = 1., 
                                               lr_scheduler = config_lr_scheduler("reduce_on_plateau", factor = 0.90, patience = 5)))$mean, simulation)
      
      result[[5]] = extract_bias(marginalEffects(cito::dnn(Y~., data = as.data.frame(data), 
                                                           activation = rep("relu", 6),
                                                           hidden = rep(50L, 6),
                                                           verbose = FALSE, 
                                                           batchsize = bs, 
                                                           epochs = 100L,
                                                           shuffle = TRUE,
                                                           dropout = 0.3,
                                                           loss = "mse",
                                                           plot=FALSE, 
                                                           lambda = 0.000, alpha = 1., 
                                                           lr_scheduler = config_lr_scheduler("reduce_on_plateau", factor = 0.90, patience = 5)))$mean, simulation)
      # L1
      result[[6]] = 
        extract_bias(marginalEffects(cva.glmnet(model.matrix(~.^2, data = data.frame(data[,-1])), data[,1], alpha = 1.0), data = data[,-1], alpha = 1.0, formula = function(d) model.matrix(~.^2, data = data.frame(d)))$mean, simulation)

      # L2 
      result[[7]] = 
        extract_bias(marginalEffects(cva.glmnet(model.matrix(~.^2, data = data.frame(data[,-1])), data[,1], alpha = 0.0), data = data[,-1], alpha = 0.0, formula = function(d) model.matrix(~.^2, data = data.frame(d)))$mean, simulation)
      
      # L1 + L2 
      result[[8]] = 
        extract_bias(marginalEffects(cva.glmnet(model.matrix(~.^2, data = data.frame(data[,-1])), data[,1], alpha = 0.2), data = data[,-1], alpha = 0.2, formula = function(d) model.matrix(~.^2, data = data.frame(d)))$mean, simulation)
      
      return(result)
    })
  parallel::stopCluster(cl)
  return(result_list)
}



sim = function(Sigma) {
  
  
  return(
    simulate(r = Sigma ,
             effs = effs_true, 
             inter = inter_true, 
             n = 300)) 
}

results = get_result(sim)
saveRDS(results, "results/interactions_random_small.RDS")

sim = function(Sigma) {
  
  
  return(
    simulate(r = Sigma ,
             effs = effs_true, 
             inter = inter_true, 
             n = 500)) 
}

results = get_result(sim)
saveRDS(results, "results/interactions_random_middle.RDS")

sim = function(Sigma) {
  
  
  return(
    simulate(r = Sigma ,
             effs = effs_true, 
             inter = inter_true, 
             n = 5000)) 
}

results = get_result(sim)
saveRDS(results, "results/interactions_random_large.RDS")
