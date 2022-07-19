library(ranger)
library(xgboost)
library(torch)
library(iml)
var_names = c("Y", "A", "B", "C","D", "E")
torch::torch_set_num_threads(3L)

predict_rf = function(model, newdata) predict(model, data=newdata)$predictions
predict_xg = function(model, newdata) as.vector(predict(model, as.matrix(newdata)))
predict_torch = function(model, newdata) as.numeric(model( torch_tensor(as.matrix(newdata))))



get_importance_ALE = function(model, predict_func, data) {
  predictor <- Predictor$new(model, data = data.frame(data[,-1]), y = data[,1], predict.function = predict_func)
  imp <- FeatureEffects$new(predictor, method = "ale")
  AL_IMP = sapply(imp$results, function(i) sd(i$.value[i$.value < quantile(i$.value, 0.975) & i$.value > quantile(i$.value, 0.025)]))
  return((AL_IMP-min(AL_IMP))/max(AL_IMP))
}


get_importance_PERM = function(model, predict_func, data) {
  predictor <- Predictor$new(model, data = data.frame(data[,-1]), y = data[,1], predict.function = predict_func)
  imp <- FeatureImp$new(predictor, loss = "mae", compare = "ratio")
  imp2 = imp$results$importance
  imp2 = imp2[order(imp$results$feature)]
  return((imp2-min(imp2))/max(imp2))
}


train_torch = function(nn, data, l1 = FALSE, l2 = FALSE, epochs = 200){
  opt = optim_adam( nn$parameters , lr = 0.01)
  nn$train()
  for(i in 1:epochs){
    ind = sample.int(nrow(data), 20L)
    X_b = torch_tensor(data[ind,-1])
    Y_b = torch_tensor(data[ind,1,drop=FALSE])
    opt$zero_grad()
    pred = nn(X_b)
    loss = nnf_mse_loss(pred, Y_b)
    if(l1) {
      reg_loss = torch_zeros(1)
      for(p in nn$parameters){
        if(sum(dim(p)) > 1) reg_loss = reg_loss + torch_norm(p,p = 1)
      }
      loss = loss + reg_loss*0.03
    }
    if(l2) {
      reg_loss = torch_zeros(1)
      for(p in nn$parameters){
        reg_loss = reg_loss + torch_norm(p,p = 2)
      }
      loss = loss + reg_loss*0.03
    }
    loss$backward()
    opt$step()
  }
  nn$eval()
}


simulate_confounder_equal = function(n = 1000) {
  B = rnorm(n) # confounder
  A = B+ rnorm(n, sd = 0.3)
  C = rnorm(n)
  D = rnorm(n)
  E = rnorm(n)
  
  Y = A + B + E + rnorm(n, sd = 0.3)
  data = cbind(Y, A, B,C, D, E)
  colnames(data) = var_names
  return(scale(data))
}

simulate_confounder = function(n = 1000) {
  B = rnorm(n) # confounder
  A = B+ rnorm(n, sd = 0.3)
  C = rnorm(n)
  D = rnorm(n)
  E = rnorm(n)
  
  Y = 0.5*A - 1*B +E+ rnorm(n, sd = 0.3)
  data = cbind(Y, A, B,C, D, E)
  colnames(data) = var_names
  return(scale(data))
}

simulate_collidier = function(n = 1000) {
  # B collidier
  C = rnorm(n)
  D = rnorm(n)
  E = rnorm(n)
  A = rnorm(n)
  Y = -A + rnorm(n, sd = 0.3)
  B = Y + A +E+ rnorm(n, sd = 0.3)
  data = cbind(Y, A, B,C, D, E)
  colnames(data) = var_names
  return(scale(data))
}


simulate_collinearity = function(r=0.95, n = 1000) {
  ABCDE = MASS::mvrnorm(n, mu=rep(0, 5), Sigma=matrix(c(1, r, 0,0,0,
                                                        r, 1, 0,0,0,
                                                        0, 0, 1,0,0,
                                                        0, 0, 0,1,0,
                                                        0, 0, 0, 0,1), nrow=5, byrow = TRUE), empirical=TRUE)
  A = ABCDE[,1] 
  B = ABCDE[,2]
  C = ABCDE[,3]
  D = ABCDE[,4]
  E = ABCDE[,5]
  
  Y = A*1.0 + 1*E+ rnorm(n, sd = 0.3)
  data = cbind(Y, A, B,C, D, E)
  colnames(data) = var_names
  return(scale(data))
}


get_result = function(sim ) {
  samples = 100
  result_list = vector("list", samples)
  
  cl = parallel::makeCluster(10L)
  parallel::clusterExport(cl, list("sim","simulate_collinearity","simulate_collidier","simulate_confounder", "simulate_confounder_equal",
                                   "get_importance2","train_torch", "var_names", "predict_rf", "predict_torch", "predict_xg"), envir = environment())
  parallel::clusterEvalQ(cl, {library(ranger);library(xgboost);library(torch);library(iml)})
  result_list = 
    parallel::parLapply(cl, 1:samples, function(i) {
      #library(torch)
      data = sim()
      result = matrix(NA, nrow = 5, ncol = 7L)
      result[,1] = get_importance_ALE(lm(Y ~ A+B+C+D+E, data = data.frame(data)), predict,data = data)
      rf = ranger(Y ~ A+B+C+D+E, data = data.frame(data), importance = "impurity")

      result[,2] = get_importance_ALE(rf, predict_rf, data)
      brt = xgboost::xgboost(data=xgboost::xgb.DMatrix(data[,-1], label = (data[,1, drop=FALSE])), 
                             nrounds = 200L, objective="reg:linear",params = list(nthread = 3))
      result[,3] = get_importance_ALE(brt, predict_xg, data)
      
      # NO
      nn = nn_sequential(
        nn_linear(ncol(data[,-1]), out_features = 20L),
        nn_relu(),
        nn_linear(20L, out_features = 20L),
        nn_relu(),
        nn_linear(20L, out_features = 20L),
        nn_relu(),
        nn_linear(20L, out_features = 1L)
      )
      train_torch(nn, data)
      result[,4] = get_importance_ALE(nn, predict_torch, data)
      
      # L1
      nn = nn_sequential(
        nn_linear(ncol(data[,-1]), out_features = 20L),
        nn_relu(),
        nn_linear(20L, out_features = 20L),
        nn_relu(),
        nn_linear(20L, out_features = 20L),
        nn_relu(),
        nn_linear(20L, out_features = 1L)
      )
      
      train_torch(nn, data, l1 = TRUE)
      
      result[,5] = get_importance_ALE(nn, predict_torch, data)
      
      # L2
      nn = nn_sequential(
        nn_linear(ncol(data[,-1]), out_features = 20L),
        nn_relu(),
        nn_linear(20L, out_features = 20L),
        nn_relu(),
        nn_linear(20L, out_features = 20L),
        nn_relu(),
        nn_linear(20L, out_features = 1L)
      )
      train_torch(nn, data, l2 = TRUE)
      result[,6] = get_importance_ALE(nn, predict_torch, data)
      
      
      # Dropout
      nn = nn_sequential(
        nn_linear(ncol(data[,-1]), out_features = 20L),
        nn_dropout(0.3),
        nn_relu(),
        nn_linear(20L, out_features = 20L),
        nn_dropout(0.3),
        nn_relu(),
        nn_linear(20L, out_features = 20L),
        nn_dropout(0.3),
        nn_relu(),
        nn_linear(20L, out_features = 1L)
      )
      train_torch(nn, data)
      result[,7] = get_importance_ALE(nn, predict_torch, data)
      #result_list[[i]] = result
      return(result)
    })
  parallel::stopCluster(cl)
  return(result_list)
}

collidier = get_result(simulate_collidier)
confounder = get_result(simulate_confounder)
confounder_equal = get_result(simulate_confounder_equal)
collinearity = get_result(simulate_collinearity)
effects = get_result(simulate_effects)
results = list(collidier=collidier,  collinearity=collinearity, confounder=confounder, confounder_equal = confounder_equal, effects=effects)
saveRDS(results, "results_ALE.RDS")

