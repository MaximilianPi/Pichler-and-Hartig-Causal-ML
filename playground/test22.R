get_importance2 = function(model, predict_func, data) {
  predictor <- Predictor$new(model, data = data.frame(data[,-1]), y = data[,1], predict.function = predict_func)
  imp <- FeatureEffects$new(predictor, method = "ale")
  AL_IMP = sapply(imp$results, function(i) sd(i$.value[i$.value < quantile(i$.value, 0.975) & i$.value > quantile(i$.value, 0.025)]))
  return((AL_IMP-min(AL_IMP))/max(AL_IMP))
}


get_importance = function(model, predict_func, data) {
  predictor <- Predictor$new(model, data = data.frame(data[,-1]), y = data[,1], predict.function = predict_func)
  imp <- FeatureImp$new(predictor, loss = "mae", compare = "ratio")
  imp2 = imp$results$importance
  imp2 = imp2[order(imp$results$feature)]
  return((imp2-min(imp2))/max(imp2))
  #return(imp)
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




r = 0.96
AB = mvtnorm::rmvnorm(1000, sigma = matrix(c(1., r, r, 1), 2, 2))
A = AB[,1]
B = AB[,2]
C = rnorm(1000)
D = rnorm(1000)
E = rnorm(1000)

Y = A  + 2*(E) + rnorm(1000, sd = 0.3)

data = cbind(Y, A, B,C, D, E)
colnames(data) = var_names
  
  brt = xgboost::xgboost(data=xgboost::xgb.DMatrix(data[,-1], label = (data[,1, drop=FALSE])), 
                         nrounds = 200L, objective="reg:linear")
get_importance2(brt, predict_xg, data)
get_importance(brt, predict_xg, data)

m = lm(Y~A+B+C+D+(E), data = data.frame(data))
get_importance(m, predict, data)
get_importance2(m, predict, data)


