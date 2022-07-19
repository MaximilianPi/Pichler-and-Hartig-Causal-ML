library(ranger)
library(xgboost)
library(torch)
library(iml)
var_names = c("Y", "A", "B", "C","D", "E")

get_importance = function(model, predict_func, data) {
  predictor <- Predictor$new(model, data = data.frame(data[,-1]), y = data[,1], predict.function = predict_func)
  imp <- FeatureImp$new(predictor, loss = "mae", compare = "ratio")
  return(imp$results[order(imp$results$feature),])
}

predict_rf = function(model, newdata) predict(model, data=newdata)$predictions
predict_xg = function(model, newdata) as.vector(predict(model, as.matrix(newdata)))
predict_torch = function(model, newdata) as.numeric(model( torch_tensor(as.matrix(newdata))))


result_list = vector("list", 20L)


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


simulate_confounder = function(n = 1000) {
  B = rnorm(n) # confounder
  A = B+ rnorm(n, sd = 0.3)
  C = rnorm(n)
  D = rnorm(n)
  E = rnorm(n)
  
  Y = 1*A + 1*B + rnorm(n, sd = 0.3)
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
  
  Y = 0.5*A + 1*B + rnorm(n, sd = 0.3)
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
  Y = A + rnorm(n, sd = 0.3)
  B = Y + A + rnorm(n, sd = 0.3)
  data = cbind(Y, A, B,C, D, E)
  colnames(data) = var_names
  return(data)
}


simulate_collinearity = function(r=0.9, n = 1000) {
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
  
  Y = A*1.0 + 2*E+ rnorm(n, sd = 0.3)
  data = cbind(Y, A, B,C, D, E)
  colnames(data) = var_names
  return(scale(data))
}


simulate_effects = function(n = 1000) {
  B = rnorm(n) # confounder
  A = rnorm(n)
  C = rnorm(n)
  D = rnorm(n)
  E = rnorm(n)
  
  Y = 1*A + 0.5*B + rnorm(n, sd = 0.3)
  data = cbind(Y, A, B,C, D, E)
  colnames(data) = var_names
  return(scale(data))
}


get_result = function(sim ) {
  samples = 21
  result_list = vector("list", samples)
  
  cl = parallel::makeCluster(7L)
  parallel::clusterExport(cl, list("sim","simulate_collinearity","simulate_collidier","simulate_confounder", "simulate_confounder_equal",
                                   "get_importance","train_torch", "var_names", "predict_rf", "predict_torch", "predict_xg"), envir = environment())
  parallel::clusterEvalQ(cl, {library(ranger);library(xgboost);library(torch);library(iml)})
  result_list = 
    parallel::parLapply(cl, 1:samples, function(i) {
      #library(torch)
      data = sim()
      result = matrix(NA, nrow = 5, ncol = 10L)
      result[,1] = coef(lm(Y ~ A+B+C+D+E, data = data.frame(data)))[-1]
      rf = ranger(Y ~ A+B+C+D+E, data = data.frame(data), importance = "impurity_corrected")
      result[,2] = (ranger:::importance.ranger(rf))
      rf = ranger(Y ~ A+B+C+D+E, data = data.frame(data), importance = "permutation")
      result[,3] = ranger:::importance.ranger(rf)
      result[,4] = get_importance(rf, predict_rf, data)[,3]
      brt = xgboost::xgboost(data=xgboost::xgb.DMatrix(data[,-1], label = (data[,1, drop=FALSE])), 
                             nrounds = 200L, objective="reg:linear")
      imp_brt = xgboost::xgb.importance(model = brt)
      result[,5] = imp_brt$Gain[order(imp_brt$Feature)]
      result[,6] = get_importance(brt, predict_xg, data)[,3]
      
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
      result[,7] = get_importance(nn, predict_torch, data)[,3]
      
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
      
      result[,8] = get_importance(nn, predict_torch, data)[,3]
      
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
      result[,9] = get_importance(nn, predict_torch, data)[,3]
      
      
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
      result[,10] = get_importance(nn, predict_torch, data)[,3]
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

saveRDS(list(collidier=collidier,  collinearity=collinearity, confounder=confounder, confounder_equal = confounder_equal, effects=effects), "results.RDS")

results = readRDS("results.RDS")
res = results$collinearity
cols = RColorBrewer::brewer.pal(11, "Set3")


res = abind::abind(res, along = -1L)
res2= apply(res, 2:3, function(i) sd(i, na.rm=TRUE))

cor(abs(res2), method = "spearman")
res3 = (abs(res2 - do.call(rbind, lapply(1:5, function(i) res2[5,]))))
res4 = (res3 / do.call(rbind, lapply(1:5, function(i) apply(res3, 2, max))))[-5,]
plot(NULL, NULL, xaxt="n", xlim = c(0.9, 4), ylim = c(0, 1.))
for(i in 1:10) points(y = res4[,i], x =(1:4)+seq(-0.15, 0.15, length.out = 9)[i], col=cols[i], pch = 15)
axis(1, at = 1:4, labels = LETTERS[1:4])
legend("topright", legend = c("LM","RF_gini", "RF_perm", "RF_global", "BRT_imp", "BRT_global","NN", "NN_l2", "NN_l1", "NN_drop"), pch = 15, col= cols)


cor(res4, method = "spearman")
