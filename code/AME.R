AME = function(data, predict_f, model, epsilon = 0.1, obs_level = FALSE,interactions=TRUE,max_indices = NULL, ...) {
  
  x0 = data
  if(is.null(max_indices)) n = ncol(x0)
  else n = max_indices
  f = function(x0) predict_f(model, x0, ...)
  h = epsilon*apply(data, 2, sd)
  H = array(NA, c(nrow(x0), n, n))
  hh = diag(h, ncol(x0))
  f_x0 = f(x0)
  N = nrow(x0)
  for (i in 1:(n - 1)) {
    hi <- hh[, i]
    hi = matrix(hi, N, ncol(x0), byrow = TRUE )
    H[,i, i] =  (f(x0 + hi) - f_x0 )/h[i]
    if(interactions) {
      for (j in (i + 1):n) {
        hj = hh[, j]
        hj = matrix(hj, N, ncol(x0), byrow = TRUE )
        H[,i, j] = (f(x0 + hi + hj) - f(x0 + hi - hj) - f(x0 - hi + hj) + f(x0 - hi - hj))/(4 * h[i]^2)
        H[,j, i] = H[,i, j]
      }
    }
  }
  hi = hh[, n]
  hi = matrix(hi, N, ncol(x0), byrow = TRUE )
  H[, n, n] <-  ( f(x0 + hi) - f_x0 )/h[n]
  effs = apply(H, 2:3, mean)
  abs = apply(H, 2:3, function(d) mean(abs(d)))
  sds = apply(H, 2:3, sd)
  if(!obs_level) return(list(effs = effs, abs = abs, sds = sds))
  else return(H)
}


marginalEffects = function(object, ...) UseMethod("marginalEffects")

marginalEffects.citodnn = function(object, interactions=TRUE, epsilon = 0.1, device = "cpu", max_indices = NULL, data = NULL, ...) {
  if(is.null(data)) {
    data = object$data$data
    resp = object$data$Y
  } 
  Y_name = as.character( object$call$formula[[2]] )
  data = data[,-which( colnames(data) %in% Y_name)]
  var_names = c(Y_name, colnames(data))
  
  result = AME(
    data = data, 
    predict_f = function(model, newdata) {
      df = data.frame(Y_name = 0, newdata)
      colnames(df) = var_names
      return(predict(model, df, device = device))
    }, 
    model = object, obs_level = TRUE, 
    interactions=interactions, 
    epsilon = epsilon,
    max_indices = max_indices
  )
  out = list()
  out$result = result
  out$mean = apply(result, 2:3, mean)
  colnames(out$mean) = colnames(data)[1:ncol(out$mean)]
  rownames(out$mean) = colnames(data)[1:ncol(out$mean)]
  out$abs = apply(result, 2:3, function(d) mean(abs(d)))
  out$sd = apply(result, 2:3, function(d) sd(d))
  class(out) = "marginalEffects"
  return(out)
}

marginalEffects.xgb.Booster = function(object, data, interactions=TRUE, epsilon = 0.1, max_indices = NULL) {
  predict_xg = function(model, newdata) as.vector(predict(model, as.matrix(newdata)))
  result = AME(
    data = data, 
    predict_f = predict_xg, 
    model = object, 
    obs_level = TRUE, 
    interactions=interactions,
    epsilon = epsilon,
    max_indices = max_indices
  )
  out = list()
  out$result = result
  out$mean = apply(result, 2:3, mean)
  colnames(out$mean) = colnames(data)[1:ncol(out$mean)]
  rownames(out$mean) = colnames(data)[1:ncol(out$mean)]
  out$abs = apply(result, 2:3, function(d) mean(abs(d)))
  out$sd = apply(result, 2:3, function(d) sd(d))
  class(out) = "marginalEffects"
  return(out)
}


marginalEffects.ranger= function(object, data, interactions=TRUE, epsilon = 0.1, max_indices=NULL,...) {
  vars = object$forest$independent.variable.names
  data2 = data
  data2 = data2[,colnames(data2) %in% vars]
  
  
  predict_rf = function(model, newdata) predict(model, data=newdata)$predictions
  result = AME(
    data = data2, 
    predict_f = predict_rf, 
    model = object, 
    obs_level = TRUE, interactions=interactions,
    epsilon = epsilon,
    max_indices = max_indices
  )
  out = list()
  out$result = result
  out$mean = apply(result, 2:3, mean)
  colnames(out$mean) = colnames(data2)[1:ncol(out$mean)]
  rownames(out$mean) = colnames(data2)[1:ncol(out$mean)]
  out$abs = apply(result, 2:3, function(d) mean(abs(d)))
  out$sd = apply(result, 2:3, function(d) sd(d))
  class(out) = "marginalEffects"
  return(out)
}



marginalEffects.lm= function(object, data = NULL, interactions=TRUE, epsilon = 0.1, max_indices = NULL, ...) {
  if(is.null(data)) data = object$model
  Y_name = as.character(object$terms[[2]])
  data = data[,-which( colnames(data) %in% Y_name)]
  predict_lm = function(model, newdata) {
    return(predict(model, data.frame(newdata)))
  }
  result = AME(
    data = data, 
    predict_f = predict_lm, 
    model = object, 
    obs_level = TRUE, interactions=interactions, 
    epsilon = epsilon,
    max_indices = max_indices
  )
  out = list()
  out$result = result
  out$mean = apply(result, 2:3, mean)
  colnames(out$mean) = colnames(data)[1:ncol(out$mean)]
  rownames(out$mean) = colnames(data)[1:ncol(out$mean)]
  out$abs = apply(result, 2:3, function(d) mean(abs(d)))
  out$sd = apply(result, 2:3, function(d) sd(d))
  class(out) = "marginalEffects"
  return(out)
}

marginalEffects.cva.glmnet= function(object, data, formula = NULL, interactions=TRUE, epsilon = 0.1, max_indices=NULL, ...) {
  
  predict_glmnet = function(model, newdata, ...) {
    if(!is.null(formula)) newdata = formula(newdata)
    return(predict(model, newx = newdata, ...))
  }
  result = AME(
    data = data, 
    predict_f = predict_glmnet, 
    model = object, 
    obs_level = TRUE, 
    interactions=interactions,
    epsilon = epsilon,
    max_indices = max_indices,
    ...
  )
  out = list()
  out$result = result
  out$mean = apply(result, 2:3, mean)
  colnames(out$mean) = colnames(data)[1:ncol(out$mean)]
  rownames(out$mean) = colnames(data)[1:ncol(out$mean)]
  out$abs = apply(result, 2:3, function(d) mean(abs(d)))
  out$sd = apply(result, 2:3, function(d) sd(d))
  class(out) = "marginalEffects"
  return(out)
}


marginalEffects.glmnet= function(object, data, formula = NULL, interactions=TRUE, epsilon = 0.1, max_indices=NULL,...) {
  
  predict_glmnet = function(model, newdata, ...) {
    if(!is.null(formula)) newdata = formula(newdata)
    return(predict(model, newx = newdata, ...))
  }
  result = AME(
    data = data, 
    predict_f = predict_glmnet, 
    model = object, 
    obs_level = TRUE, 
    interactions=interactions,
    epsilon = epsilon,
    max_indices = max_indices,
    ...
  )
  out = list()
  out$result = result
  out$mean = apply(result, 2:3, mean)
  colnames(out$mean) = colnames(data)[1:ncol(out$mean)]
  rownames(out$mean) = colnames(data)[1:ncol(out$mean)]
  out$abs = apply(result, 2:3, function(d) mean(abs(d)))
  out$sd = apply(result, 2:3, function(d) sd(d))
  class(out) = "marginalEffects"
  return(out)
}



marginalEffects.gbm= function(object, data, interactions=TRUE, epsilon = 0.1, max_indices = NULL, ...) {
  predict_gbm = function(model, newdata) {
    return(predict(model, newdata = data.frame(newdata)))
  }
  result = AME(
    data = data, 
    predict_f = predict_gbm, 
    model = object, 
    obs_level = TRUE, interactions=interactions, 
    epsilon = epsilon,
    max_indices = max_indices
  )
  out = list()
  out$result = result
  out$mean = apply(result, 2:3, mean)
  colnames(out$mean) = colnames(data)[1:ncol(out$mean)]
  rownames(out$mean) = colnames(data)[1:ncol(out$mean)]
  out$abs = apply(result, 2:3, function(d) mean(abs(d)))
  out$sd = apply(result, 2:3, function(d) sd(d))
  class(out) = "marginalEffects"
  return(out)
}


#### Helper function for single tree fit.
get_model_tree = function(x, y, ...){
  control = tree.control(nobs = length(x), ...)
  model = tree(y~., data.frame(x = x, y = y), control = control)
  pred = predict(model, newdata = data.frame(x = x, y = y))
  return(list(model = model, pred = pred))
}

get_model_linear = function(x, y, ...){
  data = data.frame(x = x, y = y)
  models = lapply(paste0("y~", colnames(data.frame(x = x))), function(f) lm(as.formula(f), data = data))
  model = models[[which.max(abs(sapply(models, coef)[2,]))]]
  pred = predict(model, newdata = data.frame(x = x, y = y))
  return(list(model = model, pred = pred))
}

#### Boost function.
get_boosting_model = function(x, y, n_trees, bootstrap = NULL, colsample = NULL, eta = 1., booster = "tree", ...){
  pred = NULL
  m_list = list()
  if(booster == "tree") get_model = get_model_tree
  else get_model = get_model_linear
  for(i in 1:n_trees){
    if(i == 1){
      m = get_model(x, y, ...)
      pred = m$pred
    }else{
      if(!is.null(bootstrap)) indices = sample.int(length(y), bootstrap*length(y), replace = TRUE)
      else indices = 1:length(y)
      if(!is.null(colsample)) indices_cols = sample.int(ncol(x), colsample*ncol(x), replace = FALSE)
      else indices_cols = 1:ncol(x)
      y_res = y[indices] - pred[indices]
      m = get_model(x[indices,indices_cols,drop=FALSE], y_res, ...)
      pred = pred + eta*predict(m$model, newdata = data.frame(x = x))
    }
    m_list[[i]] = m$model
  }
  model_list = list()
  model_list$model = m_list 
  model_list$eta = eta
  class(model_list) = "naiveBRT"
  return(model_list)
}

marginalEffects.naiveBRT= function(object, data, interactions=TRUE, epsilon = 0.1, max_indices = NULL, ...) {
  predict.naiveBRT = function(model, newdata) {
    N = model$N
    if(is.null(N)) N = length(model$model)
    eta = model$eta
    
    if(N != 1 ) return(rowSums(matrix(c(1, rep(eta, N-1)), nrow(newdata), N, byrow = TRUE) * sapply(1:N, function(k) predict(model$model[[k]], newdata = data.frame(x = newdata)))))
    else return(predict(model$model[[1]], newdata = data.frame(x = newdata)))
  }
  
  predict_naiveBRT = function(model, newdata) {
    return(predict.naiveBRT(model, newdata = data.frame(newdata)))
  }
  result = AME(
    data = data, 
    predict_f = predict_naiveBRT, 
    model = object, 
    obs_level = TRUE, interactions=interactions, 
    epsilon = epsilon,
    max_indices = max_indices
  )
  out = list()
  out$result = result
  out$mean = apply(result, 2:3, mean)
  colnames(out$mean) = colnames(data)[1:ncol(out$mean)]
  rownames(out$mean) = colnames(data)[1:ncol(out$mean)]
  out$abs = apply(result, 2:3, function(d) mean(abs(d)))
  out$sd = apply(result, 2:3, function(d) sd(d))
  class(out) = "marginalEffects"
  return(out)
}


print.marginalEffects = function(x, ...) {
  print(x$mean)
}
