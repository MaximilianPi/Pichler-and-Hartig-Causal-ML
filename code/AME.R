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


print.marginalEffects = function(x, ...) {
  print(x$mean)
}
