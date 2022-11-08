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
  
  if(!is.null(inter)) {
    interInd = matrix(1:(length(inter)*2), ncol = 2L, byrow = TRUE)
    Y = X%*%effs + rowSums(sapply(1:nrow(interInd), function(i) X[,interInd[i,1]]*X[,interInd[i,2]]*inter[i] )) + rnorm(n, sd = 0.3)
  } else {
    Y = X%*%effs + rnorm(n, sd = 0.3)
  }
  
  colnames(X) = paste0("X", 1:ncol(X))
  data = cbind(Y, X)
  colnames(data)[1] = "Y" 
  return(data)
}