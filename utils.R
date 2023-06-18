
addA = function(col, alpha = 0.25) apply(sapply(col, grDevices::col2rgb)/255, 2, function(x) grDevices::rgb(x[1], x[2], x[3], alpha=alpha))

eqarrowPlot <- function(graph, layout, edge.lty=rep(1, ecount(graph)),edge.width=rep(1, ecount(graph)),
                        edge.arrow.size=rep(1, ecount(graph)), cols = c( "pink","pink", "skyblue"), edge.arrow.mode = NULL, edge.colors = NULL,
                        rangeX = c(0, 1), rangeY = c(0, 2),vertex.label.cex = 2.1, ...) {
  vertex.size = 40
  plot(graph, edge.lty=0, edge.arrow.size=0, layout=layout,
       vertex.shape="none",  vertex.size=vertex.size, vertex.color = cols, rescale=FALSE, xlim = rangeX, ylim = rangeY, vertex.label.cex = vertex.label.cex)
  if(is.null(edge.arrow.mode)) edge.arrow.mode = rep(">", (ecount(graph)))
  if(is.null(edge.colors)) edge.colors = rep(NULL, ecount(graph))
  for (e in seq_len(ecount(graph))) {
    graph2 <- delete.edges(graph, E(graph)[(1:ecount(graph))[-e]])
    plot(graph2, edge.lty=edge.lty[e], edge.arrow.size=edge.arrow.size[e], layout=layout,edge.color = edge.colors[e],
         vertex.label=NA, add=TRUE, vertex.color = cols, edge.width=edge.width[e], vertex.size=vertex.size,edge.arrow.mode = edge.arrow.mode[e], rescale=FALSE, 
         xlim = rangeX, ylim = rangeY, vertex.label.cex = vertex.label.cex)
  }
  plot(graph, edge.lty=0, 
       edge.arrow.size=0, 
       layout=layout, 
       add=TRUE,
       vertex.size=vertex.size, 
       vertex.color = cols,
       vertex.label.cex = vertex.label.cex, xlim = rangeX, ylim = rangeY,
       edge.label.color = "black", rescale=FALSE,...)
  invisible(NULL)
}


plot_scenarios = function(cex_fac = 1.3, layout = layout, x00 = 1.4) {
  g1 <- graph(c("x\U2081", "y", "x\U2082", "y", "x\U2083", "y"),  
              #c(bquote(X[1]), bquote(y), bquote(X[2]), bquote(y), bquote(X[3]), bquote(y)),
              directed=TRUE ) 
  eqarrowPlot(g1, layout, 
              #cols = c( "skyblue","#B0A8B9", "#B0A8B9", "#B0A8B9"),
              cols = c(addA(rep("#87CEEB", 1), 1.0), "#B0A8B9", addA(rep("#87CEEB", 1), 1.0), addA(rep("#87CEEB", 2), 1.0)),
              edge.arrow.size=c(1, 0.5,1.0), 
              edge.width=c(1, 0.5,1.0)*cex_fac,
              #edge.label = c("1.0\n\n","0.5\n","1.0"), 
              edge.label = c("1.0\n\n","0.5\n","1.0"), 
              edge.label.cex = 1.4,
              edge.colors = c(rep("grey", 2), "grey"))
  
  text(letters[1], cex = 1.9, x = -1.6, y = 1.5, xpd = NA, font = 2)
  text("Simulation", x = 0.3, y = 2.3, xpd = NA, cex = 1.4, pos = 3)
  
  
  segments(x0 = x00, x1 = x00, y0 = -0.5, y1 = 2.5, xpd = NA)
  
  # Collinearity
  g1 <- graph(c("x\U2081", "y", "x\U2082","y", "x\U2083", "y", "x\U2081", "x\U2082"),  
              directed=TRUE ) 
  eqarrowPlot(g1, layout, 
              #cols = c( "skyblue","#B0A8B9", "#B0A8B9", "#B0A8B9"),
              cols = c(addA(rep("#87CEEB", 1), 1.0), "#B0A8B9", addA(rep("#87CEEB", 1), 1.0), addA(rep("#87CEEB", 2), 1.0)),
              edge.arrow.size=c(1.0, 0.0,1.0,0.95)*cex_fac, 
              edge.width=c(1.0, 0.5,1.0,0.95)*cex_fac,
              edge.label = c("1.0\n\n","\n","1.0", "\n0.90"), 
              edge.label.cex = 1.4, 
              edge.arrow.mode = c(rep(">", 3), "-"), 
              edge.colors = c(rep("grey", 1),"white","grey", "#ffab02"))
  text(letters[3], cex = 1.9, x = -1.6, y = 1.5, xpd = NA, font = 2)
  text("Simulation", x = 0.3, y = 2.3, xpd = NA, cex = 1.4, pos = 3)
  
  segments(x0 = x00, x1 = x00, y0 = -0.5, y1 = 3.5, xpd = NA)
  # Confounder
  # g1 <- graph(c("x\U2081", "y", "x\U2082","y", "x\U2083", "y", "x\U2081", "x\U2082"),  
  #             directed=TRUE ) 
  # eqarrowPlot(g1, layout, 
  #             #cols = c( "skyblue","#B0A8B9", "#B0A8B9", "#B0A8B9"),
  #             cols = c(addA(rep("#87CEEB", 1), 1.0), "#B0A8B9", addA(rep("#87CEEB", 1), 1.0), addA(rep("#87CEEB", 2), 1.0)),
  #             edge.arrow.size=c(1.0, 0.5,1.0,0.95)*cex_fac, 
  #             edge.width=c(1.0, 0.5,1.0,0.95)*cex_fac,
  #             edge.label = c("-1.0\n\n","0.5\n","1.0", "\n0.90"), 
  #             edge.label.cex = 1.4, 
  #             edge.arrow.mode = c(rep(">", 3), "-"), 
  #             edge.colors = c(rep("grey", 2),"grey", "#ffab02"))
  # text(letters[2], cex = 1.9, x = -1.6, y = 1.5, xpd = NA, font = 2)
  # text("Simulation", x = 0.3, y = 2.3, xpd = NA, cex = 1.4, pos = 3)
  # 
  # segments(x0 = x00, x1 = x00, y0 = -0.5, y1 = 3.5, xpd = NA)
  
}



create_gapped_stacked = function(tmp, 
                                 upper = 0.75, 
                                 dd = 0.03, 
                                 to = c(0, 0.75), 
                                 cols = c("#96c6ed","#e0acd5" ),
                                 to2 = c(100, 200),
                                 labels1 = c("0.00", "0.25", "0.50", "0.75"),
                                 labels2 = seq(100, 200, length.out = 3),
                                 axes = TRUE,
                                 gapped = TRUE,
                                 d_between = 0.1
) {
  scale_upper = 1.025
  plot(NULL, NULL, xlim = c(0, 1), ylim = c(0, 1), yaxt = "n", xaxt = "n", xaxs = "i", yaxs = "i", xlab = "", ylab = "")
  biases = abs(tmp %>% filter(name == "bias") %>% pull(value))
  vars = abs(tmp %>% filter(name == "var") %>% pull(value))
  
  inters = seq(0, 1, length.out = length(biases)+2)
  inters = inters[-c(1, length(inters))]
  
  for(i in 1:length(biases)) {
    rect(inters[i]-d_between, 0, inters[i]+d_between, ytop = scales::rescale(biases[i], to = c(0, upper), from = to), col = cols[1])
    upperB = scales::rescale(biases[i], to = c(0, upper), from = to) + scales::rescale(vars[i], to = c(0, upper), from = to)
    if(upperB > upper) {
      rect(inters[i]-d_between, scales::rescale(biases[i], to = c(0, upper), from = to) , inters[i]+d_between, 
           ytop = upper*scale_upper+0.001, col = cols[2]) 
      rect(inters[i]-d_between, upper*scale_upper+0.001, inters[i]+d_between, 
           ytop = scales::rescale(biases[i]+vars[i], to = c(upper+dd*2, 0.97), from = to2), col = cols[2]) 
    } else {
      rect(inters[i]-d_between, scales::rescale(biases[i], to = c(0, upper), from = to) , inters[i]+d_between, 
           ytop = upperB,  col = cols[2]) 
    }
  }
  if(gapped) {
    rect(-0.02, upper*scale_upper, 1.02, upper+dd, col = "white", xpd = NA, border = NA)
    segments(-0.02,  upper*scale_upper-0.01, 0.0,    upper*scale_upper, xpd = NA)
    segments(0.02,   upper*scale_upper+0.01, 0.0,    upper*scale_upper, xpd = NA)
    segments(-0.02,  upper-0.01+dd, 0.0, upper+dd, xpd = NA)
    segments(0.02,   upper+0.01+dd, 0.0, upper+dd, xpd = NA)
    segments(1-0.02, upper*scale_upper-0.01, 1.0,    upper*scale_upper, xpd = NA)
    segments(1+0.02, upper*scale_upper+0.01, 1.0,    upper*scale_upper, xpd = NA)
    segments(1-0.02, upper-0.01+dd, 1.0, upper+dd, xpd = NA)
    segments(1+0.02, upper+0.01+dd, 1.0, upper+dd, xpd = NA)
  }
  if(axes) {
    axis(2, at = seq(0, upper-0.004, length.out = length(labels1)), labels = labels1, las = 1)
    if(gapped) axis(2, at = seq(upper+dd*2, 0.97, length.out = length(labels2)), labels = labels2, las = 1)
  }
  
}



create_stacked_grouped = function(tmp, 
                                 upper = 0.75, 
                                 dd = 0.03, 
                                 to = c(0, 0.75), 
                                 cols = c("#96c6ed","#e0acd5" ),
                                 labels1 = c("0.00", "0.25", "0.50", "0.75"),
                                 axes = TRUE,
                                 d_between = 0.1
) {
  scale_upper = 1.025
  plot(NULL, NULL, xlim = c(0, 1), ylim = c(0, 1), yaxt = "n", xaxt = "n", xaxs = "i", yaxs = "i", xlab = "", ylab = "")
  biases = abs(tmp %>% filter(name == "bias") %>% pull(value))
  vars = abs(tmp %>% filter(name == "var") %>% pull(value))
  
  inters = seq(0, 1, length.out = length(biases)+2)
  inters = inters[-c(1, length(inters))]
  
  for(i in 1:length(biases)) {
    rect(inters[i]-d_between, 0, inters[i]+d_between, ytop = scales::rescale(biases[i], to = c(0, upper), from = to), col = cols[1])
    upperB = scales::rescale(biases[i], to = c(0, upper), from = to) + scales::rescale(vars[i], to = c(0, upper), from = to)
    if(upperB > upper) {
      rect(inters[i]-d_between, scales::rescale(biases[i], to = c(0, upper), from = to) , inters[i]+d_between, 
           ytop = upper*scale_upper+0.001, col = cols[2]) 
      rect(inters[i]-d_between, upper*scale_upper+0.001, inters[i]+d_between, 
           ytop = scales::rescale(biases[i]+vars[i], to = c(upper+dd*2, 0.97), from = to2), col = cols[2]) 
    } else {
      rect(inters[i]-d_between, scales::rescale(biases[i], to = c(0, upper), from = to) , inters[i]+d_between, 
           ytop = upperB,  col = cols[2]) 
    }
  }
 
  if(axes) {
    axis(2, at = seq(0, upper-0.004, length.out = length(labels1)), labels = labels1, las = 1)
    if(gapped) axis(2, at = seq(upper+dd*2, 0.97, length.out = length(labels2)), labels = labels2, las = 1)
  }
}





get_coefficients = function(i, gam = TRUE, path, method) {
  
  out = list()
  
  ind = c(6, 5, 4, 2)[i]
  raw = readRDS(path)
  
  if(method == "NN") {
    raw = lapply(raw, function(r) {
      r = r %>% select(-dropout)
      return(r)
    })
  }
  
  parameter = raw[[1]][,1:ind]
  
  raw = lapply(raw, function(r) {
    r$mse_pred = r$pred_mse
    r$pred_mse = NULL
    return(r)
  } )
  
  raw = lapply(raw, function(r) {
    if(nrow(r[complete.cases(r),][abs(r[complete.cases(r),]$mse_pred) > 2000, ]) > 0) {
      r[complete.cases(r),][abs(r[complete.cases(r),]$mse_pred) > 2000, ]$mse_pred = NA
    }
    return(r)
  })
  
  parameter$bias_effect = apply(sapply(raw, function(r) 1-r$eff_1), 1, mean)**2
  parameter$var_effect = apply(sapply(raw, function(r) r$eff_1), 1, var)
  parameter$bias_zero = apply(sapply(raw, function(r) r$eff_2 ), 1, mean)**2
  parameter$var_zero = apply(sapply(raw, function(r) r$eff_2 ), 1, var)
  parameter$bias_pred = apply(sapply(raw, function(r) r$pred_bias ), 1, mean)**2
  parameter$var_pred = apply(sapply(raw, function(r) r$pred_var ), 1, var)
  parameter$mse_pred = apply(sapply(raw, function(r) r$mse_pred ), 1, mean)
  parameter$mse_eff = parameter$bias_effect + parameter$var_effect
  parameter$mse_zero = parameter$bias_zero + parameter$var_zero
  parameter = parameter[complete.cases(parameter),]
  tmp = parameter
  
  if(method != "NN" ) {tmp[,(1:ind)] = scale( tmp[,(1:ind)] )
  } else { tmp[, (2:ind)] = scale( tmp[, (2:ind)] ) }
  
  tmp[,-(1:ind)] = sapply(  tmp[,-(1:ind)], function(df) df - mean(df))
  
  out$lm_eff = lm(  mse_eff ~ 0 + .,data = cbind( tmp[,1:ind] ,tmp %>% select(mse_eff )))
  out$lm_pred = lm( mse_pred ~0+ ., data = cbind( tmp[,1:ind] ,tmp %>% select(mse_pred )))
  out$lm_zero = lm( mse_zero ~0+ ., data = cbind( tmp[,1:ind] ,tmp %>% select(mse_zero )))
  out$lm_bias_eff = lm(  bias_effect ~ 0 + .,data = cbind( tmp[,1:ind] ,tmp %>% select(bias_effect )))
  out$lm_var_eff = lm( var_effect ~0+ ., data = cbind( tmp[,1:ind] ,tmp %>% select(var_effect )))
  out$lm_bias_zero = lm(  bias_zero ~ 0 + .,data = cbind( tmp[,1:ind] ,tmp %>% select(bias_zero )))
  out$lm_var_zero = lm( var_zero ~0+ ., data = cbind( tmp[,1:ind] ,tmp %>% select(var_zero )))
  out$lm_bias_pred = lm(  bias_pred ~ 0 + .,data = cbind( tmp[,1:ind] ,tmp %>% select(bias_pred )))
  out$lm_var_pred = lm( var_pred ~0+ ., data = cbind( tmp[,1:ind] ,tmp %>% select(var_pred )))
  
  coefs_eff = summary( out$lm_eff )$coefficients
  coefs_pred = summary( out$lm_pred )$coefficients
  coefs_zero = summary( out$lm_zero )$coefficients
  
  coefs_bias_effect = summary( out$lm_bias_eff )$coefficients
  coefs_var_effect = summary( out$lm_var_eff )$coefficients
  coefs_bias_zero = summary( out$lm_bias_zero )$coefficients
  coefs_var_zero = summary( out$lm_var_zero )$coefficients
  coefs_bias_pred = summary( out$lm_bias_pred )$coefficients
  coefs_var_pred = summary( out$lm_var_pred )$coefficients
  
  if(method == "NN") {
    
    tmp$depth = tmp$depth + rnorm(nrow(tmp), 0, 0.001)
    cols = cbind( tmp[,1:ind] ,tmp %>% select(mse_eff )) %>% colnames()
    form1 = paste0("mse_eff ~ 0 +activations +" , paste0("s(", cols[-length(cols)][-1], ")", collapse="+"))
    cols = cbind( tmp[,1:ind] ,tmp %>% select(mse_zero )) %>% colnames()
    form2 = paste0("mse_zero ~ 0 +activations +" , paste0("s(", cols[-length(cols)][-1], ")", collapse="+"))
    cols = cbind( tmp[,1:ind] ,tmp %>% select(mse_pred )) %>% colnames()
    form3 = paste0("mse_pred ~ 0 +activations +" , paste0("s(", cols[-length(cols)][-1], ")", collapse="+"))
    
    cols = cbind( tmp[,1:ind] ,tmp %>% select(bias_effect )) %>% colnames()
    form4 = paste0("bias_effect ~ 0 +activations +" , paste0("s(", cols[-length(cols)][-1], ")", collapse="+"))
    cols = cbind( tmp[,1:ind] ,tmp %>% select(var_effect )) %>% colnames()
    form5 = paste0("var_effect ~ 0 +activations +" , paste0("s(", cols[-length(cols)][-1], ")", collapse="+"))
    
    cols = cbind( tmp[,1:ind] ,tmp %>% select(bias_zero )) %>% colnames()
    form6 = paste0("bias_zero ~ 0 +activations +" , paste0("s(", cols[-length(cols)][-1], ")", collapse="+"))
    cols = cbind( tmp[,1:ind] ,tmp %>% select(var_zero )) %>% colnames()
    form7 = paste0("var_zero ~ 0 +activations +" , paste0("s(", cols[-length(cols)][-1], ")", collapse="+"))
    
    cols = cbind( tmp[,1:ind] ,tmp %>% select(bias_pred )) %>% colnames()
    form8 = paste0("bias_pred ~ 0 +activations +" , paste0("s(", cols[-length(cols)][-1], ")", collapse="+"))
    cols = cbind( tmp[,1:ind] ,tmp %>% select(var_pred )) %>% colnames()
    form9 = paste0("var_pred ~ 0 +activations +" , paste0("s(", cols[-length(cols)][-1], ")", collapse="+"))
    
  } else {
    cols = cbind( tmp[,1:ind] ,tmp %>% select(mse_eff )) %>% colnames()
    form1 = paste0("mse_eff ~ 0  +" , paste0("s(", cols[-length(cols)], ")", collapse="+"))
    cols = cbind( tmp[,1:ind] ,tmp %>% select(mse_zero )) %>% colnames()
    form2 = paste0("mse_zero ~ 0  +" , paste0("s(", cols[-length(cols)], ")", collapse="+"))
    cols = cbind( tmp[,1:ind] ,tmp %>% select(mse_pred )) %>% colnames()
    form3 = paste0("mse_pred ~ 0  +" , paste0("s(", cols[-length(cols)], ")", collapse="+"))
    
    cols = cbind( tmp[,1:ind] ,tmp %>% select(bias_effect )) %>% colnames()
    form4 = paste0("bias_effect ~ 0  +" , paste0("s(", cols[-length(cols)], ")", collapse="+"))
    cols = cbind( tmp[,1:ind] ,tmp %>% select(var_effect )) %>% colnames()
    form5 = paste0("var_effect ~ 0  +" , paste0("s(", cols[-length(cols)], ")", collapse="+"))
    
    cols = cbind( tmp[,1:ind] ,tmp %>% select(bias_zero )) %>% colnames()
    form6 = paste0("bias_zero ~ 0  +" , paste0("s(", cols[-length(cols)], ")", collapse="+"))
    cols = cbind( tmp[,1:ind] ,tmp %>% select(var_zero )) %>% colnames()
    form7 = paste0("var_zero ~ 0 +" , paste0("s(", cols[-length(cols)], ")", collapse="+"))
    
    cols = cbind( tmp[,1:ind] ,tmp %>% select(bias_pred )) %>% colnames()
    form8 = paste0("bias_pred ~ 0  +" , paste0("s(", cols[-length(cols)], ")", collapse="+"))
    cols = cbind( tmp[,1:ind] ,tmp %>% select(var_pred )) %>% colnames()
    form9 = paste0("var_pred ~ 0 +" , paste0("s(", cols[-length(cols)], ")", collapse="+"))
  }
  
  if(gam) {
    out$gam_eff =         qgam(as.formula(form1), data = tmp, qu = 0.5)
    out$gam_zero =        qgam(as.formula(form2), data = tmp, qu = 0.5)
    out$gam_pred =        qgam(as.formula(form3), data = tmp, qu = 0.5)
    out$gam_bias_effect = qgam(as.formula(form4), data = tmp, qu = 0.5)
    out$gam_var_effect =  qgam(as.formula(form5), data = tmp, qu = 0.5)
    out$gam_bias_zero =   qgam(as.formula(form6), data = tmp, qu = 0.5)
    out$gam_var_zero =    qgam(as.formula(form7), data = tmp, qu = 0.5)
    out$gam_bias_pred =   qgam(as.formula(form8), data = tmp, qu = 0.5)
    out$gam_var_pred =    qgam(as.formula(form9), data = tmp, qu = 0.5)
    
    out$range_eff = c(
      min(sapply(plot(getViz(out$gam_eff))[[1]], function(obj) min(obj$data$fit$y)  )),
      max(sapply(plot(getViz(out$gam_eff))[[1]], function(obj) max(obj$data$fit$y)  ))
    )
    out$range_pred = c(
      min(sapply(plot(getViz(out$gam_pred))[[1]], function(obj) min(obj$data$fit$y)  )),
      max(sapply(plot(getViz(out$gam_pred))[[1]], function(obj) max(obj$data$fit$y)  ))
    )
    out$range_zero = c(
      min(sapply(plot(getViz(out$gam_zero))[[1]], function(obj) min(obj$data$fit$y)  )),
      max(sapply(plot(getViz(out$gam_zero))[[1]], function(obj) max(obj$data$fit$y)  ))
    )
    out$range_bias_effect = c(
      min(sapply(plot(getViz(out$gam_bias_effect))[[1]], function(obj) min(obj$data$fit$y)  )),
      max(sapply(plot(getViz(out$gam_bias_effect))[[1]], function(obj) max(obj$data$fit$y)  ))
    )
    out$range_var_effect = c(
      min(sapply(plot(getViz(out$gam_var_effect))[[1]], function(obj) min(obj$data$fit$y)  )),
      max(sapply(plot(getViz(out$gam_var_effect))[[1]], function(obj) max(obj$data$fit$y)  ))
    )
    out$range_bias_zero = c(
      min(sapply(plot(getViz(out$gam_bias_zero))[[1]], function(obj) min(obj$data$fit$y)  )),
      max(sapply(plot(getViz(out$gam_bias_zero))[[1]], function(obj) max(obj$data$fit$y)  ))
    )
    out$range_var_zero = c(
      min(sapply(plot(getViz(out$gam_var_zero))[[1]], function(obj) min(obj$data$fit$y)  )),
      max(sapply(plot(getViz(out$gam_var_zero))[[1]], function(obj) max(obj$data$fit$y)  ))
    )
    
    out$range_bias_pred = c(
      min(sapply(plot(getViz(out$gam_bias_pred))[[1]], function(obj) min(obj$data$fit$y)  )),
      max(sapply(plot(getViz(out$gam_bias_pred))[[1]], function(obj) max(obj$data$fit$y)  ))
    )
    out$range_var_pred = c(
      min(sapply(plot(getViz(out$gam_var_pred))[[1]], function(obj) min(obj$data$fit$y)  )),
      max(sapply(plot(getViz(out$gam_var_pred))[[1]], function(obj) max(obj$data$fit$y)  ))
    )
  }
  tmp = data.frame(model.matrix(~0+., tmp))
  
  
  if(method == "NN") {
    ind2 = ind+6
  } else {
    ind2 = ind
  }
  
  rf_eff = ranger::ranger(  mse_eff ~ 0 + .,data = cbind( tmp[,1:ind2] ,tmp %>% select(mse_eff )), num.trees= 4000, importance = "permutation" ) 
  rf_pred = ranger::ranger( mse_pred ~0+ ., data = cbind( tmp[,1:ind2] ,tmp %>% select(mse_pred )), num.trees= 4000, importance = "permutation" ) 
  rf_zero = ranger::ranger( mse_zero ~0+ ., data = cbind( tmp[,1:ind2] ,tmp %>% select(mse_zero )), num.trees= 4000, importance = "permutation" ) 
  rf_bias_effect = ranger::ranger(  bias_effect ~ 0 + .,data = cbind( tmp[,1:ind2] ,tmp %>% select(bias_effect )), num.trees= 4000, importance = "permutation" ) 
  rf_var_effect = ranger::ranger( var_effect ~0+ ., data = cbind( tmp[,1:ind2] ,tmp %>% select(var_effect )), num.trees= 4000, importance = "permutation" ) 
  rf_bias_zero = ranger::ranger(  bias_zero ~ 0 + .,data = cbind( tmp[,1:ind2] ,tmp %>% select(bias_zero )), num.trees= 4000, importance = "permutation" ) 
  rf_var_zero = ranger::ranger( var_zero ~0+ ., data = cbind( tmp[,1:ind2] ,tmp %>% select(var_zero )), num.trees= 4000, importance = "permutation" ) 
  rf_bias_pred = ranger::ranger(  bias_pred ~ 0 + .,data = cbind( tmp[,1:ind2] ,tmp %>% select(bias_pred )), num.trees= 4000, importance = "permutation" ) 
  rf_var_pred = ranger::ranger( var_pred ~0+ ., data = cbind( tmp[,1:ind2] ,tmp %>% select(var_pred )), num.trees= 4000, importance = "permutation" )   
  
  pred_eff = which.min(predict(rf_eff, data = tmp)$predictions)
  pred_pred = which.min(predict(rf_pred, data = tmp)$predictions)
  pred_bias_effect = which.min(predict(rf_bias_effect, data = tmp)$predictions)
  pred_var_effect = which.min(predict(rf_var_effect, data = tmp)$predictions)
  pred_bias_zero = which.min(predict(rf_bias_zero, data = tmp)$predictions)
  pred_var_zero = which.min(predict(rf_var_zero, data = tmp)$predictions)
  pred_bias_pred = which.min(predict(rf_bias_pred, data = tmp)$predictions)
  pred_var_pred = which.min(predict(rf_var_pred, data = tmp)$predictions)
  
  out$tmp = tmp
  
  transf_imp = function(df) {
    df = as.data.frame(df)
    df$Feature = rownames(df)
    colnames(df)[1] = "Gain"
    df$Gain = log10(df$Gain+1)
    return(df[, c(2, 1)])
  }
  
  out$preds = list(eff = pred_eff, pred = pred_pred, bias_effect = pred_bias_effect, var_effect = pred_var_effect, bias_zero = pred_bias_zero, var_zero = pred_var_zero,
                   bias_pred = pred_bias_pred, var_pred = pred_var_pred)
  
  out$importances = list(eff =   transf_imp(ranger::importance(rf_eff)),
                         pred =   transf_imp(ranger::importance(rf_pred)),
                         zero = transf_imp(ranger::importance(rf_zero)),
                         bias_effect = transf_imp(ranger::importance(rf_bias_effect)),
                         var_effect = transf_imp(ranger::importance(rf_var_effect)),
                         bias_zero = transf_imp(ranger::importance(rf_bias_zero)),
                         var_zero = transf_imp(ranger::importance(rf_bias_zero)),
                         bias_pred = transf_imp(ranger::importance(rf_bias_pred)),
                         var_pred = transf_imp(ranger::importance(rf_bias_pred))                           
  )
  
  out$hyper = list(eff = parameter[pred_eff,], pred = parameter[pred_pred,], 
                   bias_effect = parameter[pred_bias_effect,], var_effect = parameter[pred_var_effect,],
                   bias_zero = parameter[pred_bias_zero,], var_zero = parameter[pred_var_zero,],
                   bias_pred = parameter[pred_bias_pred,], var_pred = parameter[pred_var_pred,])
  
  out$data =   rbind(
    as.data.frame(coefs_eff[,c(1, 2, 4)]) %>% mutate(group = "eff") %>% mutate(hyper = rownames(coefs_eff), algorithm = method),
    as.data.frame(coefs_pred[,c(1, 2, 4)]) %>% mutate(group = "pred") %>% mutate(hyper = rownames(coefs_pred), algorithm = method),
    as.data.frame(coefs_zero[,c(1, 2, 4)]) %>% mutate(group = "zero") %>% mutate(hyper = rownames(coefs_pred), algorithm = method),
    as.data.frame(coefs_bias_effect[,c(1, 2, 4)]) %>% mutate(group = "bias_effect") %>% mutate(hyper = rownames(coefs_pred), algorithm = method),
    as.data.frame(coefs_var_effect[,c(1, 2, 4)]) %>% mutate(group = "var_effect") %>% mutate(hyper = rownames(coefs_pred), algorithm = method),
    as.data.frame(coefs_bias_zero[,c(1, 2, 4)]) %>% mutate(group = "bias_zero") %>% mutate(hyper = rownames(coefs_pred), algorithm = method),
    as.data.frame(coefs_var_zero[,c(1, 2, 4)]) %>% mutate(group = "var_zero") %>% mutate(hyper = rownames(coefs_pred), algorithm = method),
    as.data.frame(coefs_bias_pred[,c(1, 2, 4)]) %>% mutate(group = "bias_pred") %>% mutate(hyper = rownames(coefs_pred), algorithm = method),
    as.data.frame(coefs_var_pred[,c(1, 2, 4)]) %>% mutate(group = "var_pred") %>% mutate(hyper = rownames(coefs_pred), algorithm = method)    
  )
  
  return( out )
}



draw_eff = function(y, x, eff, se, col = "black") {
  segments(x+eff+1.96*se, y, x+eff-1.96*se, y)
  points(x+eff, y, cex = 0.7, pch = 19, col = col)
}

draw_bar = function(y, x, eff, w = 0.01) {
  rect(x, ybottom = y-w, ytop = y+w, xright = x+eff, 
       col = "#32312F", 
       border = "#000000")
}

draw_line = function(y, x,se, xr , yr, to_y, p = NULL, line_col) {
  y = y - mean(y)
  yy = scales::rescale(y, to = yr, from = to_y)
  conf =  scales::rescale(c(y+1.96*se, rev(y-1.96*se)), to = yr, from = to_y)
  xx = scales::rescale(x, to = xr, from = c(min(x), max(x)))
  if(!is.null(p)) pp = scales::rescale(p, to = xr, from = c(min(x), max(x)))
  polygon(c(xx, rev(xx)), conf, col = addA(line_col[1]), border = NA)
  points(xx, yy, type = "l", col = line_col[1], lwd = 3)
  if(!is.null(p)) points(pp, predict(mgcv::gam(yy~s(xx)), newdata = data.frame(xx = pp)), col = line_col[2], cex = 1, pch = 18)
}



plot_tuning = function(data, 
                       results,
                       eff_range = list(eff_range2 = c(-0.5, 0.5),eff_range1 = c(-0.04, 0.04)),
                       vi_range =  list(c(0, 0.04), c(0, 2.5)),
                       line_col = c("#24526E", "#FF6F57")
                       ) {
  par(mar = c(2, 9, 4, 2))
  plot(NULL, NULL, xlim = c(0, 1), ylim = c(0, 1), 
       xaxt = "n", yaxt="n", xlab = "", ylab = "",  xaxs = "i", yaxs = "i")
  yys = c(seq(0.97, 0.815, length.out = 7), seq(0.785-0.04, 0.03, length.out = 16))
  diff1 =   abs(diff(yys[1:2])/2)
  diff2 =   abs(diff(yys[20:21])/2)
  diff_tmp = diff1
  for(i in 1:23) {
    if(i > 7) diff_tmp = diff2
    rect(0+0.001, yys[i]+diff_tmp, 1-0.001, yys[i]-diff_tmp, col = c(addA("#f7f2e4", 0.5), "white")[i%%2 + 1], border = NA)
  }
  
  tck = 0.015
  mgp = 0.07
  abline(v = 0.495)
  abline(v = 0.505)
  segments(y0 = yys[7]-0.015, y1 = yys[7]-0.015, x0 = 0.0, x1 = 0.3, lty = 2)
  segments(y0 = yys[7+1]+0.025, y1 = yys[7+1]+0.025, x0 = 0.0, x1 = 0.3, lty = 2)
  segments(y0 = yys[7]-0.015, y1 = yys[7]-0.015, x0 = 0.0+0.5, x1 = 0.3+0.5, lty = 2)
  segments(y0 = yys[7+1]+0.025, y1 = yys[7+1]+0.025, x0 = 0.0+0.5, x1 = 0.3+0.5, lty = 2)
  rect(0.4950-0.001, 0, 0.505+0.001, 1.00, col = "black", xpd = NA, border = "black")
  rect(0.4951, -0.05, 0.5049, 1.05, col = "white", xpd = NA, border = "white")
  abline(v = 0.30)
  text(x = 0.22, y = yys[1]*0.990, pos = 3, label = "Coefs", xpd = NA)
  text(x = 0.22+0.5, y = yys[1]*0.990, pos = 3, label = "Coefs", xpd = NA)
  text(x = 0.425, y = yys[1]*0.990, pos = 3, label = "VI", xpd = NA)
  text(x = 0.425+0.5, y = yys[1]*0.990, pos = 3, label = "VI", xpd = NA)
  ytop = 1.03
  text(x = 0.25, y = ytop, pos = 3, label = latex2exp::TeX(r"( \textbf{Effect} $ \hat{\beta}_1 $ )"), xpd = NA, font = 1) #"Effect \U03B2\U0302\U2081" Effect $\hat{\beta}_1$
  text(x = 0.25+0.5, y = ytop, pos = 3, label = latex2exp::TeX(r"( \textbf{Prediction} $ \hat{y} $ )"), xpd = NA, font = 2)
  abline(v = 0.30+0.5)
  counter1 = counter2 = 1
  methods = c("NN", "BRT", "RF", "Elastic_net")

  to_y2 = c(-0.3, 0.3)*2
  to_y = list(to_y2, to_y2*3)
  xr = c(0.05, 0.25)
  xr1 = c(0.01, 0.145)
  xr2 = c(0.165, 0.29)
  xr_ranges = list(xr1, xr2)
  segments(0.15, 1.0, 0.15, y1 =  0.5*(yys[7+1]+yys[7]), lty = 2,  col = "#AAAAAA")
  segments(0.15+0.5, 1.0, 0.15+0.5, y1 =  0.5*(yys[7+1]+yys[7]), lty = 2, col = "#AAAAAA")
  segments(0.15, yys[7]-0.015, 0.15, y1 =  0, lty = 1,  col = "#AAAAAA")
  segments(0.15+0.5,  yys[7]-0.015, 0.15+0.5, y1 = 0, lty =1, col = "#AAAAAA")
  groups = c("bias_effect", "bias_pred", "zero")
  groups2 = c("eff", "pred", "zero")
  for(column in 1:2) {
    counter1  = 1
    for(m in 1:4) {
      method_tmp = methods[m]
      NN_tmp_eff = data %>% filter(algorithm == method_tmp) %>% filter(group == groups[column])
      if(m == 1) {
        NN_tmp_eff = NN_tmp_eff[c(1:7, 8, 9, 10, 11, 12),]
      }
      NN_imp_eff = results[[m]]$importances[groups2[column]][[1]]
      for(i in 1:nrow(NN_tmp_eff)) {
        ## Eff
        if( (i < 8) & (m == 1) ) {
          eff = NN_tmp_eff[i, 1]
          eff = scales::rescale(eff, to = c(-0.18, 0.18), from = eff_range[[column]])
          ses = scales::rescale(NN_tmp_eff[i, 2], to = c(-0.18, 0.18), from = eff_range[[column]])
          col = "black"
            if(i == which.min(NN_tmp_eff[1:7,1])) col = line_col[2]
            draw_eff(eff = eff, se = ses, x = 0.15+(column-1)*0.5, y = yys[counter1], col = col)
        } else {
          n = i
          if(m == 1) n = n - 7
          for(j in 1:2) {
            v = c("_effect", "_pred")[column]
            gam_eff = results[[m]][paste0("gam_", c("bias", "var")[j], v)][[1]]
            obj = plot(mgcViz::sm(mgcViz::getViz( gam_eff ), n))
            x = obj$data$fit$x
            y = obj$data$fit$y
            se = obj$data$fit$se
            best_point = results[[m]]$tmp[ results[[m]]$preds[[groups2[[column]]]],i]
            draw_line(y, x, se, xr = xr_ranges[[j]]+(column-1)*0.5,
                      to_y = results[[m]][paste0("range_", c("bias", "var")[j], v)][[1]],
                      yr = c(yys[counter1]+abs(diff(yys)[10])*0.5-0.004, yys[counter1]-abs(diff(yys)[10])*0.5+0.004)[c(2, 1)],
                      p = best_point, line_col = line_col)
          }
        }
        ## VI
        imp = (NN_imp_eff %>% filter(Feature == NN_tmp_eff$hyper[i]) %>% dplyr::select(Gain))$Gain
        if(length(imp) == 0) imp =0
        imp = scales::rescale(imp, to = c(0, 0.2), from = vi_range[[column]])
        draw_bar(x = 0.3+(column-1)*0.5,eff= imp, y = yys[counter1])
        text(xpd = NA, pos = 2, y = yys[counter1]*0.99, x = 0.0, label = labels[NN_tmp_eff$hyper[i]])
        counter1 = counter1 + 1
        # print(NN_tmp_eff$hyper[i])
      }
      if(m < 4) abline(h= 0.5*(yys[counter1]+yys[counter1-1]), lty = 1)
    }
  }
  rect(0.4951, -0.05, 0.5049, 1.05, col = "white", xpd = NA, border = "white")
  text(pos = 4, x = 1.0, y = yys[5], label = "Neural Network", xpd = NA, srt = -90)
  text(pos = 4, x = 1.0, y = yys[14], label = "BRT", xpd = NA, srt = -90)
  text(pos = 4, x = 1.0, y = yys[18], label = "RF", xpd = NA, srt = -90)
  text(pos = 4, x = 1.0, y = yys[22], label = "EN", xpd = NA, srt = -90)
  text(x = 0.15, y = 0, pos = 1, label = "hyperparameter range", xpd = NA)
  text(x = 0.15+0.5, y = 0, pos = 1, label = "hyperparameter range", xpd = NA)
  text(x = 0.15/2, y = yys[7]-0.012, pos = 1, label = "Bias", xpd = NA)
  text(x = 0.15/2+0.15, y = yys[7]-0.012, pos = 1, label = "Variance", xpd = NA)
  text(x = 0.15/2+0.5, y = yys[7]-0.012, pos = 1, label = "Bias", xpd = NA)
  text(x = 0.15/2+0.15+0.5, y = yys[7]-0.012, pos = 1, label = "Variance", xpd = NA)
  
}

