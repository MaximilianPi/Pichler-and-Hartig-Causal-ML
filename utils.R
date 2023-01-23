
addA = function(col, alpha = 0.25) apply(sapply(col, grDevices::col2rgb)/255, 2, function(x) grDevices::rgb(x[1], x[2], x[3], alpha=alpha))

eqarrowPlot <- function(graph, layout, edge.lty=rep(1, ecount(graph)),edge.width=rep(1, ecount(graph)),
                        edge.arrow.size=rep(1, ecount(graph)), cols = c( "pink","pink", "skyblue"), edge.arrow.mode = NULL, edge.colors = NULL,
                        rangeX = c(0, 1), rangeY = c(0, 2), ...) {
  vertex.label.cex = 1.6
  plot(graph, edge.lty=0, edge.arrow.size=0, layout=layout,
       vertex.shape="none",  vertex.size=50, vertex.color = cols, rescale=FALSE, xlim = rangeX, ylim = rangeY, vertex.label.cex = vertex.label.cex)
  if(is.null(edge.arrow.mode)) edge.arrow.mode = rep(">", (ecount(graph)))
  if(is.null(edge.colors)) edge.colors = rep(NULL, ecount(graph))
  for (e in seq_len(ecount(graph))) {
    graph2 <- delete.edges(graph, E(graph)[(1:ecount(graph))[-e]])
    plot(graph2, edge.lty=edge.lty[e], edge.arrow.size=edge.arrow.size[e], layout=layout,edge.color = edge.colors[e],
         vertex.label=NA, add=TRUE, vertex.color = cols, edge.width=edge.width[e], vertex.size=50,edge.arrow.mode = edge.arrow.mode[e], rescale=FALSE, 
         xlim = rangeX, ylim = rangeY, vertex.label.cex = vertex.label.cex)
  }
  plot(graph, edge.lty=0, 
       edge.arrow.size=0, 
       layout=layout, 
       add=TRUE,
       vertex.size=50, 
       vertex.color = cols,
       vertex.label.cex = vertex.label.cex,
       vertex.label.color="black", xlim = rangeX, ylim = rangeY,
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
