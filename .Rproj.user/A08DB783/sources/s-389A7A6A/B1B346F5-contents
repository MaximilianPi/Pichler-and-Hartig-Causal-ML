library(igraph)


## The plotting function
eqarrowPlot <- function(graph, layout, edge.lty=rep(1, ecount(graph)),edge.width=rep(1, ecount(graph)),
                        edge.arrow.size=rep(1, ecount(graph)), cols = c( "pink","pink", "skyblue"),...) {
  plot(graph, edge.lty=0, edge.arrow.size=0, layout=layout,
       vertex.shape="none",  vertex.size=50, vertex.color = cols)
  for (e in seq_len(ecount(graph))) {
    graph2 <- delete.edges(graph, E(graph)[(1:ecount(graph))[-e]])
    plot(graph2, edge.lty=edge.lty[e], edge.arrow.size=edge.arrow.size[e], layout=layout,
         vertex.label=NA, add=TRUE, vertex.color = cols, edge.width=edge.width[e], vertex.size=50)
  }
  plot(graph, edge.lty=0, edge.arrow.size=0, layout=layout, add=TRUE, vertex.size=50, vertex.color = cols,...)
  invisible(NULL)
}

plot_scenarios = function() {
  # Effects
  g1 <- graph(c("X", "Y", "A", "Y"),  
              directed=TRUE ) 
  g1 = permute(g1, c(2, 1, 3))
  eqarrowPlot(g1, layout_as_tree(g1, root = "A"), 
              cols = c( "skyblue","#B0A8B9", "#B0A8B9"),
              edge.arrow.size=c(1, 0.5), 
              edge.width=c(1, 0.5), 
              edge.label = c("1\n","0.5\n"), 
              edge.label.cex = 1.4)
  
  ### Collinear
  g1 <- graph(c("X", "Y","A","Y", "A", "X"),  
              directed=TRUE ) 
  g1 = permute(g1, c(2, 1, 3))
  E(g1)$color = c("grey", "white", "grey")
  eqarrowPlot(g1, layout_as_tree(g1, root = "A"), 
              cols = c( "skyblue","#B0A8B9", "#B0A8B9"),
              edge.arrow.size=c(1, 0, 0), 
              edge.width=c(1, 0, 1), 
              edge.label = c("1\n"," \n",""), 
              edge.label.cex = 1.4)
  
  ### Confounder_equal
  g1 <- graph(c("X", "Y", "A", "Y", "A", "X"),  
              directed=TRUE ) 
  g1 = permute(g1, c(2, 1, 3))
  eqarrowPlot(g1, layout_as_tree(g1, root = "A"), 
              cols = c( "skyblue","#B0A8B9", "pink"),
              edge.arrow.size=c(1, 1, 1), 
              edge.width=c(1, 1, 1), 
              edge.label = c("1\n","1\n","1\n"), 
              edge.label.cex = 1.4)
  
  
  ### Confounder_unequal
  g1 <- graph(c("X", "Y", "A", "Y", "A", "X"),  
              directed=TRUE ) 
  g1 = permute(g1, c(2, 1, 3))
  eqarrowPlot(g1, layout_as_tree(g1, root = "A"), 
              cols = c( "skyblue","#B0A8B9", "pink"),
              edge.arrow.size=c(0.5, 1, 1), 
              edge.width=c(1, 1, 1), 
              edge.label = c("0.5\n","-1\n","1\n"), 
              edge.label.cex = 1.4)
  
  
  ### Collidier
  g1 <- graph(c("X", "Y", "Y", "A", "X", "A"),  
              directed=TRUE ) 
  g1 = permute(g1, c(2, 1, 3))
  eqarrowPlot(g1, layout_as_tree(g1, root = "A"), 
              cols = c( "skyblue","#B0A8B9", "pink"),
              edge.arrow.size=c(1, 1, 1), 
              edge.width=c(1, 1, 1), 
              edge.label = c("1\n","1\n","1\n"), 
              edge.label.cex = 1.4)
}


sc = c("effects", "collinearity", "confounder_equal", "confounder", "collidier")
results2 = 
  lapply(results, function(res) {
    res = abind::abind(res, along = -1L)
    res2= apply(res, 2:3, function(i) mean(i, na.rm=TRUE))
    res3 = (abs(res2 - do.call(rbind, lapply(1:5, function(i) apply(res2, 2, min) )) ))
    res4 = (res3 / do.call(rbind, lapply(1:5, function(i) apply(res3, 2, max))))
    return(res4)
  })
algorithms = c("LM","RF_gini", "RF_perm", "RF_global", "BRT_imp", "BRT_global","NN", "NN_l2", "NN_l1", "NN_drop")



pdf(file="results_RF.pdf", width = 10, height = 7)
par(mfrow = c(5,6), mar = rep(1, 4))
plot.new()
text(0.5, 0.5, label ="Simulated scenarios", cex = 1.4)
labs = c("Linear Regression", "RF Gini", "RF Permutatation", "RF external Perm.")
plot_scenarios()

for(j in 1:4) {
  plot.new()
  text(0.5, 0.5, label =labs[j], cex = 1.4)
  
  for(i in 1:5) {
    col_tmp = "#B0A8B9"
    if(i > 2) col_tmp = "pink"
    res = results2[[sc[i]]]
    effs = res[1:2,j]
    if(!any(is.na(effs))) {
      g1 <- graph(c("X", "Y", "A", "Y"),  
                  directed=TRUE ) 
      g1 = permute(g1, c(2, 1, 3))
      eqarrowPlot(g1, layout_as_tree(g1, root = "A"), 
                  cols = c( "skyblue","#B0A8B9", col_tmp),
                  edge.arrow.size=c(effs[1], effs[2]), 
                  edge.width=effs, 
                  edge.label = paste0(round(effs, 2), "\n"), 
                  edge.label.cex = 1.4)
    } else {
      plot.new()
    }
  }
}

dev.off()



pdf(file="results_BRT.pdf", width = 10, height = 6)
par(mfrow = c(4,6), mar = rep(1, 4))
plot.new()
text(0.5, 0.5, label ="Simulated scenarios", cex = 1.4)
labs = c("Linear Regression", "BRT Importance", "BRT external Perm.")
plot_scenarios()

counter = 1
for(j in c(1, 5, 6)) {
  plot.new()
  text(0.5, 0.5, label =labs[counter], cex = 1.4)
  counter = counter + 1
  for(i in 1:5) {
    col_tmp = "#B0A8B9"
    if(i > 2) col_tmp = "pink"
    res = results2[[sc[i]]]
    effs = res[1:2,j]
    if(!any(is.na(effs))) {
      g1 <- graph(c("X", "Y", "A", "Y"),  
                  directed=TRUE ) 
      g1 = permute(g1, c(2, 1, 3))
      eqarrowPlot(g1, layout_as_tree(g1, root = "A"), 
                  cols = c( "skyblue","#B0A8B9", col_tmp),
                  edge.arrow.size=c(effs[1], effs[2]), 
                  edge.width=effs, 
                  edge.label = paste0(round(effs, 2), "\n"), 
                  edge.label.cex = 1.4)
    } else {
      plot.new()
    }
  }
}
dev.off()


pdf(file="results_NN.pdf", width = 10, height = 8)
par(mfrow = c(6,6), mar = rep(1, 4))
plot.new()
text(0.5, 0.5, label ="Simulated scenarios", cex = 1.4)
labs = c("Linear Regression", "NN external Perm.", "NN+l2 external Perm.", "NN+l1 external Perm.", "NN+Drop external Perm.")
plot_scenarios()

counter = 1
for(j in c(1, 7:10)) {
  plot.new()
  text(0.5, 0.5, label =labs[counter], cex = 1.4)
  counter = counter + 1
  for(i in 1:5) {
    col_tmp = "#B0A8B9"
    if(i > 2) col_tmp = "pink"
    res = results2[[sc[i]]]
    effs = res[1:2,j]
    if(!any(is.na(effs))) {
      g1 <- graph(c("X", "Y", "A", "Y"),  
                  directed=TRUE ) 
      g1 = permute(g1, c(2, 1, 3))
      eqarrowPlot(g1, layout_as_tree(g1, root = "A"), 
                  cols = c( "skyblue","#B0A8B9", col_tmp),
                  edge.arrow.size=c(effs[1], effs[2]), 
                  edge.width=effs, 
                  edge.label = paste0(round(effs, 2), "\n"), 
                  edge.label.cex = 1.4)
    } else {
      plot.new()
    }
  }
}
dev.off()





algorithms = c("LM","RF_gini", "RF_perm", "RF_global", "BRT_imp", "BRT_global","NN", "NN_l2", "NN_l1", "NN_drop")




dev.off()
            