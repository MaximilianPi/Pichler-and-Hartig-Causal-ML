# results = readRDS("results.RDS")
# 
# res = results$collidier2
# cols = c("black", 
#          RColorBrewer::brewer.pal(9, name = "Spectral")[1:3], 
#          RColorBrewer::brewer.pal(9, name = "PiYG")[3:4], 
#          RColorBrewer::brewer.pal(11, name = "RdYlBu")[8:10])
# 
# 
# plot_case = function(res) {
#   res = abind::abind(res, along = -1L)
#   # standarizing:
#   
#   res2 = abs(res)
#   for(i in 1:5) {
#     res2[,i,]=  res2[,i,]- res2[,5,]
#   }
#   
#   
#   
#   res3 = abs(res2)
#   mm = apply(res2, c(1,3), max)
#   for(i in 1:5) {
#     res3[,i,]= res3[,i,] / mm
#   }
#   
#   
#   res4= apply(res3, 2:3, function(i) mean(i, na.rm=TRUE))[-5,]
#   sds = apply(res3, 2:3, function(i) sd(i, na.rm=TRUE))[-5,]
#   
#   plot(NULL, NULL, xaxt="n", xlim = c(0.9, 4.1), ylim = c(0., 1.))
#   for(i in 1:9) points(y = res4[,i], x =(1:4)+seq(-0.15, 0.15, length.out = 9)[i], col=cols[i], pch = 15)
#   for(i in 1:9) {
#     for(j in 1:4) {
#       segments(y0 = res4[j,i]-sds[j,i],y1=res4[j,i]+sds[j,i], x0 =(1:4)[j]+seq(-0.15, 0.15, length.out = 9)[i], col=cols[i])
#     }
#   }
#   axis(1, at = 1:4, labels = LETTERS[1:4])
#   legend("topright", legend = c("LM","RF_gini", "RF_perm", "RF_global", "BRT_imp", "BRT_global", "NN_l2", "NN_l1", "NN_drop"), pch = 15, col= cols)
# }  
# 
# names(results)
# 
# 
# par(mfrow = c(2,2))
# for(i in 1:4) plot_case(results[[i]])
# 

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

xx = 0
yy = 1.6
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
  
  text(x = xx, y = yy, xpd = NA, label = "Ind effects")
  
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
  text(x = xx, y = yy, xpd = NA, label = "Collinear")
  
  
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
  text(x = xx, y = yy, xpd = NA, label = "Confounder equal")
  
  
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
  text(x = xx, y = yy, xpd = NA, label = "Confounder unequal")
  
  
  
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
  text(x = xx, y = yy, xpd = NA, label = "Collidier")
  
}


sc = c("effects", "collinearity", "confounder_equal", "confounder", "collidier")



algorithms = c("LM","RF",  "BRT", "NN", "NN_l2", "NN_l1", "NN_drop")



pdf(file="results_total.pdf", width = 10, height = 12)
par(mfrow = c(8,6), mar = rep(1, 4), oma = c(1, 2, 2, 1))
plot.new()
text(0.5, 0.5, label ="Simulated scenarios", cex = 1.4)
labs =  c("LM","RF",  "BRT", "NN", "NN_l2", "NN_l1", "NN_drop")
plot_scenarios()


for(j in 1:7) {
  plot.new()
  text(0.5, 0.5, label =labs[j], cex = 1.4)
  
  for(i in 1:5) {
    col_tmp = "#B0A8B9"
    if(i > 2) col_tmp = "pink"
    res = apply(abind::abind(results[[sc[i]]], along = 0L), 2:3, function(i) mean(i, na.rm=TRUE))
    
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

