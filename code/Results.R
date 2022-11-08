files =        c("collinearity_0.5.RDS", 
                 "collinearity_0.90.RDS", 
                 "collinearity_0.99.RDS", 
                 "effects.RDS", 
                 "no_effects.RDS", 
                 "confounder_unequal.RDS", 
                 "confounder.RDS")

Results = 
  lapply(files, function(f) {
    confounder = readRDS(paste0("results/",f))
    Result = do.call(rbind, lapply(1:7, function(j) (apply(abind::abind(lapply(1:100, function(i) confounder[[i]][[j]]), along = 0L), 2, mean))))
    colnames(Result) = LETTERS[1:5]
    rownames(Result) = c("LM", "RF", "BRT", "NN", "l1", "l2", "l1l2")
    return(Result)
  })
names(Results) = unlist(strsplit(files, ".RDS", TRUE))


Results_sd = 
  lapply(files, function(f) {
    confounder = readRDS(paste0("results/",f))
    Result = do.call(rbind, lapply(1:7, function(j) (apply(abind::abind(lapply(1:100, function(i) confounder[[i]][[j]]), along = 0L), 2, sd))))
    colnames(Result) = LETTERS[1:5]
    rownames(Result) = c("LM", "RF", "BRT", "NN", "l1", "l2", "l1l2")
    return(Result)
  })
names(Results_sd) = unlist(strsplit(files, ".RDS", TRUE))




library(igraph)
addA = function(col, alpha = 0.25) apply(sapply(col, grDevices::col2rgb)/255, 2, function(x) grDevices::rgb(x[1], x[2], x[3], alpha=alpha))


## The plotting function
eqarrowPlot <- function(graph, layout, edge.lty=rep(1, ecount(graph)),edge.width=rep(1, ecount(graph)),
                        edge.arrow.size=rep(1, ecount(graph)), cols = c( "pink","pink", "skyblue"), edge.arrow.mode = NULL, edge.colors = NULL,
                        rangeX = c(0, 1), rangeY = c(0, 2), ...) {
  plot(graph, edge.lty=0, edge.arrow.size=0, layout=layout,
       vertex.shape="none",  vertex.size=50, vertex.color = cols, rescale=FALSE, xlim = rangeX, ylim = rangeY)
  if(is.null(edge.arrow.mode)) edge.arrow.mode = rep(">", (ecount(graph)))
  if(is.null(edge.colors)) edge.colors = rep(NULL, ecount(graph))
  for (e in seq_len(ecount(graph))) {
    graph2 <- delete.edges(graph, E(graph)[(1:ecount(graph))[-e]])
    plot(graph2, edge.lty=edge.lty[e], edge.arrow.size=edge.arrow.size[e], layout=layout,edge.color = edge.colors[e],
         vertex.label=NA, add=TRUE, vertex.color = cols, edge.width=edge.width[e], vertex.size=50,edge.arrow.mode = edge.arrow.mode[e], rescale=FALSE, 
         xlim = rangeX, ylim = rangeY)
  }
  plot(graph, edge.lty=0, 
       edge.arrow.size=0, 
       layout=layout, 
       add=TRUE,
       vertex.size=50, 
       vertex.color = cols,
       vertex.label.color="black", xlim = rangeX, ylim = rangeY,
       edge.label.color = "black", rescale=FALSE,...)
  invisible(NULL)
}



layout = matrix(c(0,10,
                  0,5,
                  0,0,
                  5,5), nrow = 4L, 2L, byrow = TRUE) 
plot_scenarios = function(cex_fac = 1.3, layout = layout) {
  

  # Effects
  g1 <- graph(c("X1", "Y", "X2", "Y", "X3", "Y"),  
              directed=TRUE ) 
  #g1 = permute(g1, c(2, 1, 3, 4))
  eqarrowPlot(g1, layout, 
              #cols = c( "skyblue","#B0A8B9", "#B0A8B9", "#B0A8B9"),
              cols = c(addA(rep("#87CEEB", 1), 1.0), "#B0A8B9", addA(rep("#87CEEB", 1), 1.0), addA(rep("#87CEEB", 2), 1.0)),
              edge.arrow.size=c(1, 0.5,1.0), 
              edge.width=c(1, 0.5,1.0)*cex_fac,
              edge.label = c("1.0\n","0.5\n","1.0"), 
              edge.label.cex = 1.4,
              edge.colors = c(rep("grey", 2), "grey"))
  
  text(letters[1], cex = 1.9, x = -1.6, y = 1.5, xpd = NA, font = 2)
  
  #text(x = xx, y = yy, xpd = NA, label = "Ind effects")
  
  
  # Collinearity
  g1 <- graph(c("X1", "Y", "X2","Y", "X3", "Y", "X1", "X2"),  
              directed=TRUE ) 
  #g1 = permute(g1, c(2, 1, 3, 4))
  eqarrowPlot(g1, layout, 
              #cols = c( "skyblue","#B0A8B9", "#B0A8B9", "#B0A8B9"),
              cols = c(addA(rep("#87CEEB", 1), 1.0), "#B0A8B9", addA(rep("#87CEEB", 1), 1.0), addA(rep("#87CEEB", 2), 1.0)),
              edge.arrow.size=c(1.0, 0.0,1.0,0.95)*cex_fac, 
              edge.width=c(1.0, 0.5,1.0,0.95)*cex_fac,
              edge.label = c("1.0\n","\n","1.0", "\n0.90"), 
              edge.label.cex = 1.4, 
              edge.arrow.mode = c(rep(">", 3), "-"), 
              edge.colors = c(rep("grey", 1),"white","grey", "#ffab02"))
  text(letters[3], cex = 1.9, x = -1.6, y = 1.5, xpd = NA, font = 2)
  
  # Confounder
  g1 <- graph(c("X1", "Y", "X2","Y", "X3", "Y", "X1", "X2"),  
              directed=TRUE ) 
  #g1 = permute(g1, c(2, 1, 3, 4))
  eqarrowPlot(g1, layout, 
              #cols = c( "skyblue","#B0A8B9", "#B0A8B9", "#B0A8B9"),
              cols = c(addA(rep("#87CEEB", 1), 1.0), "#B0A8B9", addA(rep("#87CEEB", 1), 1.0), addA(rep("#87CEEB", 2), 1.0)),
              edge.arrow.size=c(1.0, 0.5,1.0,0.95)*cex_fac, 
              edge.width=c(1.0, 0.5,1.0,0.95)*cex_fac,
              edge.label = c("-1.0\n","0.5\n","1.0", "\n0.90"), 
              edge.label.cex = 1.4, 
              edge.arrow.mode = c(rep(">", 3), "-"), 
              edge.colors = c(rep("grey", 2),"grey", "#ffab02"))
  text(letters[2], cex = 1.9, x = -1.6, y = 1.5, xpd = NA, font = 2)
  
  #text(x = xx, y = yy, xpd = NA, label = "Confounder")
  
  
  
  #text(x = xx, y = yy, xpd = NA, label = "Confounder")
  
}


sc = c("no_effects", "effects", "confounder_unequal", "collinearity_0.90")

algorithms = c("LM","RF",  "BRT", "NN", "l1", "l2", "l1l2")

pdf(file="plots/results.pdf", width = 10, height = 9)
par(mfcol = c(3,6), mar = c(5,0.5, 2, 1.4), oma = c(1, 2, 2, 1))
labs =  c("LM","RF",  "BRT", "NN", "l1", "l2", "Glmnet")
#plot_scenarios(1.0)
#dev.off()
cex_fac = 1.3

plot_scenarios(1.0, layout = matrix(c(1,1,
                             0,1,
                             0,0,
                             0,2), nrow = 4L, 2L, byrow = TRUE))

true_effs = matrix(c(
  NA, NA, NA,
  1, 0.5, 1,
  -1, 0.5, 1,
  1, 0, 1
), 4, 3, byrow = TRUE)

for(i in c(1, 2, 3, 4, 7)) {
 # plot.new()
  #text(0.5, 0.5, label =labs[j], cex = 1.4)
  counter = 1
  for(j in c(2, 4, 3)) {

    tmp = Results[[sc[j]]]
    sd = Results_sd[[sc[j]]][i,]
    edges = round(tmp[i,], 5)
    
    bias = edges[c(1, 2, 5)] - true_effs[j,]
    #P_val = (pt(abs(bias/(sd[c(1,2, 5)]/sqrt(n))),df = 94, lower.tail = FALSE )*2)
    
    g1 <- graph(c("X1", "Y", "X2", "Y", "X3", "Y"),  
                directed=TRUE ) 
    #g1 = permute(g1, c(2, 1, 3, 4))
    
    layout_as_tree(g1, root = "Y", circular = TRUE, flip.y = TRUE)
    eqarrowPlot(g1, matrix(c(1,1,
                             0,1,
                             0,0,
                             0,2), nrow = 4L, 2L, byrow = TRUE) ,
                #cols = c( "skyblue","#B0A8B9", "#B0A8B9", "#B0A8B9"),
                cols = c(addA(rep("#87CEEB", 1), 1.0), "#B0A8B9", addA(rep("#87CEEB", 1), 1.0), addA(rep("#87CEEB", 1), 1.0)),
                edge.arrow.size=abs(edges[c(1, 2, 5)]), 
                edge.width=abs(edges[c(1, 2, 5)])*cex_fac,
                edge.label = c(paste0(format(round(bias, 2)[1], nsmall = 1), "\n\n"),paste0(format(round(bias, 2)[2], nsmall = 1), "\n"), paste0("", format(round(bias, 2)[3], nsmall=1))),
                edge.label.cex = 1.4,
                edge.colors = ifelse(abs(edges[c(1, 2, 5)]) < 0.001, "white", "grey"))
    if(any(P_val > 0.05)) points(0.5, 1, pch = 16, col = "green")
    text(labs[i], x = 0, y = 2.3, xpd = NA, cex = 1.4, pos = 3)
    if(i == 1) {
      text(letters[counter], cex = 1.9, x = -2.2, y = 2.5, xpd = NA, font = 2)
      counter = counter + 1
    }

  }
}

dev.off()

### Fig 2
pdf(file = "plots/scenarios.pdf", width = 8, height = 3)
par(mfrow = c(1, 3))
plot_scenarios(1.0)
dev.off()




### Fig 1
# Effects
pdf(file = "plots/scenarios_fig1.pdf", width = 10, height = 5)
par(mfrow = c(1, 4), mar = rep(2, 4))

g1 <- graph(c("X1", "Y", "X2", "Y", "X3", "Y"),  
            directed=TRUE ) 
#g1 = permute(g1, c(2, 1, 3, 4))

eqarrowPlot(g1, layout_as_tree(g1, root = "Y", circular = TRUE, flip.y = TRUE), 
            #cols = c( "skyblue","#B0A8B9", "#B0A8B9", "#B0A8B9"),
            cols = c(addA(rep("#87CEEB", 1), 1.0), "#B0A8B9", addA(rep("#87CEEB", 1), 1.0), addA(rep("#87CEEB", 2), 1.0)),
            edge.arrow.size=rep(1.0, 4), 
            edge.width=rep(1.5, 4),
            edge.label = "", 
            edge.label.cex = 1.4,
            edge.colors = c(rep("grey", 2), "grey"))

text(letters[1], cex = 1.9, x = -1.6, y = 1.5, xpd = NA, font = 2)



g1 <- graph(c("X1", "Y", "X2", "X1", "X3", "Y"),  
            directed=TRUE ) 
#g1 = permute(g1, c(2, 1, 3, 4))

eqarrowPlot(g1, layout_as_tree(g1, root = "Y", circular = TRUE, flip.y = TRUE), 
            #cols = c( "skyblue","#B0A8B9", "#B0A8B9", "#B0A8B9"),
            cols = c(addA(rep("#87CEEB", 1), 1.0), "#B0A8B9", addA(rep("#87CEEB", 1), 1.0), addA(rep("#87CEEB", 2), 1.0)),
            edge.arrow.size=rep(1.0, 4), 
            edge.width=rep(1.5, 4),
            edge.label = "", 
            edge.label.cex = 1.4,
            edge.colors = c("red", "red","grey", "red"))
text(letters[2], cex = 1.9, x = -1.6, y = 1.5, xpd = NA, font = 2)


g1 <- graph(c("X1", "Y", "X2","Y", "X3", "Y", "X1", "X2"),  
            directed=TRUE ) 
#g1 = permute(g1, c(2, 1, 3, 4))
eqarrowPlot(g1, layout_as_tree(g1, root = "Y", circular = TRUE, flip.y = TRUE), 
            #cols = c( "skyblue","#B0A8B9", "#B0A8B9", "#B0A8B9"),
            cols = c(addA(rep("#87CEEB", 1), 1.0), "#B0A8B9", addA(rep("#87CEEB", 1), 1.0), addA(rep("#87CEEB", 2), 1.0)),
            edge.arrow.size=rep(1.0, 4), 
            edge.width=rep(1.5, 4),
            edge.label = c(""), 
            edge.label.cex = 1.4, 
            #edge.arrow.mode = c(rep(">", 3), "-"), 
            edge.colors = c("red", "grey","grey", "red"))
text(letters[3], cex = 1.9, x = -1.6, y = 1.5, xpd = NA, font = 2)


g1 <- graph(c("X1", "Y", "X2","Y", "X3", "Y", "X1", "X2"),  
            directed=TRUE ) 
#g1 = permute(g1, c(2, 1, 3, 4))
eqarrowPlot(g1, layout_as_tree(g1, root = "Y", circular = TRUE, flip.y = TRUE), 
            #cols = c( "skyblue","#B0A8B9", "#B0A8B9", "#B0A8B9"),
            cols = c(addA(rep("#87CEEB", 1), 1.0), "#B0A8B9", addA(rep("#87CEEB", 1), 1.0), addA(rep("#87CEEB", 2), 1.0)),
            edge.arrow.size=rep(1.0, 4), 
            edge.width=rep(1.5, 4),
            edge.label = c(""), 
            edge.label.cex = 1.4, 
            edge.arrow.mode = c("<", ">", ">", "<"), 
            edge.colors = c("red", "grey","grey", "red"))
text(letters[4], cex = 1.9, x = -1.6, y = 1.5, xpd = NA, font = 2)

dev.off()





effs_true = list(
  confounder.RDS = c(1, 1, 0, 0, 1),
  confounder_unequal.RDS = c(-1, 0.5, 0, 0, 1),
  collinearity_0.90.RDS = c(1, 0, 0, 0, 1),
  collinearity_0.5.RDS = c(1, 0, 0, 0, 1),
  collinearity_0.99.RDS = c(1, 0, 0, 0, 1),
  effects.RDS = c(1.0, 0.5, 1.0, 0.0, 1.0),
  no_effects.RDS = c(0.0, 0.0, 0.0, 0.0, 0.0)
)


Results_bias = 
  lapply(files, function(f) {
    confounder = readRDS(paste0("results/",f))
    Result = do.call(rbind, lapply(1:7, function(j) (apply(abind::abind(lapply(1:100, function(i) confounder[[i]][[j]]), along = 0L), 2, mean))))
    colnames(Result) = LETTERS[1:5]
    rownames(Result) = c("LM", "RF", "BRT", "NN", "l1", "l2", "l1l2")
    Result = Result -  matrix(effs_true[[f]], ncol = 5,nrow = 7, byrow = TRUE)
    return(Result)
  })
names(Results_bias) = unlist(strsplit(files, ".RDS", TRUE))

Results_var = 
  lapply(files, function(f) {
    confounder = readRDS(paste0("results/",f))
    Result = do.call(rbind, lapply(1:7, function(j) (apply(abind::abind(lapply(1:100, function(i) confounder[[i]][[j]]), along = 0L), 2, var))))
    colnames(Result) = LETTERS[1:5]
    rownames(Result) = c("LM", "RF", "BRT", "NN", "l1", "l2", "l1l2")
    return(Result)
  })
names(Results_var) = unlist(strsplit(files, ".RDS", TRUE))
df = as.data.frame(
  cbind(
    abs(round(Results_bias$effects[c(1,2, 3, 4, 7), c(1, 2, 5)], 3)),
    abs(round(Results_bias$confounder_unequal[c(1,2, 3, 4, 7), c(1, 2, 5)], 3)),
    abs(round(Results_bias$collinearity_0.90[c(1,2, 3, 4, 7), c(1, 2, 5)], 3)))
)
colnames(df) = as.vector(sapply(1:3, function(i) paste0(i, paste0("X",1:3))))

ft = flextable::flextable(df)
