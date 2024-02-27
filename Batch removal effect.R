#Install the package. If no package is installed, perform the following operations
#if (!requireNamespace("BiocManager", quietly = TRUE))
#install.packages("BiocManager")
#BiocManager::install("limma")
#BiocManager::install("Biobase")
#BiocManager::install("edgeR")
#BiocManager::install('bladderbatch')
#BiocManager::install('sva')
#install.packages("FactoMineR")
#install.packages("factoextra")
#install.packages("rstatix")
#install.packages("readxl")


library("limma")
library("FactoMineR")
library("factoextra")
library("sva")
library("readxl")
library(ggplot2)

exp <- read_excel("./data_information.xlsx")#Sample and feature information
group <- read.csv("./MDD_trait.csv")#Sample, batch, and class information

#Raw data box diagram
boxplot(exp,las=2,main='exp-before')

#Cluster analysis is performed directly on the raw data
label <- group$dataset
dist_mat <- dist(t(exp[,2:179]))
clustering <- hclust(dist_mat, method = "complete")
plot(clustering,labels = label)

#PCA analysis
#dat.pca <- PCA(dist_mat, graph = FALSE)
dat=as.data.frame(t(exp[,2:179]))
dat.pca <- PCA(dat, graph = FALSE)
fviz_pca_ind(dat.pca,
             geom.ind = "point", # show points only (nbut not "text")
             col.ind = label, # color by groups
             #palette = c("#00AFBB", "#E7B800"),
             addEllipses = TRUE, # Concentration ellipses
             legend.title = "dataset")


#limma batch removal
exp2 = exp[,2:179]
y2 <- removeBatchEffect(exp2, label)
dist_mat <- dist(t(y2))
clustering <- hclust(dist_mat, method = "complete")
plot(clustering,labels = label)

dat=as.data.frame(t(y2))
dat.pca <- PCA(dat, graph = FALSE)
fviz_pca_ind(dat.pca,
             geom.ind = "point", # show points only (nbut not "text")
             col.ind = label, # color by groups
             #palette = c("#00AFBB", "#E7B800"),
             addEllipses = TRUE, # Concentration ellipses
             legend.title = "dataset")
#sva batch removal
exp3 = exp[,2:179]
mod = model.matrix(~as.factor(group$diease))
y = ComBat(dat=exp3, batch=label, mod=mod)
dist_mat <- dist(t(y))
clustering <- hclust(dist_mat, method = "complete")
plot(clustering,labels = label)
dat=as.data.frame(t(y))
dat.pca <- PCA(dat, graph = FALSE)
fviz_pca_ind(dat.pca,
             geom.ind = "point", # show points only (nbut not "text")
             col.ind = label, # color by groups
             #palette = c("#00AFBB", "#E7B800"),
             addEllipses = TRUE, # Concentration ellipses
             legend.title = "dataset")

write.csv(y, file = "./sva_result_516.csv",row.names = F)

