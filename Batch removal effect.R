#安装包，对于未安装相应包的需执行以下安装
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

#加载安装包
library("limma")
library("FactoMineR")
library("factoextra")
library("sva")
library("readxl")
library(ggplot2)
#加载去批次文件，路径与文件存储相关
#exp <- read.csv("G:/wjq/去批次效应/MDD.csv")
#group <- read.csv("G:/wjq/去批次效应/MDD_trait.csv")
exp <- read_excel("D:/proteomics/血浆/tsne-hiplot/数据信息-tsne.xlsx")
group <- read.csv("D:/proteomics/血浆/去批次效应/去批次效应结果/仅去批次_279特征-opls-da/MDD_trait.csv")

#绘制箱线图
boxplot(exp,las=2,main='exp-before')

#1.原始数据直接进行聚类分析
label <- group$dataset
dist_mat <- dist(t(exp[,2:179]))
clustering <- hclust(dist_mat, method = "complete")
plot(clustering,labels = label)


#2.PCA分析绘图
#dat.pca <- PCA(dist_mat, graph = FALSE)
dat=as.data.frame(t(exp[,2:179]))
dat.pca <- PCA(dat, graph = FALSE)
fviz_pca_ind(dat.pca,
             geom.ind = "point", # show points only (nbut not "text")
             col.ind = label, # color by groups
             #palette = c("#00AFBB", "#E7B800"),
             addEllipses = TRUE, # Concentration ellipses
             legend.title = "dataset")

#3.数据四分位归一化以及对数转化后进行聚类以及PCA分析
exp1 = normalizeBetweenArrays(exp)
exp1 = normalizeBetweenArrays(exp[,2:179])
dist_mat <- dist(t(exp1))
clustering <- hclust(dist_mat, method = "complete")
plot(clustering,labels = label)
exp1_log = log2(exp1+1)
dist_mat <- dist(t(exp1_log))
clustering <- hclust(dist_mat, method = "complete")
plot(clustering,labels = label)

dat=as.data.frame(t(exp1_log))
dat.pca <- PCA(dat, graph = FALSE)
fviz_pca_ind(dat.pca,
             geom.ind = "point", # show points only (nbut not "text")
             col.ind = label, # color by groups
             #palette = c("#00AFBB", "#E7B800"),
             addEllipses = TRUE, # Concentration ellipses
             legend.title = "dataset")
#4.数据进行limma去批次后进行聚类以及PCA分析
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
#5.数据进行sva去批次后进行聚类以及PCA分析
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
#6.数据进行对数，标准化，limma去批次后进行聚类以及PCA分析
exp4 = exp[,2:179]
exp4_log = log2(exp2 + 1)
#exp4_norm <- apply(exp4_log, 1, function(x) x - median(x, na.rm = TRUE))
exp4_norm <- normalizeBetweenArrays(exp4_log, method = "quantile")
y2 <- removeBatchEffect(exp4_norm, label)
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
write.csv(y2, file = "D:/proteomics/血浆/去批次效应/去批次效应结果/limma_result_516.csv",row.names = F)
#7.数据进行对数，标准化，sva去批次后进行聚类以及PCA分析
exp5 = exp[,2:179]
exp5_log = log2(exp5 + 1)
exp5_norm <- normalizeBetweenArrays(exp5_log, method = "quantile")
mod = model.matrix(~as.factor(group$diease))
y = ComBat(dat=exp5_norm, batch=label, mod=mod)
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
write.csv(y, file = "D:/proteomics/血浆/去批次效应/去批次效应结果/sva_516_log_norm.csv",row.names = F)
