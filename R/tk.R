resource_path <- function(filename) {
  script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
  base_dir <- normalizePath(file.path(script_dir, ".."))
  file.path(base_dir, "resources", filename)
}

# Data preprocessing
# Importing dataset
dataset = read.csv(resource_path('Mall_Customers.csv'))
X <- dataset[4:5]

# Dendrogram
dendrogram = hclust(dist(X, method = 'euclidean'), method = 'ward.D')
plot(
  dendrogram,
  main = paste('Dendrogram'),
  xlab = 'Customers',
  ylab = 'Euclidean Distances'
)


# Applying Hierarchical Clustering
hc = hclust(dist(X, method = 'euclidean'), method = 'ward.D')
y_hc = cutree(hc, 5)


# Visualizing the clusters
library(cluster)
clusplot(
  X,
  y_hc,
  lines = 0,
  shade = TRUE,
  color = TRUE,
  labels = 2,
  plotchar = FALSE,
  span = TRUE,
  main = paste('Clusters of clients'),
  xlab = 'Annual Income',
  ylab = 'Spending Score'
)