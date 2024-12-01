resource_path <- function(filename) {
  script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
  base_dir <- normalizePath(file.path(script_dir, ".."))
  file.path(base_dir, "resources", filename)
}

# Data preprocessing
# Importing dataset
dataset = read.csv(resource_path('Mall_Customers.csv'))
X <- dataset[4:5]

# Elbow Method
set.seed(6)
wcss <- vector()
for (i in 1:10)
  wcss[i] <- sum(kmeans(X, i)$withinss)
plot(
  1:10,
  wcss,
  type = 'b',
  main = paste('Clusters of clients'),
  xlab = 'Number of clusters',
  ylab = 'WCSS'
)


# Applying K-Means Clustering
set.seed(29)
kmeans <- kmeans(X, 5, iter.max = 300, nstart = 10)

# Visualizing the clusters
library(cluster)
clusplot(
  X,
  kmeans$cluster,
  lines = 0,
  shade = TRUE,
  color = TRUE,
  labels = 2,
  plotchar = FALSE,
  span = TRUE,
  main = paste('Clusters of clients'),
  xlab = 'Number of clusters',
  ylab = 'WCSS'
)