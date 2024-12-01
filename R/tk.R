resource_path <- function(filename) {
  script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
  base_dir <- normalizePath(file.path(script_dir, ".."))
  file.path(base_dir, "resources", filename)
}

# Data preprocessing
# Importing dataset
# Creating Matrix
library(arules)
dataset = read.transactions(
  resource_path('Market_Basket_Optimisation.csv'),
  sep = ',',
  rm.duplicates = TRUE
)
summary(dataset)

itemFrequencyPlot(dataset, topN = 10)

# Training Apriori Model
rules = apriori(dataset, parameter = list(support =0.003, confidence = 0.4))
inspect(sort(rules, by = 'lift')[1:10])
