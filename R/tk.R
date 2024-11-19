resource_path <- function(filename) {
  script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
  base_dir <- normalizePath(file.path(script_dir, ".."))
  file.path(base_dir, "resources", filename)
}

# Data preprocessing
# Importing dataset
dataset = read.csv(resource_path('Position_Salaries.csv'))
dataset = dataset[2:3]

library(randomForest)
set.seed(1234)
regressor = randomForest(x = dataset[1],
                         y = dataset$Salary,
                         ntree = 100)

# Visualizing
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)

library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid)), ), colour = 'green') +
  ggtitle('Truth or Bluff (Random Forest Regression)') +
  ylab('Salary') + xlab('Level')

# Predicting Values
value = predict(regressor, newdata = data.frame(Level = 6.5))
