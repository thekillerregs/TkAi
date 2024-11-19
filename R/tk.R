resource_path <- function(filename) {
  script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
  base_dir <- normalizePath(file.path(script_dir, ".."))
  file.path(base_dir, "resources", filename)
}

# Data preprocessing
# Importing dataset
dataset = read.csv(resource_path('Position_Salaries.csv'))
dataset = dataset[2:3]

library(rpart)
regressor = rpart(formula = Salary ~ .,
                  data = dataset,
                  control = rpart.control(minsplit = 1))

# Visualizing
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)

library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
  geom_line(aes(
    x = x_grid,
    y = predict(regressor, newdata = data.frame(Level = x_grid)),
    
  ), colour = 'green') +
  ggtitle('Truth or Bluff (Decision Tree Regression)') +
  ylab('Salary') + xlab('Level')

# Predicting Values
value = predict(regressor, newdata = data.frame(Level = 6.5))
