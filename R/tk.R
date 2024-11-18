resource_path <- function(filename) {
  script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
  base_dir <- normalizePath(file.path(script_dir, ".."))
  file.path(base_dir, "resources", filename)
}

# Data preprocessing
# Importing dataset
dataset = read.csv(resource_path('Position_Salaries.csv'))
dataset = dataset[2:3]

# SVR Regression
library(e1071)
regressor = svm(Salary ~ ., data = dataset, type = 'eps-regression')

# Visualizing Poly and Linear Regression
library(ggplot2)

ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
  geom_line(aes(
    x = dataset$Level,
    y = predict(regressor, newdata = dataset),
    
  ), colour = 'green') +
  ggtitle('Truth or Bluff (SVR)') +
  ylab('Salary') + xlab('Level')

# Predicting Values
# SVR
value = predict(regressor, newdata = data.frame(Level = 6.5))
