resource_path <- function(filename) {
  script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
  base_dir <- normalizePath(file.path(script_dir, ".."))
  file.path(base_dir, "resources", filename)
}

# Data preprocessing
# Importing dataset
dataset = read.csv(resource_path('Position_Salaries.csv'))
dataset = dataset[2:3]

# Linear Regression
lin_reg = lm(formula = Salary ~ ., data = dataset)

dataset$Level2 = dataset$Level ^ 2
dataset$Level3 = dataset$Level ^ 3
dataset$Level4 = dataset$Level ^ 4

# Polynomial Regression
poly_reg = lm(formula = Salary ~ ., data = dataset)

# Visualizing Poly and Linear Regression
library(ggplot2)

ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newData = dataset), ), colour = 'blue') +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset), ), colour = 'green') +
  ggtitle('Truth or Bluff (Linear Regression)') +
  ylab('Salary') + xlab('Level')

# Predicting Values

# Linear
len_value = predict(lin_reg, newdata = data.frame(Level = 6.5))

# Poly
poly_value = predict(poly_reg,
                     newdata = data.frame(
                       Level = 6.5,
                       Level2 = 6.5 ^ 2,
                       Level3 = 6.5 ^ 3,
                       Level4 = 6.5 ^ 4
                     ))

