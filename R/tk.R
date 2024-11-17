resource_path <- function(filename) {
  script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
  base_dir <- normalizePath(file.path(script_dir, ".."))
  file.path(base_dir, "resources", filename)
}

# Data preprocessing
# Importing dataset
dataset = read.csv(resource_path('Salary_Data.csv'))

#Splitting the dataset into test and training
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)

training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Linear Regression
regressor=lm(formula = Salary ~ YearsExperience,
              data = training_set)

#Summary of statistical significance
summary(regressor)

#Predicting the test set results
y_pred = predict(regressor, newdata = test_set)
library(ggplot2)
ggplot() +
  geom_point(aes(x=training_set$YearsExperience, y=training_set$Salary),
             colour = 'red') +
  geom_line(aes(x=training_set$YearsExperience, y= predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Training Set)') +
  xlab('Years of Experience') +
  ylab('Salary')

ggplot() +
  geom_point(aes(x=test_set$YearsExperience, y=test_set$Salary),
             colour = 'red') +
  geom_line(aes(x=training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Test Set)') +
  xlab('Years of Experience') +
  ylab('Salary')

             