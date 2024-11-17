resource_path <- function(filename) {
  script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
  base_dir <- normalizePath(file.path(script_dir, ".."))
  file.path(base_dir, "resources", filename)
}

# Data preprocessing
# Importing dataset
dataset = read.csv(resource_path('50_Startups.csv'))

# Encoding categorical data
dataset$State = factor(
  dataset$State,
  levels = c('New York', 'California', 'Florida'),
  labels = c(1, 2, 3)
)


#Splitting the dataset into test and training
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Linear Regression
regressor = lm(formula = Profit ~ ., data = training_set)

#Summary of statistical significance
summary(regressor)

#Predicting the test set results
y_pred = predict(regressor, newdata = test_set)
