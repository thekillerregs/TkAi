resource_path <- function(filename) {
  script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
  base_dir <- normalizePath(file.path(script_dir, ".."))
  file.path(base_dir, "resources", filename)
}

# Data preprocessing
# Importing dataset
dataset = read.csv(resource_path('Social_Network_Ads.csv'))

# Encoding the target feature as factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Splitting dataset
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])

# SVM Regression
library(e1071)
classifier = svm(
  formula = Purchased ~ .,
  data = training_set,
  type = 'C-classification',
  kernel = 'radial'
)

y_pred = predict(classifier, newdata = test_set)

# Creating Confusion Matrix
cm = table(test_set[, 3], y_pred)

# K-Fold Cross Validation
library(caret)
classifier = train(form=Purchased~., data=training_set, method='svmRadial')
classifier$Besttune