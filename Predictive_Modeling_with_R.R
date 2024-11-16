# Load necessary libraries
install.packages("caret")
install.packages("ggplot2")
install.packages("lattice")
install.packages("tree")
install.packages("randomForest")
install.packages("e1071")
install.packages("cluster")
install.packages("factoextra")
install.packages("neuralnet")
library(MASS)
library(caret)
library(tree)
library(randomForest)
library(e1071)
library(cluster)
library(factoextra)
library(neuralnet)

#1a. Load the dataset insurance.csv into memory.
#set working directory
setwd("C:/Users/Utkarsh")
insurance <- read.csv("C:/Users/Utkarsh/Downloads/insurance.csv")
colnames(insurance)

# Step 1b: Transform the charges variable by taking the log
insurance$charges <- log(insurance$charges)

# Step 1c: Create dummy variables for categorical data
dummy <- model.matrix(~. -charges, data=insurance)  # Create dummy variables without charges
dummy <- dummy[, -1]  # Remove the first column with ones
colnames(dummy)  # Check column names of the dummy data

# Step 1d: Create row indexes for training and testing sets (2/3 for training, 1/3 for testing)
set.seed(1)
train_size <- floor(2 * nrow(insurance) / 3)
train_idx <- sample(seq_len(nrow(insurance)), size = train_size)

# Step 1e: Create training and testing datasets from the log-transformed data
train_1e <- insurance[train_idx, ]
test_1e <- insurance[-train_idx, ]

# Step 1f: Create training and testing datasets from dummy variables data
train_1f <- dummy[train_idx, ]
test_1f <- dummy[-train_idx, ]

# Step 2: Fit a multiple linear regression model
lm.fit <- lm(charges ~ age + sex + bmi + children + smoker + region, data = train_1e)
summary(lm.fit)  # Print summary of the linear model

# Step 3: Perform backward stepwise selection using AIC
full <- lm(charges ~ age + sex + bmi + children + smoker + region, data = train_1e)
lm.bwd <- stepAIC(full, direction = "backward")

# Step 4: Perform LOOCV on the model
train_control <- trainControl(method = "LOOCV")
model_2e <- train(charges ~ age + sex + bmi + children + smoker + region, data = train_1e, trControl = train_control, method = "lm")
print(model_2e)

# Step 5: Perform 10-fold cross-validation on the model
train_control_2f <- trainControl(method = "CV", number = 10)
model_2f <- train(charges ~ age + sex + bmi + children + smoker + region, data = train_1e, trControl = train_control_2f, method = "lm")
print(model_2f)

# Calculate the mean squared error for model_2f
MSE_2f <- (0.4233641)^2
MSE_2f

# Step 6: Train another model on the test data and calculate MSE
train_control <- trainControl(method = "LOOCV")
model_2g <- train(charges ~ age + sex + bmi + children + smoker + region, data = test_1e, trControl = train_control, method = "lm")
print(model_2g)
MSE_2g <- (0.4845217)^2
MSE_2g

# Step 7: Build a regression tree model
insurance$sex <- as.factor(insurance$sex)
insurance$smoker <- as.factor(insurance$smoker)
insurance$region <- as.factor(insurance$region)
tree.insurance <- tree(charges ~ age + sex + bmi + children + smoker + region, data = insurance, subset = train_idx)
summary(tree.insurance)

# Cross-validate the tree model
cv.insurance <- cv.tree(tree.insurance)
plot(cv.insurance$size, cv.insurance$dev, type = 'b')

# Prune the tree to the best size
prune.insurance = prune.tree(tree.insurance, best=6)
plot(prune.insurance)
text(prune.insurance, pretty = 0)

# Calculate test error before and after pruning
yhat0 <- predict(tree.insurance, insurance[-train_idx, ])
insurance.test0 <- insurance[-train_idx, "charges"]
mean((yhat0 - insurance.test0)^2)  # MSE before pruning

yhat <- predict(prune.insurance, newdata = insurance[-train_idx, ])
insurance.test <- insurance[-train_idx, "charges"]
mean((yhat - insurance.test)^2)  # MSE after pruning

# Step 8: Build a random forest model
rf.insurance <- randomForest(charges ~ age + sex + bmi + children + smoker + region, data = insurance, subset = train_idx, importance = TRUE, na.action = na.exclude)
print(rf.insurance)

# Calculate test MSE for random forest
yhat.rf <- predict(rf.insurance, newdata = insurance[-train_idx, ])
insurance.test <- insurance[-train_idx, "charges"]
mean((yhat.rf - insurance.test)^2)

# Plot variable importance in the random forest model
varImpPlot(rf.insurance)

# Step 9: Build a support vector machine model
svm.fit <- svm(charges ~ age + sex + bmi + children + smoker + region, data = insurance[train_idx, ], kernel = "radial", gamma = 5, cost = 50)
summary(svm.fit)

# Perform a grid search for best SVM model parameters
tune.out <- tune(svm, charges ~ age + sex + bmi + children + smoker + region, data = insurance[train_idx, ], ranges = list(cost = c(1, 10, 50, 100), gamma = c(1, 3, 5), kernel = c("linear", "radial", "sigmoid")))
summary(tune.out)

# Make predictions with the best model and calculate MSE
pred <- predict(tune.out$best.model, newdata = insurance[-train_idx, ])
trueobservation <- insurance[-train_idx, "charges"]
MSE_5e <- mean((trueobservation - pred)^2)
MSE_5e

# Step 10: K-means cluster analysis
insurance <- insurance[, c(-2, -5, -6)]  # Remove categorical variables (sex, smoker, region)
colnames(insurance)

# Determine the optimal number of clusters using gap statistic
set.seed(101)
fviz_nbclust(insurance, FUNcluster = stats::kmeans, method = "gap_stat")

# Perform k-means clustering with 3 clusters
km.res <- kmeans(insurance, 3, nstart = 25)
print(km.res)

# Visualize clusters
fviz_cluster(km.res, data = insurance)

# Step 11: Build a neural network model
scaled.insurance <- scale(insurance[,-5])  # Standardize inputs
scaled.insurance <- as.data.frame(scaled.insurance)

# Split the data into training and test sets for the neural network
set.seed(101)
index <- sample(1:nrow(scaled.insurance), 0.80 * nrow(scaled.insurance))
train <- scaled.insurance[index, ]
test <- scaled.insurance[-index, ]

# Fit the neural network model
nn.model <- neuralnet(charges ~ age + bmi + children, data = train, hidden = 1)
summary(nn.model)

# Plot the neural network
plot(nn.model)

# Make predictions with the neural network and calculate MSE
predict.nn <- compute(nn.model, test[, c("age", "bmi", "children")])
observ.test <- test$charges
mean((observ.test - predict.nn$net.result)^2)

# Step 12: Reverse log transformation in pruned tree model and plot
copy_of_my_pruned_tree <- prune.tree(tree.insurance, best = 6)
copy_of_my_pruned_tree$frame$yval <- exp(copy_of_my_pruned_tree$frame$yval)  # Reverse log transformation
plot(copy_of_my_pruned_tree)
text(copy_of_my_pruned_tree, pretty = 0)

