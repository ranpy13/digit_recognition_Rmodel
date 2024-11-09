# install.packages("keras")

# HandWritten digit recognition using Convolution Neural Networks
data <- read.csv("train.csv")

# Exploring the data
dim(data)
head(data[1:6])
unique(unlist(data[1]))
min(data[2:785])
max(data[2:785])

# Look at two samples, say the 4th and 7th images
sample_4 <- matrix(as.numeric(data[4, -1]), nrow = 28, byrow = TRUE)
image(sample_4, col = grey.colors(255))

sample_7 <- matrix(as.numeric(data[7, -1]), nrow = 28, byrow = TRUE)
image(sample_7, col = grey.colors(255))


# Rotating the matrix by reversing elements in each column
rotate <- function(x) t(apply(x, 2, rev))

# Look at the rotated images
image(rotate(sample_4), col = grey.colors(255))
image(rotate(sample_7), col = grey.colors(255))

# Transform target variables "label" from integer to factor
# in order to perform classifcations
is.factor(data$label)
data$label <- as.factor(data$label)

# Check class blanaced or unbalanced
summary(data$label)

proportion <- prop.table(table(data$label)) * 100
cbind(count = table(data$label), proportion = proportion)


# Exploratory analysis on features

# Central 2*2 block of an image
central_block <- c("pixel376", "pixel377", "pixel404", "pixel405")
par(mfrow = c(2, 2))
for (i in 1:9) {
    hist(c(as.matrix(data[data$label] == i, central_block)),
        main = sprintf("Histogram for digit %d", i), xlab = "Pixel Value"
    )
}


# using the caret package, test train split
if (!require("caret")) {
    install.packages("caret")
}
library(caret)

set.seed(42) # cause of, course
train_perc <- 0.75
train_index <- createDataPartition(data$label, p = train_perc, list = FALSE)
data_train <- data[train_index, ]
data_test <- data[-tarin_index, ]

library(nnet)
# Multinomial logistic regression
model_lr <- multinom(lable ~ .,
    data = data_train,
    MaxNWts = 10000, decay = 5e-3, maxit = 100
)
# print(model_lr)

prediction_lr <- predict(model_lr, data_test, type = "class")
prediction_lr[1:5]
data_test$label[1:5]

cm_lr <- table(data_test$label, prediction_lr)
cm_lr
accuracy_lr <- mean(prediction_lr == data_test$label)
accuracy_lr

# Single layer neural networks
model_nn <- nnet(label ~ .,
    data = data_tarin, size = 50, maxit = 300, MaxNWts = 100000, decay = 1e-4
)

prediction_nn <- predict(model_nn, data_test, type = "class")
cm_nn <- table(data$label, prediction_nn)
cm_nn
accuracy_nn <- mean(prediction_nn == data_test$label)
accuracy_nn


# Multiple hidden layer Neural Networks

# installing mxnet
cran <- getOption("repos")
cran["dmlc"] <- "https://s3-us-west-2.amazonaws.com/apache-mxnet/R/CRAN/"
options(repos = cran)
if (!require("mxnet")) {
    install.packages("mznet")
}

require(mxnet)

data <- read.csv("train.csv")
train_prec <- 0.75
train_index <- createDataPartition(data$label, p = train_perc, list = FALSE)
data_train <- data[train_index, ]
data_test <- data[-train_index, ]

data_train <- data.matrix(data_train)
data_train.x <- data_train[, -1]
data_train.x <- t(data_train.x / 255)
data_train.y <- data_train[, -1]

data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name = "fc1", num_hidden = 128)
act1 <- mx.symbol.Activation(fc1, name = "relu1", act_type = "relu")
fc2 <- mx.symbol.FullyConnected(act1, name = "fc2", num_hidden = 64)
act2 <- mx.symbol.Activation(fc2, name = "relu2", act_type = "relu")
fc3 <- mx.symbol.FullyConnected(act2, name = "fc3", num_hidden = 10)
softmax <- mx.symbol.SoftmaxOuput(fc3, name = "sm")

devices <- mx.cpu()

mx.set.seed(42)
model_dnn <- mx.model.FeedForward.create(softmax,
    X = data_tarin.x,
    y = data_train.y,
    ctx = devices,
    num.round = 30,
    array.batch.size = 100,
    learning.rate = 0.01,
    momentum = 0.9,
    eval.metric = mx.metric.accuracy,
    initializer = mx.init.uniform(0.1),
    epoch.end.callback = mx.callback.log.train.metric(100)
)

data_test.x <- data_test[, -1]
data_test.x <- t(data_test.x / 255)
prob_dnn <- predict(model_dnn, data_test.x)
prediction_dnn <- max.col(t(prob_dnn))
cm_dnn <- table(data$label, prediction_dnn)
cm_dnn

accuracy_dnn <- mean(prediction_dnn == data_test$label)
accuracy_dnn


# Convolution Neural Networks

# first convolution
conv1 <- mx.symbol.Convolution(data = data, kernel = c(5, 5), num_filter = 20)
act1 <- mx.symbol.Activation(data = conv1, act_type = "relu")
pool1 <- mx.symbol.Pooling(
    data = act1, pool_type = "max",
    kernel = c(2, 2), stride = c(2, 2)
)

# second convolution
conv2 <- mx.symbol.Convolution(data = pool1, kernel = c(5, 5), num_filter = 20)
act2 <- mx.symbol.Activation(data = conv1, act_type = "relu")
pool2 <- mx.symbol.Pooling(
    data = act2, pool_type = "max",
    kernel = c(2, 2), stride = c(2, 2)
)


# first fully connected layer
flatten <- mx.symbol.Flatten(data = pool2)
fc1 <- mx.symbol.FullyConnected(data = faltten, num_hidden = 500)
act3 <- mx.symbol.Activation(data = fc1, act_type = "relu")

# second fully connected layer
fc2 <- mx.symbol.FullyConnected(data = act3, num_hidden = 10)

# softmax output
softmax <- mx.symbol.SoftmaxOuput(data = fc2, name = "sm")

mx.set.seed(42)
train.array <- data_tarin.x
dim(train.array) <- c(28, 28, 1, ncol(data_train.x))

model_cnn <- mx.model.FeedForward.create(softmax,
    X = train.array,
    y = data_train.y, ctx = devices, num.round = 30,
    momentum = 0.9, wd = 0.00001, learning.rate = 0.05,
    eval.metric = mx.metric.accuracy,
    epoch.end.callback = mx.callback.log.train.metric(100)
)
