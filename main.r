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
print(model_lr)

predicution_lr <- predict(model_lr, data_test, type = "class")
predicution_lr[1:5]
data_test$label[1:5]

cm_lr <- table(data_test$label, predicution_lr)
cm_lr

accuracy_lr <- mean(predicution_lr == data_test$label)
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
