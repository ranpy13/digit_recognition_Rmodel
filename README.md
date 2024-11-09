# Handwritten Digit Recognition using Convolutional Neural Networks

This project implements a handwritten digit recognition system using various machine learning techniques, including logistic regression, single-layer neural networks, multi-layer neural networks, and convolutional neural networks (CNNs). The dataset used is the popular MNIST dataset, which consists of 28x28 pixel images of handwritten digits (0-9).

## Table of Contents

- [Introduction](#introduction)
- [Data Exploration](#data-exploration)
- [Modeling Techniques](#modeling-techniques)
  - [Logistic Regression](#logistic-regression)
  - [Single-layer Neural Networks](#single-layer-neural-networks)
  - [Multi-layer Neural Networks](#multi-layer-neural-networks)
  - [Convolutional Neural Networks](#convolutional-neural-networks)
- [Results](#results)
- [Visualizations](#visualizations)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction

The goal of this project is to classify handwritten digits using deep learning techniques. The project explores different models and evaluates their performance on the MNIST dataset.

## Data Exploration

The dataset is loaded from a CSV file, and the following steps are performed:

- Dimensions and structure of the data are examined.
- Unique labels and pixel value ranges are analyzed.
- Sample images are visualized.
- The target variable is transformed from integer to factor for classification.

```r
data <- read.csv("train.csv")
dim(data)
head(data[1:6])
summary(data$label)
```

## Modeling Techniques

### Logistic Regression

A multinomial logistic regression model is trained and evaluated.

```r
library(nnet)
model_lr <- multinom(label ~ ., data=data_train)
prediction_lr <- predict(model_lr, data_test, type = "class")
```

### Single-layer Neural Networks

A single-layer neural network is implemented using the `nnet` package.

```r
model_nn <- nnet(label ~ ., data=data_train, size=50)
prediction_nn <- predict(model_nn, data_test, type = "class")
```

### Multi-layer Neural Networks

A multi-layer neural network is built using the `mxnet` package.

```r
require(mxnet)
model_dnn <- mx.model.FeedForward.create(softmax, X=data_train.x, y=data_train.y)
```

### Convolutional Neural Networks

A convolutional neural network is constructed to improve classification accuracy.

```r
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20)
model_cnn <- mx.model.FeedForward.create(softmax, X=train.array, y=data_train.y)
```

## Results

The performance of each model is evaluated using confusion matrices and accuracy metrics.

```r
cm_lr <- table(data_test$label, prediction_lr)
accuracy_lr <- mean(prediction_lr == data_test$label)
```

## Visualizations

Visualizations include:

- Sample images from the dataset.
- Activation maps from convolutional layers.
- Learning curves for model training.

```r
par(mfrow=c(4,4))
for (i in 1:16) {
  outputData <- as.array(executor$ref.outputs$activation15_output)[,,i,1]
  image(outputData, xaxt='n', yaxt='n', col=grey.colors(255))
}
```

## Installation

To run this project, you need to have R and the following packages installed:

- `nnet`
- `caret`
- `mxnet`

You can install the required packages using:

```r
install.packages(c("nnet", "caret"))
```

For `mxnet`, follow the installation instructions on the [MXNet R package page](https://mxnet.apache.org/get_started/).

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/ranpy13/handwritten-digit-recognition.git
   cd handwritten-digit-recognition
   ```

2. Load the dataset and run the R script:
   ```r
   source("main.R")
   ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to modify the content as needed to better fit your project specifics!
