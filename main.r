# install.packages("keras")

# HandWritten digit recognition using Convolution Neural Networks
data <- read.csv("train.csv")

# Exploring the data
dim(data)
head(data[1:6])
unique(unlist(data[1]))
min(data[2:785])
max(data[2:785])
