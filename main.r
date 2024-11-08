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
