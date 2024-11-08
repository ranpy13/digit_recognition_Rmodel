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
