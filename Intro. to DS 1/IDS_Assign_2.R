library(readxl)

#Read the restaurant data excel sheet present at my local computer directory
Data <-read_excel("C:\\Users\\juver\\Downloads\\restaurant.xlsx")

##Visualizing the data using histogram plot and we can notice in histogram it is like bell-shaped and we use normal distribution
hist(Data$`Number of Customers`,
     col = "red",                # Fill color (red)
     border = "yellow",           # Border color 
     main = "Restaurant Distribution",  # Main title
     xlab = "X",   # X-axis label
     ylab = "Y"              # Y-axis label
)

#mean
mean_x <- mean(Data$`Number of Customers`, na.rm= TRUE)
mean_x

#variance
var_x <- var(Data$`Number of Customers`, na.rm= TRUE)
var_x

#probability distribution
probabilty <- mean_x/ nrow(Data)
probabilty

#standard deviation
std <- sd(Data$`Number of Customers`, na.rm= TRUE) 
std
#85% food wont run out
limit <- 0.85

nor_dist <- qnorm(limit, mean_x, std)
nor_dist