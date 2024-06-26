# Load the dataset from myData.csv
sample_data <- read.csv("C:\\Users\\juver\\Downloads\\myData.csv")

# Initialize vectors to store the means, standard deviations, and standard errors
mean_x_values <- numeric(5)
mean_y_values <- numeric(5)
sd_x_values <- numeric(5)
sd_y_values <- numeric(5)
se_x_values <- numeric(5)
se_y_values <- numeric(5)
upper_range_x<- numeric(5)
upper_range_y<- numeric(5)
lower_range_x<- numeric(5)
lower_range_y<- numeric(5)
j <- 1

# Loop for 5 iterations
for (i in seq(20, 100, by = 20)) {
  # Initialize j to 1 at the start of each iteration
  
  # Create a matrix to store the current sample for columns "x" and "y"
  current_sample <- sample_data[j:i, c("x", "y")]
  print(current_sample)
  # Calculate the mean of the current sample for column "x"
  mean_x_values[i/20] <- mean(current_sample$x)
  
  # Calculate the mean of the current sample for column "y"
  mean_y_values[i/20] <- mean(current_sample$y)
  
  # Calculate the standard deviation of the current sample for column "x"
  sd_x_values[i/20] <- sd(current_sample$x)
  
  # Calculate the standard deviation of the current sample for column "y"
  sd_y_values[i/20] <- sd(current_sample$y)
  
  # Calculate the standard error of the current sample for column "x"
  se_x_values[i/20] <- sd_x_values[i/20] / sqrt(20)
  
  # Calculate the standard error of the current sample for column "y"
  se_y_values[i/20] <- sd_y_values[i/20] / sqrt(20)
  
  
  
  # Calculate the upper and lower range for column "x"
  upper_range_x[i/20] <- mean_x_values[i/20] + 2.576 * se_x_values[i/20]
  lower_range_x[i/20] <- mean_x_values[i/20] - 2.576 * se_x_values[i/20]
  
  # Calculate the upper and lower range for column "y"
  upper_range_y[i/20] <- mean_y_values[i/20] + 2.576 * se_y_values[i/20]
  lower_range_y[i/20] <- mean_y_values[i/20] - 2.576 * se_y_values[i/20]
  
  # Increment j by 20 for the next iteration
  j <- j + 20
}

# Print the results
cat("Means for x:", mean_x_values, "\n")
cat("Means for y:", mean_y_values, "\n")
cat("Standard Deviations for x:", sd_x_values, "\n")
cat("Standard Deviations for y:", sd_y_values, "\n")
cat("Standard Errors for x:", se_x_values, "\n")
cat("Standard Errors for y:", se_y_values, "\n")
# Print the upper and lower ranges for both "x" and "y"
cat("Upper Range for x:", upper_range_x, "\n")
cat("Lower Range for x:", lower_range_x, "\n")
cat("Upper Range for y:", upper_range_y, "\n")
cat("Lower Range for y:", lower_range_y, "\n")


#Dataframe for table_20
table_20_x <- data.frame(Sample_Size = 20,
                       Mean_x = mean_x_values,
                       StandardDeviation_x = sd_x_values,
                       StandardError_x = se_x_values,
                       Lower_Range_x = lower_range_x,
                       Upper_Range_x = upper_range_x
                       
                       
)
print(table_20_x)
table_20_y <- data.frame(Sample_Size = 20,
                         Mean_y = mean_y_values,
                         StandardDeviation_y = sd_y_values,
                         StandardError_y = se_y_values,
                         Lower_Range_y = lower_range_y,
                         Upper_Range_y = upper_range_y
                         
)
print(table_20_y)


# for sample 75


mean_x_values <- numeric(5)
mean_y_values <- numeric(5)
sd_x_values <- numeric(5)
sd_y_values <- numeric(5)
se_x_values <- numeric(5)
se_y_values <- numeric(5)
upper_range_x<- numeric(5)
upper_range_y<- numeric(5)
lower_range_x<- numeric(5)
lower_range_y<- numeric(5)
j <- 1

# Loop for 5 iterations
for (i in seq(75, 375, by = 75)) {
  # Initialize j to 1 at the start of each iteration
  
  # Create a matrix to store the current sample for columns "x" and "y"
  current_sample <- sample_data[j:i, c("x", "y")]
  print(current_sample)
  # Calculate the mean of the current sample for column "x"
  mean_x_values[i/75] <- mean(current_sample$x)
  
  # Calculate the mean of the current sample for column "y"
  mean_y_values[i/75] <- mean(current_sample$y)
  
  # Calculate the standard deviation of the current sample for column "x"
  sd_x_values[i/75] <- sd(current_sample$x)
  
  # Calculate the standard deviation of the current sample for column "y"
  sd_y_values[i/75] <- sd(current_sample$y)
  
  # Calculate the standard error of the current sample for column "x"
  se_x_values[i/75] <- sd_x_values[i/75] / sqrt(75)
  
  # Calculate the standard error of the current sample for column "y"
  se_y_values[i/75] <- sd_y_values[i/75] / sqrt(75)
  
  
  
  # Calculate the upper and lower range for column "x"
  upper_range_x[i/75] <- mean_x_values[i/75] + 2.576 * se_x_values[i/75]
  lower_range_x[i/75] <- mean_x_values[i/75] - 2.576 * se_x_values[i/75]
  
  # Calculate the upper and lower range for column "y"
  upper_range_y[i/75] <- mean_y_values[i/75] + 2.576 * se_y_values[i/75]
  lower_range_y[i/75] <- mean_y_values[i/75] - 2.576 * se_y_values[i/75]
  
  # Increment j by 75 for the next iteration
  j <- j + 75
}

# Print the results
cat("Means for x:", mean_x_values, "\n")
cat("Means for y:", mean_y_values, "\n")
cat("Standard Deviations for x:", sd_x_values, "\n")
cat("Standard Deviations for y:", sd_y_values, "\n")
cat("Standard Errors for x:", se_x_values, "\n")
cat("Standard Errors for y:", se_y_values, "\n")
# Print the upper and lower ranges for both "x" and "y"
cat("Upper Range for x:", upper_range_x, "\n")
cat("Lower Range for x:", lower_range_x, "\n")
cat("Upper Range for y:", upper_range_y, "\n")
cat("Lower Range for y:", lower_range_y, "\n")


#Dataframe such that its neat to visualize


table_75_x <- data.frame(Sample_Size = 75,
                         Mean_x = mean_x_values,
                         StandardDeviation_x = sd_x_values,
                         StandardError_x = se_x_values,
                         Lower_Range_x = lower_range_x,
                         Upper_Range_x = upper_range_x
                         
                         
)
print(table_75_x)
table_75_y <- data.frame(Sample_Size = 75,
                         Mean_y = mean_y_values,
                         StandardDeviation_y = sd_y_values,
                         StandardError_y = se_y_values,
                         Lower_Range_y = lower_range_y,
                         Upper_Range_y = upper_range_y
                         
)
print(table_75_y)