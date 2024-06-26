
library(readxl)

#Read the sales data excel sheet present at my local computer directory
temp_sales_data <- read_excel("C:\\Users\\juver\\Downloads\\Sales_Data.xlsx")
sales_data <- temp_sales_data$`No. of Sales`

#Visualizing the data using histogram plot
hist(temp_sales_data$`No. of Sales`,
     col = "green",          # Fill color
     border = "black",      # Border color
     main = "Sales Distribution",  # Main title
     xlab = "Number of Sales",     # X-axis label
     ylab = "Frequency"            # Y-axis label
)

#calculating mean for the sales data
mean_x <- mean(sales_data)
mean_x
#calculating variance for the probability
var_x <- var(sales_data)
var_x
#Given people, n=1000
N <- 1000
# probability using binomial distribution, here mean = n*p, p is probability => p=mean/n
probability <- mean_x / N
probability

#1.i

#probability  between 6 and 8 purchases in a day.
prob_between_6_and_8_binomial <- sum(dbinom(6:8, N, probability))
prob_between_6_and_8_binomial

prob_between_6_and_8_poisson <- sum(dpois(6:8, mean_x))
prob_between_6_and_8_poisson



#1.ii
#the probability that there will be exactly 7 purchases in a day
prob_exactly_7_binomial <- dbinom(7, N, probability)
prob_exactly_7_binomial

prob_exactly_7_poisson <- dpois(7, mean_x)
prob_exactly_7_poisson

#1.iii.
#probability that among 1000 customers there will be at most 5 purchases in a day
prob_atmost_5 <- pbinom(5, N, probability)
prob_atmost_5

prob_atmost_5_poisson <- ppois(5, mean_x)
prob_atmost_5_poisson


