#Have necessary libraries
library(tidyverse)
library(readxl)
library(ggplot2)
library(arules)
library(arulesViz)
library(data.table)
library(plyr)
library(dplyr)
library(knitr)
library(RColorBrewer)

#Reading csv file of BreadBasket_DMS which has transaction and its respective items into R dataframe
arules_data = read_csv("C:\\Users\\juver\\Downloads\\BreadBasket_DMS.csv")

#Descriptive statistics analysis of our data BreadBasket_DMS.csv

summary(arules_data$Transaction)
mean(arules_data$Transaction)
median(arules_data$Transaction)
mode=function(a){
  uni=unique(a)
  tab= tabulate(match(a,uni))
  uni[tab==max(tab)]
}
mode(arules_data$Transaction)
quantile(arules_data$Transaction)
var(arules_data$Transaction)
sd(arules_data$Transaction)
unique(arules_data$Item)


nrow(arules_data)
#complete.cases(data) will return a logical vector indicating which rows 
#have no missing values. Then we can have only rows whose data is'nt missing.
arules_data_preprocess <- arules_data[complete.cases(arules_data), ]
#formatting date & time
arules_data_preprocess$Time<- format(arules_data_preprocess$Time,"%H:%M:%S")
#converting into numeric
arules_data_preprocess$Transaction <- as.numeric(as.character(arules_data_preprocess$Transaction))
arules_data_preprocess$Item = as.factor(arules_data_preprocess$Item)

glimpse(arules_data_preprocess)

#Merging the similar transaction items

transactionData <- ddply(arules_data_preprocess,c("Transaction","Date"),
                         function(df1)paste(df1$Item, collapse = ","))
transactionData

#set column Date of dataframe transactionData
transactionData$Date <- NULL
#Rename column to items
colnames(transactionData) <- c("Transaction","items")
#Show Dataframe transactionData
transactionData

updated_data = subset(transactionData, items!="NONE")
updated_data
newdata=updated_data %>% 
  group_by(Transaction) %>%summarise_all(funs(trimws(paste(., collapse = ','))))



data_split = strsplit(as.character(newdata$items),',',fixed=T) # split by comma    
transac = as(data_split, "transactions")
inspect(transac)

#creating a new dataframe basket_items_transaction.csv which has unique transaction with items

#Make sure this file is created at your location
write.csv(transactionData,"C:\\Users\\juver\\OneDrive\\Desktop\\gnments\\Intro. to DS 2\\basket_items_transaction.csv", quote = FALSE, row.names = FALSE)
basket_items_transaction <- read.transactions("C:\\Users\\juver\\OneDrive\\Desktop\\gnments\\Intro. to DS 2\\basket_items_transaction.csv", format = 'basket', sep=',')
basket_items_transaction

#basket analysis

summary(basket_items_transaction)


#mar is a numeric vector of size 4, which sets the margin sizes in the 
#following order: bottom, left, top, and right. 
par(mar=c(0,10,1,.1))
itemFrequencyPlot(basket_items_transaction,topN=20,type="absolute",col=brewer.pal(8,'Pastel2'), main="Absolute Item Frequency Plot")

# Now we can find the association rules in the data set
#Generating association rules with minimum support of 0.001 and minimum confidence of 0.001
rules = apriori(basket_items_transaction,
                   parameter = list(supp = 0.001, conf = 0.001))
inspect(sort(rules,decreasing = TRUE,by="lift"))

#removing coffee from right hand side
rules_no_coffee = subset(rules, subset = !(rhs %in% "Coffee")) 

#rules_no_coffee <- subset(rules, subset = !(lhs %in% "NONE") | !(rhs %in% "NONE"))
rules_no_coffee_no_none <- rules[!(lhs(rules) %in% "NONE" | rhs(rules) %in% "NONE")]

#sorting the rules by using lift metric
inspect(sort(rules_no_coffee_no_none,decreasing = TRUE,by="lift"))

#'''

#Commenting such that this combination gives coffee on the rhs as coffee frequency is high
#We can not infer much from this kind of setting

# Now we can find the association rules in the data set
#Generating association rules with minimum support of 0.01 and minimum confidence of 0.5
#rules = apriori(basket_items_transaction,
#                parameter = list(supp = 0.01, conf = 0.5))
#inspect(sort(rules,decreasing = TRUE,by="lift"))

#removing coffee from right hand side
#rules_no_coffee = subset(rules, subset = !(rhs %in% "Coffee")) 

#rules_no_coffee <- subset(rules, subset = !(lhs %in% "NONE") | !(rhs %in% "NONE"))
#rules_no_coffee_no_none <- rules[!(lhs(rules) %in% "NONE" | rhs(rules) %in% "NONE")]

#sorting the rules by using lift metric
#inspect(sort(rules_no_coffee_no_none,decreasing = TRUE,by="lift"))

#'''