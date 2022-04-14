library(tidyverse)
install.packages('caret')
install.packages('nnet')
library(caret)
library(nnet)


games = read.csv('Desktop/Stat 4970W/Data Fest/games.csv')
wellness = read.csv('Desktop/Stat 4970W/Data Fest/wellness.csv')
rpe = read.csv('Desktop/Stat 4970W/Data Fest/rpe.csv')


#Making dates
games$Date = as.Date(games$Date)
wellness$Date = as.Date(wellness$Date)


#Subsets of Win/Loss games
games$Outcome = as.factor(games$Outcome)
win = games[games$Outcome == 'W',]
loss = games[games$Outcome == 'L',]

par(mfrow = c(1,1))
plot(games$Outcome, main = 'Outcome of Games',
     xlab = 'Outcome', ylab = 'Frequency',
     col = c('indian red','cornflower blue'))

#Cleaning data
wellness$TrainingReadiness =
  as.numeric(sub("%", "", wellness$TrainingReadiness,fixed=TRUE))/100


#Data of day-of losses — cleaning for unique dates only
l2 = wellness[wellness$Date == '2018-01-28',]
l3 = wellness[wellness$Date == '2018-04-14',]
l4 = wellness[wellness$Date == '2018-05-13',]
l5 = wellness[wellness$Date == '2018-05-12',]
l6 = wellness[wellness$Date == '2018-04-22',]


loss.bind = rbind(l2, l3, l4, l5, l6)
mean(loss.bind$TrainingReadiness)

#Day-of wins —cleaning for unique dates only
w1 = wellness[wellness$Date == '2017-11-30',]
w2 = wellness[wellness$Date == '2017-12-01',]
w3 = wellness[wellness$Date == '2018-01-26',]
w4 = wellness[wellness$Date == '2018-06-08',]
w6 = wellness[wellness$Date == '2018-04-13',]
w13 = wellness[wellness$Date == '2018-06-10',]

win.bind = rbind(w1, w2, w3, w4, w6, w13)

#Cleaning data
loss.bind = loss.bind[, -c(2, 7:8, 17:18)]
loss.bind[, 9:13] = lapply(loss.bind[,9:13], function(x) as.factor(as.character(x)))
loss.bind[, 2:8] = lapply(loss.bind[,2:8], function(x) as.numeric(as.character(x)))

win.bind = win.bind[, -c(2, 7:8, 17:18)]
win.bind[, 9:13] = lapply(win.bind[,9:13], function(x) as.factor(as.character(x)))
win.bind[, 2:7] = lapply(win.bind[,2:7], function(x) as.numeric(as.character(x)))

#Plotting the averages
ab = ggplot(data = win.bind, 
       mapping = aes(x = Date, y = TrainingReadiness)) + 
  stat_summary(fun = "mean", geom = "line", color = "cornflower blue",
               size = 1.5) +
  theme_bw() + ylim(0,1) + ggtitle('Training Readiness on the Day of Winning Games') + labs(x = 'Date of Game',
                                   y = 'Training Rediness %' )

bc = ggplot(data = loss.bind, 
       mapping = aes(x = Date, y = TrainingReadiness)) + 
  stat_summary(fun = "mean", geom = "line", color = "indian red",
               size = 1.5) +
  theme_bw() + ylim(0,1) + ggtitle('Training Readiness on Day of Losing Games') +
  labs(x = 'Date of Game', y = 'Training Readiness %') 
  
#Visualization
ggarrange(ab, bc)
mean(loss.bind$TrainingReadiness)
mean(win.bind$TrainingReadiness)



#######################
#Training Readiness Analysis
#######################



#Clearing out unneeded variables
well = wellness[, -c(1:2, 7:8, 17:18)]
well[, 8:12] = lapply(well[,8:12], function(x) as.factor(as.character(x)))
well[, 2:7] = lapply(well[,2:7], function(x) as.numeric(as.character(x)))



#Making cutoff for readiness at 0.95
well$ready[well$TrainingReadiness >= 0.90] = 'Ready'
well$ready[well$TrainingReadiness < 0.90 ] = 'Not Ready'
well$ready = as.factor(well$ready)

#Taking out original variable
#well.real = well[, -13] #Come back to this
well = na.omit(well)

#Training and Test data
set.seed(1)
train = sample(1:nrow(well), 0.9*nrow(well))
well.train = well[train,]
well.test = well[-train,]

#Visualizing the responses
plot(well$ready, main = 'Breakdown by Classification',
     xlab = 'Ready for Game?', 
     ylab = 'Frequency', col = c('indian red', 'cornflower blue'))


#Best subsets
library(ISLR)
library(leaps)
regfit.full = regsubsets(ready ~. - TrainingReadiness - MonitoringScore, well)
reg.summary = summary(regfit.full)
reg.summary 

par(mfrow = c(2,2))
#RSS
plot(reg.summary$rss, xlab = 'Number of Variables', ylab = 'RSS',
     type = 'b')

#ADJ R^2
plot(reg.summary$adjr2, xlab = 'Number of Variables', ylab = 'Adjusted R^2',
     type = 'h')

which.max(reg.summary$adjr2)

points(which.max(reg.summary$adjr2), reg.summary$adjr2[which.max(reg.summary$adjr2)],
       col = 'red', cex = 2, pch = 20)

#CP
plot(reg.summary$cp, xlab = 'Number of Variables', ylab = 'Cp',
     type = 'b')
which.min(reg.summary$cp)
points(which.min(reg.summary$cp), 
       reg.summary$cp[which.min(reg.summary$cp)],
       col = 'red', cex = 2, pch = 20)

#BIC
plot(reg.summary$bic, xlab = 'Number of Variables', ylab = 'bic',
     type = 'b')
which.min(reg.summary$bic)
points(which.min(reg.summary$bic), 
       reg.summary$bic[which.min(reg.summary$bic)],
       col = 'red', cex = 2, pch = 20)
#Plots show that 8 variables is the best

#Logistic Regression
glm.fit = glm(ready ~ . - TrainingReadiness - MonitoringScore, well, family = binomial)

glm.probs = predict(glm.fit, well.test, type = 'response')
glm.pred = rep('Not Ready', nrow(well.test))
glm.pred[glm.probs > 0.5] = 'Ready'

table(well.test$ready, glm.pred)
mean(well.test$ready != glm.pred)





#Random Forest
#Looping the best number for shrinkage
set.seed(1)
m = c(1:9)
error.rf2 = rep(0, length(m))

for (i in 1:13) {
  
  rf.well = randomForest(ready~. - TrainingReadiness- MonitoringScore, data = well.train,
                         mtry = i)
  
  yhat.rf = predict(rf.well, newdata = well.test)
  rf.error[yhat.rf >= 0.5] = 'Ready'
  rf.error[yhat.rf < 0.5] = 'Not Ready'
  error.rf2 = mean((rf.error != well.test$ready)^2)
}

error2
par(mfrow = c(1,1))


#Using best number of variables as 8
set.seed(1)
rf.well = randomForest((as.numeric(ready)-1)~. - TrainingReadiness - MonitoringScore, data = well.train,
                        mtry = 8, importance = TRUE, proximity = TRUE)
rf.well

rf.error = rep(0, length(well.test))
yhat.rf = predict(rf.well, newdata = well.test)
rf.error[yhat.rf >= 0.5] = 'Ready'
rf.error[yhat.rf < 0.5] = 'Not Ready'
mean((rf.error == well.test$ready)^2)
table(rf.error, well.test$ready)

importance(rf.well)
varImpPlot(rf.well)

#Decision Tree
library(tree)
par(mfrow = c(1,1))
well.tree = tree(ready~. - TrainingReadiness, data = well.train)
summary(well.tree)
well.tree
plot(well.tree)
text(well.tree)

well.pred = predict(well.tree, well.test, type = 'class')
table(well.pred, well.test$ready)
mean(well.pred == well.test$ready)

set.seed(1)
well.cv = cv.tree(well.tree, FUN = prune.misclass)
well.cv
plot(well.cv$size, well.cv$dev, type = 'b')







  
  
  