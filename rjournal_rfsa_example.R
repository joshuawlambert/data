library(rFSA)
url<-"http://raw.githubusercontent.com/joshuawlambert/data/master/census_data_nopct.csv"
census_data_nopct <- read.csv(file=url)

#two-way
fit_2_way<-FSA(formula = "y~1",data=census_data_nopct, fitfunc = lm,
         fixvar=NULL,quad=F,m = 2,numrs = 50, cores = 1, interactions = T,
         criterion =int.p.val, minmax = "min")
 
print(fit_2_way) #summary of solutions found
summary(fit_2_way) #list of summaries from each lm fit
plot(fit_2_way) #diagnostic plots


#three-way
fit_3_way<-FSA(formula = "y~1",data=census_data_nopct, fitfunc = lm,
         fixvar=NULL,quad=F,m = 3,numrs = 50, cores = 1, interactions = T,
         criterion =int.p.val, minmax = "min")

print(fit_3_way) #summary of solutions found
summary(fit_3_way) #list of summaries from each lm fit
plot(fit_3_way) #diagnostic plots
