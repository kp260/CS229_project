# A step-by-step description is provided throughout this code.
# This code extracts necessary information from Acquisition and Performance files and combines these using key = LOAN_ID

#######################################################################################################################################

# Load Necessary Packages for this analysis

if (!(require(foreach))) install.packages ("foreach")
if (!(require(data.table))) install.packages ("data.table")
if (!(require(zoo))) install.packages ("zoo")
if (!(require(reshape))) install.packages('reshape')
if (!(require(ade4))) install.packages('ade4')
if (!(require(dummies))) install.packages('dummies')
library(ade4)
library(reshape)
library(data.table)
library(dummies)


# You will need the path to where you have saved the downloaded text files, please copy and paste or type the path below.
fileslocation<- '...' 


# Check the number of files downloaded (should be even, equal number of Acquisition and Performance Files).
numberoffiles<-length(list.files(fileslocation, pattern = glob2rx("*txt"), full.names=TRUE))

# The "foreach" package contructs a loop so that R can iterate through all pairs of related Acquisition and Performance files.
# Calculate the number of iterations/cores in parallel processing allowing each pair to be processed simultaneously.
numberofloops<-(numberoffiles/2)


substrRight <- function(x, n){
  substr(x, nchar(x)-n+1, nchar(x))
}


# Read in historical quarter FedRates 30y, add the path details for where the file is saved 
rates<-read.csv(".../FedRates.csv", header=TRUE, sep=",")
#

####################################################################
# Start of Part 1; Data Preperation Step
####################################################################

# After defining the Acquisition and Performance variables and their classes, the files are read into R and then data manipulation is carried out. 
# Acquisition and Performance files (from one or many quarters) will be merged into an R dataframe called "Combined_Data."
Combined_Data_list = list() 
foreach(k=1:numberofloops, .inorder=FALSE,.packages=c("data.table", "zoo", "foreach")) %do% {
                    
                            
                           # Define Acquisition variables and classes, and read the files into R.
                           Acquisitions <- list.files(fileslocation, pattern = glob2rx("*Acquisition*txt"), full.names=TRUE)
                           
                           file_name = substrRight(Acquisitions[k], 10)
                           file_name = substr(file_name, 1, 6)
                           
                           colA = c("character", "NULL", "NULL", "numeric", "numeric", "NULL", "NULL", 
                                    "NULL", "numeric","NULL", "character", "numeric", "numeric", "NULL", "NULL", "NULL", 
                                    "NULL", "NULL","NULL", "character", "numeric", "NULL", "NULL", "NULL", "NULL")
                           avar = c("LOAN_ID", "ORIG_RT", "ORIG_AMT","OLTV",  "NUM_BO", "DTI", "CSCORE_B",  "ZIP_3", "MI_PCT")
                             
                           Data_A<- fread(Acquisitions[k], sep = "|", colClasses = colA, showProgress=TRUE)
                           setnames(Data_A, avar)
                           setkey(Data_A, "LOAN_ID")
                           
                           Data_A<-Data_A[complete.cases(Data_A), ]
                           
                           # Scale ORIG_RT column by the 30y FedRate from the corresponding year quarter so that rates can be compared through time
                           rate = rates$Average[rates$Quarter == file_name]
                           orig_rate = Data_A$ORIG_RT
                           Data_A$ORIG_RT<-Data_A$ORIG_RT/rate
                           
                           # Remove not-needed Acquisition data from R environment.
                           rm('colA', 'avar')
                           
                           # Define Performance variables and classes, and read the files into R.
                           Performance <- list.files(fileslocation, pattern = glob2rx("*Performance*txt"), full.names=TRUE)
                           
                           # Read and Process Performance data
                           colP = c("character", "NULL", "NULL", "NULL", "NULL", "numeric", "NULL", "NULL", "NULL",
                                    "NULL", "numeric", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL",
                                    "NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL")
                          
                           Data_P = fread(Performance[k], sep = "|", colClasses = colP, showProgress=TRUE)
                           pvar = c("LOAN_ID",  "Loan.Age",  "Delq.Status")

                           setnames(Data_P, pvar)

                           setkey(Data_P, "LOAN_ID")

                           # Count the number of months a loan is active 
                           Data_P[,AgeCount:=1:.N, by="LOAN_ID"]
                           
                           
                           # Delete data where age of loan is more than 24 months, delinquency status has non NA values and no 999 status
                           Data_Pmod<-Data_P[(AgeCount<=24 & Delq.Status!="X"),]
                          
                           # Melt and pivot the modified dataset
                           Data_Pmod = melt(Data_Pmod, id.vars = c("LOAN_ID", "AgeCount"), 
                                                measure.vars = c("Delq.Status"))
                           
                           Data_Pmod<-dcast(Data_Pmod, LOAN_ID ~ AgeCount, value.var = "value")
                           
                           # Remove rows with NA values, i.e. loans that have been live less than 24 months
                           Data_Pmod<-Data_Pmod[complete.cases(Data_Pmod), ]
                           
                           # Classify months 13-24

                           Data_Pmod[, c("Class"):=
                                    list(apply(Data_Pmod[,14:ncol(Data_Pmod)],1,max))]
                           Data_Pmod$Class[Data_Pmod$Class>=6] = 6
  
                           
                           # Remove not-needed data from R environment.
                           Data_Pmod[, c("13","14","15", "16", "17", "18", "19", "20", "21", "22", "23", "24" ):=NULL]
                           
                           # Merge together full Acquisition and Performance files.
                           Combined_Data = as.data.table(merge(Data_A, Data_Pmod, by.x = "LOAN_ID", by.y = "LOAN_ID", all = TRUE))
                           Combined_Data<-Combined_Data[complete.cases(Combined_Data), ]
                           Combined_Data_mod<-rbind(Combined_Data)
                           
                           # Output data for each Q add path
                           write.csv(Combined_Data, paste0(".../FNMA_Performance_Data_", file_name, ".csv"))
                           print("data saved for ... ")
                           print(file_name)
                           
                           Combined_Data_list[[k]]<-Combined_Data_mod
                           rm(Data_P, Data_A, Data_Pmod)

                           
                           }
                           
Combined_Data<-rbindlist(Combined_Data_list, fill=TRUE)
#Combined_Data<-rbind(Combined_Data_list)


# Save a Copy to disk or write a .csv file.
write.csv(Combined_Data, file = "path/FNMA_Performance_Data_all.csv")

####################################################################
# End of Part 1; Data Preparation Step
####################################################################






