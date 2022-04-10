# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 18:32:59 2022

@author: Markus Fritsche
"""

import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Classification Dataset
df_cla = pd.read_csv("data/phpV5QYya.csv")

#Barplot Predictor variable
sns.countplot(x="Result", data=df_cla)


#Regression Dataset
with open('data/wikivital_mathematics.json') as data_file:    
    d= json.load(data_file)

keys = d.keys()

topics = d["node_ids"].keys()
topics_id = d["node_ids"]

#Select only the Numbers per day
time_series_keys = list(d.keys())[4:]
d_subset = {key: d[key] for key in time_series_keys}

i=1
dates = {}
sum_no_of_visitors = {}
for key in d_subset:
    i=i+1
    date = str(d_subset[key]["year"])+"-"+str(d_subset[key]["month"])+"-"+str(d_subset[key]["day"])
    dates[key] = date
    
    sum_no_of_visitors[key] = sum(d_subset[key]["y"])
    
    #if i > 2:
    #    break #just to not have to iterate through everything
    
#Create DataFrame with Dates & Sum of Visitors    
new_df = pd.DataFrame(data= (dates,sum_no_of_visitors), index=["Date","NoVisitors"]).transpose()

new_df["NoVisitors"] = pd.to_numeric(new_df["NoVisitors"])

#Create Histogram for Dates & No of Visitors
plt.bar(new_df["Date"], new_df["NoVisitors"], color ='blue',
        width = 1)

plt.xlabel("Dates")
plt.ylabel("No of Visitors in Million Clicks")
#plt.title("No of Visitors on all Math Wikipedia Pages per day")

x=list(range(0, 744, 31))
values = ['March','April','May','June','July','August','September','October','November','December','January','February','March','April','May','June','July','August','September','October','November','December','January','February']

plt.xticks(x,values,rotation=90)

#y=list(range(0,1400000,int(1400000/8)))
#yvalues= ['0','200000','400000','600000','800000','1000000','1200000','1400000']

#plt.yticks(y, yvalues)
#plt.show()

plt.savefig("barplot_novisitors_per_date.png")

#print(len(d_subset["0"]["y"]))

#print(topics_id)
#print(d["edges"])
#print(d["weights"])
#print(d["node_ids"]) #Topics
#print(d["time_periods"]) #731?