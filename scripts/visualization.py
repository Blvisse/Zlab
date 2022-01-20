
#%%


'''
This script creates plot visualization of data

'''

import pandas as pd
import numpy as np
import sys
import os
from faker import Faker
import matplotlib.pyplot as plt
from torch import xlogy_
import seaborn as sns

print("Successfully imported libraries")


faker = Faker()
#we read data file

#we generate fake age data
age=[np.random.randint(18,70) for i in range(1000)]
print(age[:10])

#we also generate fake gender data
gender=[np.random.choice(['M','F']) for i in range(1000)]
print(gender[:10])

#create a years of experience column as well
experience=[np.random.randint(1,18) for i in range(1000)]



#we zip the age and gender lists 
data=[*zip(gender,age)]
print(data[:10])

#we read data file
try:
    dataset=pd.read_csv("../data/date_data.csv")
    print("read data file")
    
except FileNotFoundError as fe:
    print("Error file not found error!")
    print("Exiting program")
    sys.exit(1)
    
except Exception as e:
    
    print("Error occurred exiting program")
    sys.exit(1)
    
    
#we create new columns for our new features
print(dataset.head())

dataset['gender']=gender
dataset['age']=age
dataset['experience']=experience

print("New data: /n")
print(dataset.head())

dataset.to_csv("updated_date.csv",index=False)
print("Saved dataset to csv flat file")


# we create functions for visualization of our data

def scatter_plot(col1,col2):
    plt.figure(figsize=(10,10))
    sns.scatterplot(data=dataset,x=col1,y=col2)
    plt.title("Scatter plot showing distribution of {} vs {} ".format(col1,col2))
    plt.show()
    
    
scatter_plot('age','experience')

def bar_plot(col1,col2):
    plt.figure(figsize=(10,10))
    # col2=np.array(col2)
    
    sns.barplot(data=dataset,x=col1,y=col2)
    plt.title("Bar plot showing {} vs {} ".format(col1,col2))
    plt.show()
    
bar_plot('gender','experience')
def pie_chart(col):
    plt.figure(figsize=(10,10))
    plt.pie(x=dataset[col].value_counts(),labels=dataset[col].unique(),autopct='%1.1f%%')
    plt.title("Pie chart showing {} distribution".format(col))
    plt.show()

pie_chart('gender')
    
# %%
