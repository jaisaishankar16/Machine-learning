import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

titanic_data=pd.read_csv("titanic_data.csv")
print(titanic_data)
# print(sns.countplot(x="Survived",data=titanic_data))          #to plot the graph determining how many died and how many survived
# print(sns.countplot(x="Survived",hue="Sex",data=titanic_data))  #tp plot the graph determining how many male survived vs female and for death

#data wrangling or data cleaning


# print(titanic_data.isnull().sum())            #gives total null values in a column
titanic_data.dropna(inplace=True)               #drops every record containing Nan value 
print(titanic_data)
plt.show()
