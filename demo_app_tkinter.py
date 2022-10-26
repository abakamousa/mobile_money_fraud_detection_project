# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 18:34:15 2022

@author: engantchou
"""
import numpy             as np 
import pandas            as pd 
# import opendatasets      as od
import seaborn           as sns
import matplotlib.pyplot as plt


from tkinter import *
from tkinter import Frame, Tk, Button
import tkinter.messagebox
import tkinter.filedialog

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from sklearn.preprocessing     import LabelEncoder
from sklearn.ensemble          import RandomForestClassifier
from sklearn.linear_model      import SGDClassifier
from sklearn.linear_model      import LogisticRegression
from xgboost                   import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing     import RobustScaler
# from imblearn.over_sampling    import SMOTE 
from sklearn.model_selection   import train_test_split, GridSearchCV
from collections               import Counter
from sklearn.pipeline          import Pipeline

#====================================================
root= tk.Tk()
root.title('MOBILE MONEY FRAUD DETECTION APPLICATION')
root.configure(bg ="#87CEEB")

#=============================== Canavas===========================
canvas1 = tk.Canvas(root, width = 800, height = 300)
canvas1.pack()


#=======================================Frame======================
label1 = tk.Label(root, text="MOBILE MONEY FRAUD DETECTION")
label1.config(font=('Arial', 20), bg="#87CEEB", fg = "white", relief=RIDGE)
canvas1.create_window(400, 50, window=label1)
    
#==============================================
def create_charts():
    global ax
    global bar1
    global pie1
    df = pd.read_csv('fraud_detection_dataset.csv' , encoding = "ISO-8859-1" , sep = ",")
    df.isnull().sum()
    df['isFraud'].value_counts(normalize=True)
    df['isFlaggedFraud'].value_counts(normalize=True)
    #================ figure 1
    figure1 = plt.figure(figsize=(10,8))
    ax = sns.countplot(x = "type", hue="isFraud", data = df)
    plt.title('Countplot of different types of transaction (nonFraud and Fraud)')
    for p in ax.patches:
      ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+50))
    bar1 = FigureCanvasTkAgg(figure1, root)
    bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=0)
    
    #================= figure 2
    type = df['type'].value_counts()
    transaction = type.index
    count = type.values
    
    figure2 = plt.figure(figsize=(10,8))
    plt.pie(count, labels=transaction, autopct='%1.0f%%')
    plt.legend(loc='lower left')
    plt.legend(loc='lower left')
    pie1 = FigureCanvasTkAgg(figure2, root)
    pie1.get_tk_widget().pack()
    plt.show()


#========================clear all charts
def clear_charts():
    bar1.get_tk_widget().pack_forget()
    pie1.get_tk_widget().pack_forget()
    
#==================================
button1 = tk.Button(root, text="Create Charts", command=create_charts, bg='palegreen2', font=('Arial', 11, 'bold'))
canvas1.create_window(400, 180, window=button1)
    
button2 = tk.Button(root, text="Clear Charts", command=clear_charts, bg='lightskyblue2', font=('Arial', 11, 'bold'))
canvas1.create_window(400, 220, window=button2)
    
button3 = tk.Button(root, text="Exit Application", command=root.destroy, bg='lightsteelblue2', font=('Arial', 11, 'bold'))
canvas1.create_window(400, 260, window=button3)

#==============================
root.mainloop()
