# lib
import joblib
import numpy as np
import tkinter as tk
from classsfication_prmtr import my_model_test
from tkinter import messagebox
from tkinter import *
import  tkinter.ttk
import matplotlib.pyplot as plt
import tarfile
import seaborn  as sns
from sklearn.neighbors import KNeighborsClassifier
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split #for split the data
from sklearn.metrics import accuracy_score  #for accuracy_score
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.metrics import confusion_matrix #for confusion matrix
# data
path = "base.xlsx"
data = pd.read_excel(path)
# test model
def result():
    res=my_model_test(int(text1.get()),int(text2.get()), int(text3.get()))
    q=res
    lbl4 = tk.Label(win, text=q, font=("Arial Bold", 15))
    lbl4.place(x=500, y=280)
    a = None
    b = None
    c = None
# window
win = tk.Tk()
win.title(' Classification Maintenance ') # titre de fenetre
win.geometry('1000x600')
title_label=tk.Label(win, text="Classification of maintenance parameters")
title_label.pack()
r= IntVar()
R1=Radiobutton(win, text="KNN algorithm", variable=r, value=1).pack()
#R1.place(x=600, y=160)
lbl = tk.Label(win, text="Gravity", font=("Arial Bold", 10 ))
lbl.place(x=10, y=100)
text1= tk.Entry(win, width=20) # On demande ici la saisie dans le champ
text1.place(x=200, y=100)
lbl2 = tk.Label(win, text="Frequency", font=("Arial Bold", 10 ))
lbl2.place(x=10, y=140)
text2 = tk.Entry(win, width=20) # On demande ici la saisie dans le champ
text2.place(x=200, y=140)
lbl3 = tk.Label(win, text="Detection", font=("Arial Bold", 10 ))
lbl3.place(x=10, y=180)
text3= tk.Entry(win, width=20) # On demande ici la saisie dans le champ
text3.place(x=200, y=180)
lbl4=tk.Label(win ,text="Results:",font=("Arial Bold", 15 ))
lbl4.place(x=300, y=280)
button1=tk.Button(win ,text="Submit",font=("Arial Bold", 15 ),command=result)
button1.place(x=120, y=275)
lbl5=tk.Label(win, text='Model', font=("Arial Bold", 15 ))
lbl5.place(x=600, y=100)
win.mainloop()
