# -*- coding: utf-8 -*-
"""
Created on Sa Jan 2 10:45:30 2021

@author: Jovana
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC, LinearSVC


#ucitavanje baze

#trening podaci
cookies_train = pd.read_csv("cookies_train.csv")

#test podaci
cookies_test = pd.read_csv("cookies_test.csv")

print(cookies_train.shape)
#u trening skupu je 1738 uzoraka
    
print(cookies_test.shape)
#u test skupu je 193 uzorka

print(cookies_train.shape[1])
#133 obelezja i klasna labela


#podela na podatke za predvidjanje i klasne labele
X1 = cookies_train.iloc[:, :-1]
y1 = cookies_train.iloc[:, -1]

X2 = cookies_test.iloc[:, :-1]
y2 = cookies_test.iloc[:, -1]


cookies_train.loc[cookies_train['class']=='Cookies','class']= 0
cookies_train.loc[cookies_train['class']=='Pastries','class']= 1
cookies_train.loc[cookies_train['class']=='Pizzas','class']= 2

cookies_test.loc[cookies_test['class']=='Cookies','class']= 0
cookies_test.loc[cookies_test['class']=='Pastries','class']= 1
cookies_test.loc[cookies_test['class']=='Pizzas','class']= 2

print('nedostajućih vrednosti ima: ', X1.isnull().sum().sum())
print('oznake klasa su: ', y1.unique())
print('uzoraka u klasi cookies ima: ', sum(y1==0))
print('uzoraka u klasi pastries ima: ', sum(y1==1))
print('uzoraka u klasi pizzas: ', sum(y1==2))

print('uzoraka u klasi cookies ima: ', sum(y2==0))
print('uzoraka u klasi pastries ima: ', sum(y2==1))
print('uzoraka u klasi pizzas: ', sum(y2==2))

klasa_cookies = cookies_train[cookies_train["class"] == 0]
klasa_cookies = klasa_cookies.set_index('class')
klasa_pastries = cookies_train[cookies_train["class"] == 1]
klasa_pastries = klasa_pastries.set_index('class')
klasa_pizzas = cookies_train[cookies_train["class"] == 2]
klasa_pizzas = klasa_pizzas.set_index('class')

#histogrami pojavljivanja
plt.figure(figsize = (25, 5))
klasa_cookies.sum().plot(kind='bar')
plt.ylabel("Broj recepata u kojima se pojavljuje sastojak")
plt.xlabel("Sastojci za kolačiće")

plt.figure(figsize = (25, 5))
klasa_pastries.sum().plot(kind='bar')
plt.ylabel("Broj recepata u kojima se pojavljuje sastojak")
plt.xlabel("Sastojci za pecivo")


plt.figure(figsize = (25, 5))
klasa_pizzas.sum().plot(kind='bar')
plt.ylabel("Broj recepata u kojima se pojavljuje sastojak")
plt.xlabel("Sastojci za picu")


Xtr = cookies_train.iloc[:, :-1]
ytr = cookies_train.iloc[:, -1].astype('int')
print(ytr.unique())

Xts = cookies_test.iloc[:, :-1]
yts = cookies_test.iloc[:, -1].astype('int')


x_train, x_val, y_train, y_val = train_test_split(Xtr, ytr, test_size=0.1, random_state=10, stratify=ytr)

#matrica konfuzije za klasu cookies
def evaluation_classif_class_cookies(conf_mat):
    TPc = conf_mat[0, 0]
    TNc1 = conf_mat[1, 1]
    TNc2 = conf_mat[1, 2]
    TNc3 = conf_mat[2, 1]
    TNc4 = conf_mat[2, 2]
    FPc1 = conf_mat[1, 0]
    FPc2 = conf_mat[2, 0]
    FNc1 = conf_mat[0, 1]
    FNc2 = conf_mat[0, 2]
    
    precision = TPc/(TPc + FPc1 + FPc2)
    accuracy = (TPc + TNc1 + TNc2 + TNc3 + TNc4)/(TPc + TNc1 + TNc2 + TNc3 + TNc4 + FPc1 + FPc2 + FNc1 + FNc2)
    sensitivity = TPc / (TPc + FPc1 + FPc2)
    specificity = (TNc1 + TNc2 + TNc3 + TNc4)/(TNc1 + TNc2 + TNc3 + TNc4 + FPc1 + FPc2)
    F_score = 2*precision*sensitivity/(precision+sensitivity)
    print("Klasa cookies: ")
    print('precision: ', precision)
    print('accuracy: ', accuracy)
    print('sensitivity/recall: ', sensitivity)
    print('specificity: ', specificity)
    print('F score: ', F_score)
    print("")
    return accuracy

#matrica konfuzije za klasu pastries
def evaluation_classif_class_pastries(conf_mat):
    TPp = conf_mat[1, 1]
    TNp1 = conf_mat[0, 0]
    TNp2 = conf_mat[0, 2]
    TNp3 = conf_mat[2, 0]
    TNp4 = conf_mat[2, 2]
    FPp1 = conf_mat[0, 1]
    FPp2 = conf_mat[2, 1]
    FNp1 = conf_mat[1, 0]
    FNp2 = conf_mat[1, 2]

    precision1 = TPp/(TPp + FPp1 + FPp2)
    accuracy1 = (TPp + TNp1 + TNp2 + TNp3 + TNp4)/(TPp + TNp1 + TNp2 + TNp3 + TNp4 + FPp1 + FPp2 + FNp1 + FNp2)
    sensitivity1 = TPp / (TPp + FPp1 + FPp2)
    specificity1 = (TNp1 + TNp2 + TNp3 + TNp4)/(TNp1 + TNp2 + TNp3 + TNp4 + FPp1 + FPp2)
    F_score1 = 2*precision1*sensitivity1/(precision1 + sensitivity1)
    print("Klasa pastries: ")
    print('precision: ', precision1)
    print('accuracy: ', accuracy1)
    print('sensitivity/recall: ', sensitivity1)
    print('specificity: ', specificity1)
    print('F score: ', F_score1)
    print("")
    return accuracy1
    

#matrica konfuzije za klasu piazzas
def evaluation_classif_class_pizzas(conf_mat): 
    TPps = conf_mat[2, 2]
    TNps1 = conf_mat[0, 0]
    TNps2 = conf_mat[0, 1]
    TNps3 = conf_mat[1, 0]
    TNps4 = conf_mat[1, 1]
    FPps1 = conf_mat[0, 2]
    FPps2 = conf_mat[1, 2]
    FNps1 = conf_mat[2, 0]
    FNps2 = conf_mat[2, 1]
    precision2 = TPps/(TPps + FPps1 + FPps2)
    accuracy2 = (TPps + TNps1 + TNps2 + TNps3 + TNps4)/(TPps + TNps1 + TNps2 + TNps3 + TNps4 + FPps1 + FPps2 + FNps1 + FNps2)
    sensitivity2 = TPps / (TPps + FPps1 + FPps2)
    specificity2 = (TNps1 + TNps2 + TNps3 + TNps4)/(TNps1 + TNps2 + TNps3 + TNps4 + FPps1 + FPps2)
    F_score2 = 2*precision2*sensitivity2/(precision2 + sensitivity2)
    print("Klasa pizzas: ")
    print('precision: ', precision2)
    print('accuracy: ', accuracy2)
    print('sensitivity/recall: ', sensitivity2)
    print('specificity: ', specificity2)
    print('F score: ', F_score2)
    print("")
    return accuracy2

#KNN klasifikator

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
acc = []
for m in ['hamming', 'jaccard']:
    for d in ['distance', 'uniform']:
        for i in range(1, 10):
            indexes = kf.split(Xtr, ytr)
            acc_t = []
            fin_conf_mat = np.zeros((len(np.unique(ytr)),len(np.unique(ytr))))
            for train_index, test_index in indexes:
                knn = KNeighborsClassifier(n_neighbors=i, metric=m, weights = d)
                knn.fit(Xtr.iloc[train_index,:], ytr.iloc[train_index])
                y_pred = knn.predict(Xtr.iloc[test_index,:])
                acc_t.append(accuracy_score(ytr.iloc[test_index], y_pred))
                fin_conf_mat += confusion_matrix(ytr.iloc[test_index], y_pred, labels=[0, 1, 2])
            print('za matriku=', m, ', tezinu=', d, ' i broj susjeda i = ', i ,'tacnost je: ', np.mean(acc_t),
                  ' a mat. konf. je:')
            print(fin_conf_mat)
            pr_tacnost =((evaluation_classif_class_cookies(fin_conf_mat) + evaluation_classif_class_pizzas(fin_conf_mat) + evaluation_classif_class_pastries(fin_conf_mat))/3)
            print("prosjecna tacnost klasifikatora: " ,pr_tacnost)
            acc.append(np.mean(acc_t))
print('najbolja tacnost je u iteraciji broj: ', np.argmax(acc))


#testiranje na test skupu
classifierknn = KNeighborsClassifier(n_neighbors=8, metric='hamming', weights = 'distance')
classifierknn.fit(Xtr, ytr)
y_pred1 = classifierknn.predict(Xts)
conf_mat1 = confusion_matrix(yts, y_pred1)
print('finalna matrica je: ')
print(conf_mat1)
pr_tacnost =((evaluation_classif_class_cookies(conf_mat1) + evaluation_classif_class_pizzas(conf_mat1) + evaluation_classif_class_pastries(conf_mat1))/3)
print(pr_tacnost)
print("Prosjecna tacnost klasifikatora je: ", pr_tacnost)
print('procenat pogodjenih uzoraka: ', accuracy_score(yts, y_pred1))
print('preciznost mikro: ', precision_score(yts, y_pred1, average='micro'))
print('preciznost makro: ', precision_score(yts, y_pred1, average='macro'))
print('osetljivost mikro: ', recall_score(yts, y_pred1, average='micro'))
print('osetljivost makro: ', recall_score(yts, y_pred1, average='macro'))
print('f mera mikro: ', f1_score(yts, y_pred1, average='micro'))
print('f mera makro: ', f1_score(yts, y_pred1, average='macro'))


#SVM klasifikator
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
acc = []
for c in [1, 10, 50]:
    for F in ['linear', 'rbf']:         
        for mc in ['ovo', 'ovr']:
            indexes = kf.split(Xtr, ytr)
            acc_tmp = []
            fin_conf_mat = np.zeros((len(np.unique(ytr)),len(np.unique(ytr))))
            for train_index, test_index in indexes:
                classifier = SVC(C=c, kernel=F, decision_function_shape=mc)
                classifier.fit(Xtr.iloc[train_index,:], ytr.iloc[train_index])
                y_pred = classifier.predict(Xtr.iloc[test_index,:])
                acc_tmp.append(accuracy_score(ytr.iloc[test_index], y_pred))
                fin_conf_mat += confusion_matrix(ytr.iloc[test_index], y_pred, labels=[0,1,2])
            print('za parametre C=', c, ', kernel=', F, ' i pristup ', mc, ' tacnost je: ', np.mean(acc_tmp),
                  ' a mat. konf. je:')
            print(fin_conf_mat)
            pr_tacnost =((evaluation_classif_class_cookies(fin_conf_mat) + evaluation_classif_class_pizzas(fin_conf_mat) + evaluation_classif_class_pastries(fin_conf_mat))/3)
            print("prosjecna tacnost klasifikatora:", pr_tacnost)
            acc.append(np.mean(acc_tmp))
print('najbolja tacnost je u iteraciji broj: ', np.argmax(acc))

classifier = SVC(C=50, kernel='linear', decision_function_shape='ovr')
classifier.fit(Xtr, ytr)
y_pred = classifier.predict(Xts)
conf_mat = confusion_matrix(yts, y_pred, labels=[0, 1, 2])
print(conf_mat)
pr_tacnost =((evaluation_classif_class_cookies(conf_mat) + evaluation_classif_class_pizzas(conf_mat) + evaluation_classif_class_pastries(conf_mat))/3)
print(pr_tacnost)
print('procenat pogodjenih uzoraka: ', accuracy_score(yts, y_pred))
print('preciznost mikro: ', precision_score(yts, y_pred, average='micro'))
print('preciznost makro: ', precision_score(yts, y_pred, average='macro'))
print('osetljivost mikro: ', recall_score(yts, y_pred, average='micro'))
print('osetljivost makro: ', recall_score(yts, y_pred, average='macro'))
print('f mera mikro: ', f1_score(yts, y_pred, average='micro'))
print('f mera makro: ', f1_score(yts, y_pred, average='macro'))


