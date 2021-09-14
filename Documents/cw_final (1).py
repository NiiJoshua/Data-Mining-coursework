#!/usr/bin/env python
# coding: utf-8

# ### Library imports

# In[2]:


import pandas as pd
import numpy.random as nr
import pandas_profiling
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.preprocessing import StandardScaler
import sklearn.decomposition as skde
from sklearn.ensemble import ExtraTreesClassifier
import sklearn.model_selection as ms
from sklearn.model_selection import train_test_split
from sklearn import linear_model


# In[3]:


df = pd.read_csv('CW_data.csv')
df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[13]:


df.Class.value_counts().plot(kind='bar')


# ### Data Profiling

# In[157]:


# profile = ProfileReport(df, title="Pandas Profiling Report")
# profile


# ### Class imbalance

# In[158]:


class_count = df['Class'].value_counts()
print(class_count)


# In[159]:


# Calculating percentage of positive and negative classes
total = 4907
negative = 3418
positive = 1489

print("Percentage of positive classes: %.2f%%" % ((positive/total) *100))
print("Percentage of negative classes: %.2f%%" % ((negative/total) *100))


# ### Findings from the first EDA
# 
# A preliminary profile of the data was conducted using the pandas data profling libray. This Library was created by Simon Brugman(#search year). Initial exploration of the training dataset revealed the following:
# 
# 1. There are 451 variables or columns in total
# 2. There are 4907 total oberservations or rows
# 3. No missing values were detected, hence the data is complete irrespective of the fact that there are a lot of zero's
# 4. There are no duplicate rows
# 5. In all, the data size is 16.9 MB
# 6. Most of the numerical values have a normal distribution
# 7. 327 of the columns are of float data type
# 8. 120 columns are of int64 data type
# 9. 4 columns are of string data type
# 10. The dataset is imbalanced; out of 4907, 3418 representing 69.66% had an outcome of -1 and 1489 i.e 30.34% had an outcome of 1
# 

# ## Data Cleaning and Pre-processing
# 
# ### Data cleaning

# In[160]:


cw_df = df.copy()   # making a copy of the df and naming it cw_df
cw_df.columns


# In[161]:


# Remove the categorical columns since they play no role
cw_df.drop(['Info_PepID', 'Info_protein_id', 'Info_center_pos', 'Info_AA',
       'Info_window_seq'], axis=1, inplace=True)
cw_df.columns


# In[162]:


cw_df.info()


# #### Data decomposition
# 
# * This is allow for easier analysis of the data. Each sub will have 50 columns

# In[163]:


# slicing cw_1 with 49 rows
cw_1 = cw_df.loc[:, "feat_seq_entropy":"feat_VHSE4":1].copy()
cw_2 = cw_df.loc[:, "feat_VHSE5":"feat_CT022":1].copy()
cw_3 = cw_df.loc[:, "feat_CT023":"feat_CT123":1].copy()
cw_4 = cw_df.loc[:, "feat_CT124":"feat_CT224":1].copy()
cw_5 = cw_df.loc[:, "feat_CT225":"feat_CT325":1].copy()
cw_6 = cw_df.loc[:, "feat_CT326":"feat_CT426":1].copy()
cw_7 = cw_df.loc[:, "feat_CT430":"feat_CT530":1].copy()
cw_8 = cw_df.loc[:, "feat_CT531":"feat_CT631":1].copy()
cw_9 = cw_df.loc[:, "feat_CT632":"feat_Perc_Y":1].copy()


# In[164]:


print(len(cw_1.columns))
print(len(cw_2.columns))
print(len(cw_3.columns))
print(len(cw_4.columns))
print(len(cw_5.columns))
print(len(cw_6.columns))
print(len(cw_7.columns))
print(len(cw_8.columns))
print(len(cw_9.columns))
print(len(cw_df.columns))


# #### Data visualization
# 
# This process is to have an overview of the distribution of the columns and to detect outliers. This will be done using the seaborn
# 
# 

# In[165]:


def box_plot(cw_1,col):
    for col in cw_1.columns:
        sns.set_style("whitegrid")
        sns.boxplot(col, data=cw_1)
        plt.show()
        
box_plot(cw_1, cw_1.columns)


# In[166]:


cw_1.head()


# In[167]:


cw_1['feat_S_atoms'].value_counts()


# ### Analysis of EDA for cw_1
# 
# 1. All the columns have outliers except feat_Perc__small
# 2. feat_S_atoms is an irrelevant columns becuase it has a lot of zeros

# In[168]:


# Box plot for cw_2
box_plot(cw_2, cw_2.columns) 


# ### Analysis of EDA for cw_2
# 
# 1. All the columns have outliers except feat_VHSE7, feat_MSWHIM2, feat_MSWHIM3
# 2. feat_CT000 to feat_CT022 are irrelevant columns becuase they have a lot of zeros

# In[169]:


box_plot(cw_3, cw_3.columns) 


# ### Analysis of EDA for cw_3
# 
# 1. All the columns in cw_3 are irrelevant because all the columns are almost zero

# In[170]:


box_plot(cw_4, cw_4.columns) 


# ### Analysis of EDA for cw_4
# 
# 1. All the columns in cw_4 are irrelevant because all the columns are almost zero

# In[171]:


box_plot(cw_5, cw_5.columns) 


# ### Analysis of EDA for cw_5
# 
# 1. All the columns in cw_5 are irrelevant because all the columns are almost zero

# In[172]:


box_plot(cw_6, cw_6.columns) 


# ### Analysis of EDA for cw_6
# 
# 1. All the columns in cw_6 are irrelevant because all the columns are almost zero

# In[173]:


box_plot(cw_7, cw_7.columns) 


# ### Analysis of EDA for cw_7
# 
# 1. All the columns in cw_7 are irrelevant because all the columns are almost zero

# In[174]:


box_plot(cw_8, cw_8.columns) 


# ### Analysis of EDA for cw_8
# 
# 1. All the columns in cw_8 are irrelevant because all the columns are almost zero

# In[175]:


box_plot(cw_9, cw_9.columns) 


# ### Analysis of EDA for cw_9
# 
# 1. feat_Perc_A, feat_Perc_D, feat_Perc_E, feat_Perc_G, feat_Perc_I, feat_Perc_K, feat_Perc_L, feat_Perc_N, feat_Perc_P, feat_Perc_Q, feat_Perc_R, feat_Perc_S, feat_Perc_T, feat_Perc_V are the relevant columns.
# 2. The other columns are irrelevant since they have almost all entries being zero

# ### Data Profiling
# 
# Data profile was perfomed on cw_1, cw_2 and cw_9

# In[176]:


"""
profile_cw_1 = ProfileReport(cw_1, title="Pandas Profiling Report")
profile_cw_1.to_file("cw_1.html")

"""


# In[177]:


"""
profile_cw_2 = ProfileReport(cw_2[['feat_VHSE5', 'feat_VHSE6', 'feat_VHSE7', 'feat_VHSE8', 'feat_ProtFP1',
       'feat_ProtFP2', 'feat_ProtFP3', 'feat_ProtFP4', 'feat_ProtFP5',
       'feat_ProtFP6', 'feat_ProtFP7', 'feat_ProtFP8', 'feat_ST1', 'feat_ST2',
       'feat_ST3', 'feat_ST4', 'feat_ST5', 'feat_ST6', 'feat_ST7', 'feat_ST8',
       'feat_BLOSUM1', 'feat_BLOSUM2', 'feat_BLOSUM3', 'feat_BLOSUM4',
       'feat_BLOSUM5', 'feat_BLOSUM6', 'feat_BLOSUM7', 'feat_BLOSUM8',
       'feat_BLOSUM9', 'feat_BLOSUM10', 'feat_MSWHIM1', 'feat_MSWHIM2',
       'feat_MSWHIM3']], title="Pandas Profiling Report")
profile_cw_2.to_file("profile_cw_2.html")

"""


# In[178]:


"""
profile_cw_9 = ProfileReport(cw_9[['feat_Perc_A', 'feat_Perc_D', 'feat_Perc_E', 'feat_Perc_G', 'feat_Perc_I', 'feat_Perc_K', 'feat_Perc_L', 'feat_Perc_N', 'feat_Perc_P', 'feat_Perc_Q', 'feat_Perc_R', 'feat_Perc_S','feat_Perc_T', 'feat_Perc_V']], title="Pandas Profiling Report")
profile_cw_9.to_file("profile_cw_9.html")

"""


# #### Final take
# 
# * From the preliminary EDA, relevant columns we're working with are from cw_1, cw_2, and cw_9.
# * A total of 95 columns out of 450 (ecluding the label column) were relevant columns

# ### Visualizing class separation by numeric features
# 
# The goal of this visualization is to understand which features are useful for class separation and for the machine learning process.
# 
# By their very construction, box plots will help us focus on overlaps(or not) of the quartiles of the distribution. If there is sufficient differences in the quartiles for the feature to useful in separation of the label classes?. This is the question we want to answer with the box plot

# In[179]:


def box_plot_separation(cw_df, cols, label = 'Class'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.boxplot(label, col, data = cw_df)
        plt.xlabel(label)
        plt.ylabel(col)
        plt.show()
        


# #### Separation for cw_1
# 
# * From the box plot, most of the features are able to give some separation for the classes, i.e, -1 and 1. These columns however, excludes the 'feat_S_atom' column since its full of zeros.
# 

# In[180]:


cw_1_cols = ['feat_seq_entropy', 'feat_C_atoms', 'feat_H_atoms', 'feat_N_atoms',
       'feat_O_atoms', 'feat_molecular_weight',
       'feat_Perc_Tiny', 'feat_Perc_Small', 'feat_Perc_Aliphatic',
       'feat_Perc_Aromatic', 'feat_Perc_NonPolar', 'feat_Perc_Polar',
       'feat_Perc_Charged', 'feat_Perc_Basic', 'feat_Perc_Acidic', 'feat_PP1',
       'feat_PP2', 'feat_PP3', 'feat_KF1', 'feat_KF2', 'feat_KF3', 'feat_KF4',
       'feat_KF5', 'feat_KF6', 'feat_KF7', 'feat_KF8', 'feat_KF9', 'feat_KF10',
       'feat_Z1', 'feat_Z2', 'feat_Z3', 'feat_Z4', 'feat_Z5', 'feat_F1',
       'feat_F2', 'feat_F3', 'feat_F4', 'feat_F5', 'feat_F6', 'feat_T1',
       'feat_T2', 'feat_T3', 'feat_T4', 'feat_T5', 'feat_VHSE1', 'feat_VHSE2',
       'feat_VHSE3', 'feat_VHSE4']
box_plot_separation(cw_df, cw_1_cols)


# #### Separation for cw_2
# 
# * From the box plot, most of the features are able to give some separation for the classes, i.e, -1 and 1. These columns however, excludes from CT_000 to CT_022 column since they're full of zeros.

# In[181]:


cw_2_cols = ['feat_VHSE5', 'feat_VHSE6', 'feat_VHSE7', 'feat_VHSE8', 'feat_ProtFP1',
       'feat_ProtFP2', 'feat_ProtFP3', 'feat_ProtFP4', 'feat_ProtFP5',
       'feat_ProtFP6', 'feat_ProtFP7', 'feat_ProtFP8', 'feat_ST1', 'feat_ST2',
       'feat_ST3', 'feat_ST4', 'feat_ST5', 'feat_ST6', 'feat_ST7', 'feat_ST8',
       'feat_BLOSUM1', 'feat_BLOSUM2', 'feat_BLOSUM3', 'feat_BLOSUM4',
       'feat_BLOSUM5', 'feat_BLOSUM6', 'feat_BLOSUM7', 'feat_BLOSUM8',
       'feat_BLOSUM9', 'feat_BLOSUM10', 'feat_MSWHIM1', 'feat_MSWHIM2',
       'feat_MSWHIM3']
box_plot_separation(cw_df, cw_2_cols)


# #### Separation for cw_9
# 
# From the box plot, most of the selected features are not giving a clear separation of the classes. These columns will most likely not be relevant for the machine learning process

# In[182]:


cw_9_cols = ['feat_Perc_A', 'feat_Perc_D',
       'feat_Perc_E', 'feat_Perc_G',
       'feat_Perc_I', 'feat_Perc_K', 'feat_Perc_L',
       'feat_Perc_N', 'feat_Perc_P', 'feat_Perc_Q', 'feat_Perc_R',
       'feat_Perc_S', 'feat_Perc_T', 'feat_Perc_V']
box_plot_separation(cw_df, cw_9_cols)


# ### Treating of outliers
# 
# From the detected outliers,we will use the .clip of dataframe properties to set all values below the 25th percentile to the value at the 25th percentile and all values above the the 75th percentile to the value at the 75th percentile.

# In[183]:


cw_1_relevant = cw_1.drop(['feat_S_atoms'], axis=1).copy()
cw_1_relevant = cw_1_relevant.clip(lower=cw_1_relevant.quantile(0.05), upper=cw_1_relevant.quantile(0.95), axis=1)


# In[184]:


box_plot(cw_1_relevant, cw_1_relevant.columns)


# #### cw_1_relevant
# 
# * cw_1_relevant dataframe was selected from cw_1 which excluded the 'feat_S_atom' column
# * feat_Perc_Aromatic, feat_Perc_Aliphatic, and feat_F5 still couldn't clip all values but it can be ignored.

# In[185]:


cw_2_relevant = cw_2[['feat_VHSE5', 'feat_VHSE6', 'feat_VHSE7', 'feat_VHSE8', 'feat_ProtFP1',
       'feat_ProtFP2', 'feat_ProtFP3', 'feat_ProtFP4', 'feat_ProtFP5',
       'feat_ProtFP6', 'feat_ProtFP7', 'feat_ProtFP8', 'feat_ST1', 'feat_ST2',
       'feat_ST3', 'feat_ST4', 'feat_ST5', 'feat_ST6', 'feat_ST7', 'feat_ST8',
       'feat_BLOSUM1', 'feat_BLOSUM2', 'feat_BLOSUM3', 'feat_BLOSUM4',
       'feat_BLOSUM5', 'feat_BLOSUM6', 'feat_BLOSUM7', 'feat_BLOSUM8',
       'feat_BLOSUM9', 'feat_BLOSUM10', 'feat_MSWHIM1', 'feat_MSWHIM2',
       'feat_MSWHIM3']].copy()
cw_2_relevant = cw_2_relevant.clip(lower=cw_2_relevant.quantile(0.05), upper=cw_2_relevant.quantile(0.95), axis=1)


# In[186]:


box_plot(cw_2_relevant, cw_2_relevant.columns)


# #### cw_2_relevant
# 
# * cw_2_relevant dataframe was selected from cw_2 consisting of columns with all rows not zeros
# * All columns were well clipped.

# In[187]:


cw_9_relevant = cw_9[['feat_Perc_A', 'feat_Perc_D',
       'feat_Perc_E', 'feat_Perc_G',
       'feat_Perc_I', 'feat_Perc_K', 'feat_Perc_L',
       'feat_Perc_N', 'feat_Perc_P', 'feat_Perc_Q', 'feat_Perc_R',
       'feat_Perc_S', 'feat_Perc_T', 'feat_Perc_V']].copy()
cw_9_relevant = cw_9_relevant.clip(lower=cw_9_relevant.quantile(0.05), upper=cw_9_relevant.quantile(0.95), axis=1)


# In[188]:


box_plot(cw_9_relevant, cw_9_relevant.columns)


# #### cw_9_relevant
# 
# * cw_9_relevant dataframe was selected from cw_9 consisting of columns with all rows not zeros
# * There were some extreme values which did not clip but they can be ignored.

# In[189]:


cw_1_relevant.head()


# In[190]:


cw_1_relevant.describe()


# ### Data tranformation
# 
# From cw_1, feat_C_atoms, feat_H_atoms, feat_N_atoms, feat_O_atoms and feat_molecular_weight have large values which need to be Standardaized so their values do not overshadow those with lesser values.

# In[191]:


def hist_plot(vals, lab):
    sns.distplot(vals)
    plt.title('Histogram of ' + lab)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()
    
hist_plot(cw_1_relevant['feat_C_atoms'], 'feat_C_atoms')


# In[192]:


hist_plot(np.log(cw_1_relevant['feat_C_atoms']), 'log of feat_C_atoms')


# In[193]:


hist_plot(cw_1_relevant['feat_H_atoms'], 'feat_H_atoms')


# In[194]:


hist_plot(np.log(cw_1_relevant['feat_H_atoms']), 'log of feat_H_atoms')


# In[195]:


hist_plot(cw_1_relevant['feat_N_atoms'], 'feat_N_atoms')


# In[196]:


hist_plot(np.log(cw_1_relevant['feat_N_atoms']), 'log of feat_N_atoms')


# In[197]:


hist_plot(cw_1_relevant['feat_O_atoms'], 'feat_O_atoms')


# In[198]:


hist_plot(np.log(cw_1_relevant['feat_O_atoms']), 'log of feat_O_atoms')


# In[199]:


hist_plot(cw_1_relevant['feat_molecular_weight'], 'feat_molecular_weigt')


# In[200]:


hist_plot(np.log(cw_1_relevant['feat_molecular_weight']), 'log of feat_molecular_weigt')


# In[201]:


# Convert the necessary features to log
cw_1_relevant[['feat_C_atoms','feat_H_atoms','feat_N_atoms','feat_O_atoms','feat_molecular_weight']] = cw_1_relevant[['feat_C_atoms','feat_H_atoms','feat_N_atoms','feat_O_atoms','feat_molecular_weight']].apply(lambda x: np.log10(x))
cw_1_relevant.head()


# In[202]:


cw_1_relevant.head()


# In[203]:


# concatenate the relevant features
features = pd.concat([cw_1_relevant, cw_2_relevant, cw_9_relevant], join = 'outer', axis=1)
features.head()


# In[204]:


print(len(cw_1_relevant.columns))
print(len(cw_2_relevant.columns))
print(len(cw_9_relevant.columns))


# In[205]:


label = cw_df[['Class']]
label.head()


# In[206]:


# Checking for shapes to ensure both featrues and label are of identical shape
print(features.shape)
print(label.shape)


# In[207]:


# Scale all features with the StandardScaler
scaler = StandardScaler()
Features = pd.DataFrame(scaler.fit_transform(features), columns = features.columns)
# Features[:, 0:10]
Features.head()


# In[208]:


#Transforming the features and labl into array and renaming them X,y respectively
X,y = Features.values , label.values


# In[209]:


# Split the data into train set and test set with a training size of .70 and test set of .30 respectively 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.30, random_state = 0)


# ### Feature selection methods
# 
# #### Method 1: Using ExtraTreesClassifier from sklearn

# In[210]:


model = ExtraTreesClassifier()
model.fit(Features, label)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_,index=Features.columns)
feat_importances.nlargest(18).plot(kind='barh')
plt.show()


# The algorithm returned the 18 most influencial features from the initial 95. The number 18 was chosen after the drawing the scree diagram which showed the 18 features will help in the classification

# #### Method 2: Using Information Gain approach

# In[211]:


from sklearn.feature_selection import mutual_info_classif as mic
get_ipython().run_line_magic('matplotlib', 'inline')

importances = mic(Features, label,n_neighbors=18)
feat_importances = pd.Series(importances, Features.columns)
feat_importances.plot(kind='barh', color='teal')
plt.show()


# Mutual Gain is not really showing any relevant information and thus not useful

# ### Method 4: Principal Compnent Analysis (PCA)

# In[212]:


pca_mod = skde.PCA()
pca_comps = pca_mod.fit(X_train)
pca_comps


# In[213]:


print(pca_comps.explained_variance_ratio_)
print(np.sum(pca_comps.explained_variance_ratio_))


# In[214]:


# Drawing a scree plot to tell how many features are really relevant

def plot_explained(mod):
    comps = mod.explained_variance_ratio_
    x = range(len(comps))
    y = [y + 1 for y in x]
    plt.plot(x, comps)
    
plot_explained(pca_comps)


# #### Interpretation of the scree plot
# 
# From the scree plot (Cattell 1966) above,we focus on the point where there is a sharp decline in size of the eigenvalues. When the eigenvalues drop dramatically in size, an additional factor would add relatively little to the information already extracted.
# 
# From the graph, we will choose 18 features for the machine learning model

# In[215]:


# pca_choose is the variable defined for the number of components
pca_choose = skde.PCA(n_components = 18)
pca_choose.fit(X_train)
Comps = pca_choose.transform(X_train) # Comps is the transformed X_train we will use to train the model
Comps.shape


# In[216]:


pd.DataFrame(Comps)


# ## Machine Learning models
# 
# ### Using PCA for the model training
# 
# #### 1. Using XGBoost

# In[217]:


import xgboost as xgb


# In[218]:


xgb_model = xgb.XGBClassifier(n_estimators = 1200, max_depth = 6)
xgb_model


# In[219]:


xgb_model.fit(Comps, y_train)


# In[220]:


y_pred = xgb_model.predict(pca_choose.transform(X_test))
predictions = [round(value) for value in y_pred]
print(y_test, y_pred)


# In[221]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[222]:


from sklearn. metrics import classification_report

print(classification_report(y_test, predictions))


# In[223]:


from sklearn.metrics import confusion_matrix

# Print the confusion matrix
cm = confusion_matrix(y_test, predictions)
print (cm)


# In[224]:


from sklearn.metrics import roc_curve
# calculate ROC curve
y_scores = xgb_model.predict_proba(pca_choose.transform(X_test))
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])

# plot ROC curve
fig = plt.figure(figsize=(6, 6))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# In[225]:


## Matthews Correlation coefficient
from sklearn.metrics import matthews_corrcoef


mcc = matthews_corrcoef(y_test, y_pred) 
mcc


# #### 2. Logistic Regression with cost-sensitive classification
# 
# * This idea of the cost-sensitive classification is to take care of the class imbalance.
# * First, we will determine the best parameter for C and 
# * Choose "balanced" for the class_weight arguement to handle the class imbalance

# In[78]:


# Estimating the best estimator for the regularization parameter C
logistic_mod = linear_model.LogisticRegression(class_weight = "balanced", solver='liblinear')
nr.seed(123)
inside = ms.KFold(n_splits=10, shuffle=True)
nr.seed(321)
outside = ms.KFold(n_splits=10, shuffle=True)
nr.seed(3456)
param_grid = {"C": [0.1, 1, 10, 100]}
clf = ms.GridSearchCV(estimator = logistic_mod, param_grid = param_grid,
                     cv = inside,
                     scoring = 'roc_auc',
                     return_train_score = True)
clf.fit(X, y)
clf.best_estimator_.C # The best regularization parameter for "C" is 10


# In[79]:


logistic_mod = linear_model.LogisticRegression(C=10, class_weight = "balanced", solver='liblinear')
logistic_mod.fit(Comps, y_train)
print(logistic_mod.intercept_)
print(logistic_mod.coef_)


# In[80]:


y_pred = logistic_mod.predict(pca_choose.transform(X_test))
predictions = [round(value) for value in y_pred]
print(y_test, y_pred)


# In[81]:


accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[82]:


print(classification_report(y_test, predictions))
cm = confusion_matrix(y_test, predictions)
print (cm)


# In[83]:


# calculate ROC curve
y_scores = logistic_mod.predict_proba(pca_choose.transform(X_test))
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])

# plot ROC curve
fig = plt.figure(figsize=(6, 6))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# In[84]:


## Matthews Correlation coefficient
mcc = matthews_corrcoef(y_test, y_pred) 
mcc


# #### 3. Random Forest Classifier 

# In[85]:


from sklearn.ensemble import RandomForestClassifier
# Choosing the best parameter for the classifier
param_grid = {"max_features":[5,10, 15,18,20], "min_samples_leaf":[3,5,10,20]}
nr.seed(3456)
rf_clf = RandomForestClassifier(class_weight = "balanced") # balanced class weight will handle the class imbalance
nr.seed(4455)
rf_clf = ms.GridSearchCV(estimator = rf_clf, param_grid = param_grid, 
                        cv = inside,
                        scoring = 'roc_auc',
                        return_train_score=True)
rf_clf.fit(X,y)
print(rf_clf.best_estimator_.max_features)
print(rf_clf.best_estimator_.min_samples_leaf) # Ramdom Forest confirms the number of features to use


# From the result, Random Forest selected 18 to be the maximum features to select. This confirms the results from the scree diagram. 

# In[86]:


rforest = RandomForestClassifier(max_features=18, class_weight = "balanced", min_samples_leaf = 5)
rforest.fit(Comps, y_train)


# In[87]:


y_pred = rforest.predict(pca_choose.transform(X_test))
predictions = [round(value) for value in y_pred]
print(y_test, y_pred)


# In[88]:


accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[89]:


print(classification_report(y_test, predictions))
cm = confusion_matrix(y_test, predictions)
print (cm)


# In[90]:


# calculate ROC curve
y_scores = rforest.predict_proba(pca_choose.transform(X_test))
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])

# plot ROC curve
fig = plt.figure(figsize=(6, 6))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# In[91]:


## Matthews Correlation coefficient
mcc = matthews_corrcoef(y_test, y_pred) 
mcc


# # Working with the test dataset

# In[92]:


# Import test data set
test_data = pd.read_csv('CW_test.csv')
test_data.head()


# ## Data preparation

# In[93]:


#Step_1: Get relevant features, which mirrors cw_relevant in the training set
test_columns = ['feat_seq_entropy', 'feat_C_atoms', 'feat_H_atoms', 'feat_N_atoms',
       'feat_O_atoms', 'feat_molecular_weight', 'feat_Perc_Tiny',
       'feat_Perc_Small', 'feat_Perc_Aliphatic', 'feat_Perc_Aromatic',
       'feat_Perc_NonPolar', 'feat_Perc_Polar', 'feat_Perc_Charged',
       'feat_Perc_Basic', 'feat_Perc_Acidic', 'feat_PP1', 'feat_PP2',
       'feat_PP3', 'feat_KF1', 'feat_KF2', 'feat_KF3', 'feat_KF4', 'feat_KF5',
       'feat_KF6', 'feat_KF7', 'feat_KF8', 'feat_KF9', 'feat_KF10', 'feat_Z1',
       'feat_Z2', 'feat_Z3', 'feat_Z4', 'feat_Z5', 'feat_F1', 'feat_F2',
       'feat_F3', 'feat_F4', 'feat_F5', 'feat_F6', 'feat_T1', 'feat_T2',
       'feat_T3', 'feat_T4', 'feat_T5', 'feat_VHSE1', 'feat_VHSE2',
       'feat_VHSE3', 'feat_VHSE4','feat_VHSE5', 'feat_VHSE6', 'feat_VHSE7', 'feat_VHSE8', 'feat_ProtFP1',
       'feat_ProtFP2', 'feat_ProtFP3', 'feat_ProtFP4', 'feat_ProtFP5',
       'feat_ProtFP6', 'feat_ProtFP7', 'feat_ProtFP8', 'feat_ST1', 'feat_ST2',
       'feat_ST3', 'feat_ST4', 'feat_ST5', 'feat_ST6', 'feat_ST7', 'feat_ST8',
       'feat_BLOSUM1', 'feat_BLOSUM2', 'feat_BLOSUM3', 'feat_BLOSUM4',
       'feat_BLOSUM5', 'feat_BLOSUM6', 'feat_BLOSUM7', 'feat_BLOSUM8',
       'feat_BLOSUM9', 'feat_BLOSUM10', 'feat_MSWHIM1', 'feat_MSWHIM2',
       'feat_MSWHIM3','feat_Perc_A', 'feat_Perc_D', 'feat_Perc_E', 'feat_Perc_G',
       'feat_Perc_I', 'feat_Perc_K', 'feat_Perc_L', 'feat_Perc_N',
       'feat_Perc_P', 'feat_Perc_Q', 'feat_Perc_R', 'feat_Perc_S',
       'feat_Perc_T', 'feat_Perc_V']


# In[94]:


features.shape


# In[95]:


test_df = test_data[test_columns].copy()
test_df.head()


# In[96]:


# Step two: Log columns with high values
test_df[['feat_C_atoms','feat_H_atoms','feat_N_atoms','feat_O_atoms','feat_molecular_weight']] = test_df[['feat_C_atoms','feat_H_atoms','feat_N_atoms','feat_O_atoms','feat_molecular_weight']].apply(lambda x: np.log10(x))
test_df.head()


# In[97]:


# Step 3: Scale all features with the StandardScaler and call it Test_df

Test_df = pd.DataFrame(scaler.fit_transform(test_df), columns = test_df.columns)
# Features[:, 0:10]
Test_df.head()


# In[104]:


# Step 4: Use PCA on the test set and call it Test_transformed
# pca_mod = skde.PCA()
# pca_comps = pca_mod.fit(X_train)
# pca_compstes
Test_transformed = pca_choose.transform(Test_df)


# In[105]:


Test_transformed.shape


# In[106]:


# Step 5: Predict using the XGBoost Classifier specifically call it pred_label
pred_label = xgb_model.predict(Test_transformed)


# In[107]:


pred_label.shape


# In[108]:


test_df.shape


# In[117]:


pred_label_df = pd.DataFrame(pred_label, columns=["Prediction"])
pred_label_df


# In[118]:


# Merging the required columns with the the predicted column
cw_test_final = pd.concat([test_data['Info_PepID'],test_data['Info_center_pos'], pred_label_df['Prediction']], join = 'outer', axis=1)
cw_test_final.head()


# In[120]:


cw_test_final.to_csv('cw_test_final.csv')


# In[ ]:




