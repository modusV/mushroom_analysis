#%% [markdown]

# # Safe to eat or deadly poisonous?
# ### An analysis on mushroom classification by Lorenzo Santolini
#
# ### Index:
#   1. [Introduction](#introduction)
#   2. [Dataset Analysis](#dsl)
#   3. [Preprocessing](#preprocessing)
#   4. [Principal Component Analysis](#pca)
#   5. [Classification](#classification)
#   6. [Conclusions](#conclusions)
#%% [markdown]
# This is a little code to import automatically the dataset into google colab. 
# Provide your kaggle's API key (profile section) when file requested

# ### Code snippet for google colab

#%%
# Little code snippet to import on Google Colab the dataset
'''
!pip install -U -q kaggle
!mkdir -p ~/.kaggle

# Insert here your kaggle API key
from google.colab import files
files.upload()

!cp kaggle.json ~/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json
!kaggle datasets download -d uciml/mushroom-classification
!unzip mushroom-classification.zip
!ls
'''

#%% 
# Define all the constants that will be used

PLOTLY_COLORS = ['#140DFF', '#FF0DE2']
COLOR_PALETTE = ['#140DFF', '#FF0DE2', '#CAFFD0', '#C9E4E7', '#B4A0E5', '#904C77']
COLORSCALE_HEATMAP = [         [0.0, 'rgb(70,0,252)'], 
                [0.1111111111111111, 'rgb(78,0,252)'], 
                [0.2222222222222222, 'rgb(90,0,252)'], 
                [0.3333333333333333, 'rgb(110,0,248)'], 
                [0.4444444444444444, 'rgb(130,0,238)'], 
                [0.5555555555555556, 'rgb(145,0,228)'], 
                [0.6666666666666666, 'rgb(166,0,218)'], 
                [0.7777777777777778, 'rgb(187,0,213)'], 
                [0.8888888888888888, 'rgb(200,0,202)'], 
                               [1.0, 'rgb(210,0,191)']]
PLOTLY_OPACITY = 0.7
RANDOM_SEED = 11

LOGISTIC_REGRESSION_PARAMS = [{
    'clf__solver': ['liblinear'],  # best for small datasets
    'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], # smaller value, stronger regularization, like svm
    'clf__penalty': ['l2', 'l1']
},
{
    'clf__solver': ['newton-cg', 'lbfgs'], 
    'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'clf__penalty': ['l2'] # `newton-cg` and `lbfgs` accept only l2
}]

SVM_PARAMS = [
{
    'clf__kernel': ['linear'],
    'clf__C': [0.1, 1, 10, 100],
}, 
{
    'clf__kernel': ['rbf'],
    'clf__C': [0.1, 1, 10, 100],
    'clf__gamma': [0.1, 1, 10, 100],
}]

RANDOM_FOREST_PARAMS = {
    'clf__max_depth': [50, 75, 100],
    'clf__max_features': ["sqrt", "log2"], # sqrt is the same as auto
    'clf__criterion': ['gini', 'entropy'],
    'clf__n_estimators': [100, 300, 500]
}

KNN_PARAMS = {
    'clf__n_neighbors': [2, 3, 5, 15, 25],
    'clf__weights': ['uniform', 'distance'],
    'clf__p': [1, 2, 10]
}


#%% [markdown]
# <a id='introduction'></a>
## Introduction
# This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom. Each species is identified as edible or poisonous. Rows are composed by 23 different fields, each one of them identifying a specific charateristic:

# - `Class`: poisonous=p, edible=e
# - `Cap-surface`: fibrous=f, grooves=g, scaly=y, smooth=s
# - `Cap-shape`: bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s
# - `Cap-color`: brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y
# - `Bruises`: bruises=t, no=f
# - `Odor`: almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s
# - `Gill-attachment`: attached=a, descending=d, free=f, notched=n
# - `Gill-spacing`: close=c, crowded=w, distant=d
# - `Gill-size`: broad=b, narrow=n
# - `Gill-color`: black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o, pink=p, purple=u,red=e, white=w, yellow=y
# - `Stalk-shape`: enlarging=e, tapering=t
# - `Stalk-root`: bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=?
# - `Stalk-surface-above-ring`: fibrous=f, scaly=y, silky=k, smooth=s
# - `Stalk-surface-below-ring`: fibrous=f, scaly=y, silky=k, smooth=s
# - `Stalk-color-above-ring`: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y
# - `Stalk-color-below-ring`: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y
# - `Veil-type`: partial=p, universal=u
# - `Veil-color`: brown=n, orange=o, white=w, yellow=y
# - `Ring-number`: none=n, one=o, two=t
# - `Ring-type`: cobwebby=c, evanescent=e, flaring=f, large=l, none=n, pendant=p, sheathing=s, zone=z
# - `Spore-print-color`: black=k, brown=n, buff=b, chocolate=h, green=r, orange=o, purple=u, white=w, yellow=y
# - `Population`: abundant=a, clustered=c, numerous=n, scattered=s, several=v, solitary=y
# - `Habitat`: grasses=g, leaves=l, meadows=m, paths=p, urban=u, waste=w, woods=d  
# 
# This analysis was conducted in *Python 3.7.1* using Jupyter Notebook allows you to combine code, 
# comments, multimedia, and visualizations in an interactive document — called a notebook, 
# naturally — that can be shared, re-used, and re-worked.  
#
# In addition, the following packages were used:  

# - sklearn
# - pandas
# - numpy
# - plotly
# - scipy
# - prettytable
# - imblearn

#%%
# Import all the libraries

import pandas as pd
import numpy as np
import math

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, learning_curve, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, confusion_matrix, roc_curve, auc, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle

import plotly
import plotly.plotly as py
from plotly.plotly import plot, iplot
import plotly.graph_objs as go
import plotly.figure_factory as ff

from scipy.cluster import hierarchy as hc
import scipy.spatial as scs
import scipy.stats as ss

from imblearn.pipeline import make_pipeline, Pipeline

import warnings
from collections import defaultdict
from collections import Counter
from prettytable import PrettyTable
from functools import wraps
import time

plotly.tools.set_credentials_file(username='modusV', api_key='R8hqiaiaxOXvVkwIcXFU')
warnings.filterwarnings("ignore")
#%%

# Wrapper to calculate functions speed

def watcher(func):
    """
    Shows how much time it
    takes to execute function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f" ===> took {end-start} seconds")
        return result
    return wrapper

#%% [markdown]
# <a id='dsl'></a>
## Dataset load and overall view
# Let's start importing the data:

#%%
# Load the dataset
dataset = pd.read_csv("./Input/mushrooms.csv")
# dataset = pd.read_csv("./mushrooms.csv")

#%%
# Shape of the dataset
print("The dataset has %d rows and %d columns." % dataset.shape)

#%% [markdown]
# We will look now at the dataset to understand what are the different fields and their types:
#%%
# Count number of classes for classification
print(f"There are {dataset['class'].unique().size} different classes:"
      f"\n {dataset['class'].unique().tolist()}")

# Count number of unique data for every column
print(f"Unique values for every field: \n{dataset.nunique()}")

#%% [markdown]
# We can notice that `veil-type` has just one value, therefore that column is useless
# for our analysis. We will remove it later.  
# 
# We can now look deeper inside the dataset. Thanks to the pandas library, we can 
# see all the fields of the dataset with the respective values.

#%% [markdown]
# <a id='preprocessing'></a>
## Preprocessing

# Before starting the classification phase, we need to preprocess the dataset, in 
# such a way that our classifiers will score with more accuracy and reliability. 
# This is the most important step, if data are messy the classification will perform 
# poorly.

# The steps that we will go trough are:  
# 
# 1. Check data types  
# 2. Remove not significat columns, if any  
# 3. Remove null values and duplicates, if any
# 4. Encode string values
# 5. Check class distribution and, if classes are unbalanced, apply balancing techniques
# 6. Check data distribution using of bar graphs and box plots
# 7. Analyze correlation matrix to understand which fields are more important to classify our samples
# 8. Divide the dataset in classes array and unclassified samples
# 9. Scale our data, in such a way to center and standardize them

#%% [markdown]
# ### 1 - Check data types
#%%
# See data types 
print("Data types:")
dataset.head(5)

#%% [markdown]
# From the above snippet we can notice that the fields are all string values; 
# converting them to numeric values will make our analysis much easier. We will use a
# library called `LabelEncoder`. It allows us with a few line of code to create a mapping
# of every value in each field and transform the data in this way. We can go back to the 
# original mapping simply using the `inverse_transform` function.  
#
# Before this step though, we will firstly remove all the useless columns, in this 
# case just `veil-type`.

#%% [markdown]
# ### 2 - Remove any not significant column
# Now we will remove all the fields that do not add any information to our analysis,
# specifically, the fields that contain only one value (zero variance).
#%% 
n_columns_original = len(dataset.columns)
to_drop = [col for col in dataset.columns if dataset[col].nunique() == 1]
dataset.drop(to_drop, axis=1, inplace=True)

for d in to_drop:
    print(str(d) + " ", end="")
print("have been removed because they have zero variance")
print(f"{n_columns_original - len(dataset.columns)} not significant columns have been removed")

#%% [markdown]
# As we can notice, only one field was removed. 

#%% [markdown]
# ### 3 - Handling missing values
# When we find any missing value in a dataset, there are different 
# approaches that can be considered:  
#
# 1. Delete all rows containing a missing value
# 2. Substitute with a constant value that has meaning within the domain, such as 0, distinct from all other values.
# 3. Substitute with a value from another randomly selected record.
# 4. Substitute with mean, median or mode value for the column.
# 5. Substitute with a value estimated by another predictive model.  
#
# 
# We will approach this problem using the first and the second techniques:
# 
# 1. We will create a parallel dataset in which all the rows containing a missing 
# value will be dropped, classifying 
# them as incomplete samples. This may cause a large decrement in the dataset size
# but is the only way
# to be sure that we are not going to influence our classification algorithm in any
# way. 
# 
# 2. It is evident from the `dataset.head()` function that our fileds are composed
# by all string values. 
# Given the fact that we would need to translate in any case every field to a
# numeric one, to better display 
# them in graphs, a simple approach is to keep the missing data as a peculiar number
# different from the others,
# and simply apply the transformation as they were present.
# 
#
# In any case, let's start counting how many null/missing values we will find.
#%%
# Check if any field is null
if dataset.isnull().any().any():
    print("There are some null values")
else:
    print("There are no null values")
#%% [markdown]
# It may seem that we have no missing value from the previous analysis... Great!
# 
# But wait a minute ... If we look better,from the data description we can notice that in the field 
# `stalk-root` there are some missing values, marked with the question mark; let's count how many of them there are:

#%%
print("There are " + str((dataset['stalk-root'] == "?").sum()) + " missing values in stalk-root column")

#%% [markdown]
# More than 25% of our samples is incomplete. Dropping all those rows may lead to a shortage of samples.
# This is why we will use the two approaches and see which one performs better.
# In any case, we will replace those value with another character. 
# We will use the character `m` to indicate a missing value

#%%

dataset['stalk-root'] = dataset['stalk-root'].replace({"?": "m"})
print(dataset['stalk-root'].unique())

#%% [markdown]
# Beore we continue, let's check if there are any duplicates in our data - meaning, 
# do we know of two or more mushrooms with exactly the same features?

#%%
print('Known mushrooms: {}\nUnique mushrooms: {}'.format(len(dataset.index),len(dataset.drop_duplicates().index)))
#%% [markdown]
# Perfect! But wait, are there any equal mushrooms but with different class?

#%%
print('Known mushrooms: {}\nMushrooms with same features: {}'.format(
    len(dataset.index),
    len(dataset.drop_duplicates(subset=dataset.drop(['class'], axis=1)
    .columns).index)))
#%% [markdown]
# Very good, two mushrooms with the same features having different classes may be really
# dangerous. 
#%% [markdown]
# ### 4 - Encode string values
# As already said, we need to encode all the string values into integers, in such a way to continue our analysis in a more easy way. 
#%%

def encode_values(dataset):
    """
    Encode string values of a dataset using numbers

    :param (array of arrays) dataset: Input dataset to encode
    """
    mapping = {}  
    d = dataset.copy()
    labelEncoder = LabelEncoder()
    for column in dataset.columns:
        labelEncoder.fit(dataset[column])
        mapping[column] = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
        d[column] = labelEncoder.transform(dataset[column])
        
    return d, labelEncoder, mapping

def print_encoding(mapping):
    """
    Prints a table with the key-value corrispondence of an encoding

    :param (dict) mapping: dictionary containing value before and after encoding
    """
    t = PrettyTable()
    rows = []
    for key, value in mapping.items():
        r = []
        r.append(key)
        for k, v in value.items():
            r.append(k)
        rows.append(r)
    max = []
    for r in rows:
        if len(r) > len(max):
            max = r

    for r in rows:
        r = r + ['-'] * (len(max) - len(r))
        t.add_row(r)
    t.field_names = ["Columns / Values"] + list(range(0, len(max)-1))
    print(t)

le = 0
pre_data, l_encoder, le_mapping = encode_values(dataset)

# Check mapping
print_encoding(le_mapping)

# Check new data
pre_data.head(5)

#%% [markdown]
# As we can see data have been transformed; all the strings values now are equal to integers, 
# and we can see the direct corrispondence from the table above.
#
# We have encoded our dataset but, depending on the data (as in our case), label encoding may 
# introduce a new problem. For example, we have encoded a set of colour names 
# into numerical data. This is actually categorical data and there is no relation, 
# of any kind, between the rows. 
# 
# The problem here is, since there are different numbers in the same column, 
# the model will misunderstand the data to be in some kind of order, 0 < 1 < 2. 
# But this isn’t the case at all. To overcome this problem, we use `OneHotEncoder`.
#
# What one hot encoding does is, it takes a column which has categorical data, 
# which has been label encoded, and then splits the column into multiple columns. 
# The numbers are replaced by 1s and 0s, depending on which column has what value.
# 
# We will obtain two datasets; the one Label Encoded contains values that have no 
# geometrical interpretation (we cannot use classified based on distances such as kNN).
# The other one, on the other hand, will be bigger but with more data significance.
#
# We also drop the column with missing value, beacuse it is not significative.
#%%

def one_hot_encode(X_dataset):

    ohc = defaultdict(OneHotEncoder)
    d = defaultdict (LabelEncoder)
    Xfit = X_dataset.apply(lambda x: d[x.name].fit_transform(x))
    final = pd.DataFrame()

    for i in range(len(X_dataset.columns)):
        # Transform columns using OneHotEncoder
        Xtemp_i = pd.DataFrame(ohc[Xfit.columns[i]].fit_transform(Xfit.iloc[:,i:i+1]).toarray())
    
        # Naming the columns
        ohc_obj  = ohc[Xfit.columns[i]]
        labelEncoder_i= d[Xfit.columns[i]]
        Xtemp_i.columns= Xfit.columns[i]+"-"+labelEncoder_i.inverse_transform(ohc_obj.active_features_)
        
        # Take care of dummy variable trap dropping of new coulmns
        X_ohc_i = Xtemp_i.iloc[:,1:]
        
        # Append columns to dataframe
        final = pd.concat([final,X_ohc_i],axis=1)

    return final

pre_ohc_data = one_hot_encode(dataset.iloc[:,1:])
pre_ohc_data.drop(['stalk-root-m'], axis=1, inplace=True)
pre_ohc_data.head(5)


#%% [markdown]
# As we can see, we obtained a huge dataset, where all the fields are binary.

#%% [markdown]

# ### 5/6 - Check class and data distribution
#In this phase, we will analyze the distribution of the data. The steps will be:
#
# 1. Check amount of samples belonging to a class or to another.
# 2. Analyze the overall distribution using box plots, a very useful tool to identify outliers and values taken by samples. 
# 3. Compare the two classes distributions with the help of an histogram.

#%% [markdown]
# #### 5 - Check classes distribution
# Let's see how many samples belong to the different classes
#%%
y = dataset["class"].value_counts()
print(y)
class_dict = ["edible", "poisonous"]

#%% [markdown]
# Luckily the dataset is pretty balanced: 
# We have almost the same amount of samples in a class and in the other; this simplifies the analysis because we can assign the same weight to the two classes in the classification phase.
# 
# Moreover, we can notice that the classification task will be binary. Infact, data have been transformed, and now the labels are represented with a 0/1 integer value. 
# Now we can look deeper into some statistical details about the dataset, using the `pre_df.describe()` command on our pandas DataFrame dataset. The output shows:
# 
# - count: number of samples (rows)
# - mean: the mean of the attribute among all samples
# - std: the standard deviation of the attribute
# - min: the minimal value of the attribute
# - 25%: the lower percentile
# - 50%: the median
# - 75%: the upper percentile
# - max: the maximal value of the attribute
#%%
# Get insights on the dataset
pre_data.describe()

#%% [markdown]
# This is the class distribution plot on a graph bar. As we already saw before the class distribution is pretty balanced. 
# 
# In this report, all the graphic part will use the `plotly` library. 
# 
# *plotly.py* is an interactive, open-source, and browser-based graphing library for Python, which allows you to create interactive plots in a few steps.

#%%

def create_bar(type, data, col, visible=False):
    if type == "edible":
        c = PLOTLY_COLORS[0]
    else:
        c = PLOTLY_COLORS[1]
    return go.Histogram(
        x = data[col],
        name = type,
        marker=dict(color = c),
        visible=visible,
        opacity=PLOTLY_OPACITY,
    )

def feature_histogram(data, feature):
    
    trace1 = create_bar("edible", data[data['class'] == 'e'], feature, True)
    trace2 = create_bar("poisonous", data[data['class'] == 'p'], feature, True)

    data = [trace1, trace2]

    layout = dict(
        autosize=True,
        yaxis=dict(
            title='value',
            automargin=True,
        ),
        legend=dict(
            x=0,
            y=1,
        ),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1
    )
    fig = dict(data=data, layout=layout)
    return py.iplot(fig, filename='bar_slider')
#%%
feature_histogram(dataset, "class")

#%% [markdown]
# #### 6 - Box plot
# 
# At this point we can analyze the distribution of our data using a boxplot. 
# A boxplot is a standardized way of displaying the distribution of data based on a five number summary (“minimum”, first quartile (Q1), median, third quartile (Q3), and “maximum”). 
#
# It can tell you about your outliers and what their values are. 
# It can also tell you if your data is symmetrical, how tightly your data is grouped,and if and how your data is skewed.
# The information that we can find in a box plot are:
# 
# - **median** (Q2/50th Percentile): the middle value of the dataset.
# - **first quartile** (Q1/25th Percentile): the middle number between the smallest number (not the “minimum”) and the median of the dataset.
# - **third quartile** (Q3/75th Percentile): the middle value between the median and the highest value (not the “maximum”) of the dataset.
# - **interquartile range** (IQR): 25th to the 75th percentile.
# - **outliers** (shown as green circles)
# - **maximum**: Q3 + 1.5*IQR
# - **minimum**: Q1 -1.5*IQR
#
# It makes no sense showing binary or with few different values fields, so we are going to filter them before plotting.

#%%

def create_box(type, data, col, visible=False):
    if type == "edible":
        c = PLOTLY_COLORS[0]
    else:
        c = PLOTLY_COLORS[1]
    return go.Box(
        y = data[col],
        name = type,
        marker=dict(color = c),
        visible=visible,
        opacity=PLOTLY_OPACITY,
    )

edible = pre_data[pre_data["class"] == 0]
poisonous = pre_data[pre_data["class"] == 1]
box_features = [col for col in pre_data.columns if ((col != 'class') and (dataset[col].nunique() > 5))]

active_index = 0
box_edible = [(create_box("edible", edible, col, False) if i != active_index 
               else create_box("edible", edible, col, True)) 
              for i, col in enumerate(box_features)]

box_poisonous = [(create_box("poisonous", poisonous, col, False) if i != active_index 
               else create_box("poisonous", poisonous, col, True)) 
              for i, col in enumerate(box_features)]

data = box_edible + box_poisonous
n_features = len(box_features)
steps = []

for i in range(n_features):
    step = dict(
        method = 'restyle',  
        args = ['visible', [False] * len(data)],
        label = box_features[i],
    )
    step['args'][1][i] = True # Toggle i'th trace to "visible"
    step['args'][1][i + n_features] = True # Toggle i'th trace to "visible"
    steps.append(step)
    
sliders = [dict(
    active = active_index,
    currentvalue = dict(
        prefix = "Feature: ", 
        xanchor= 'center',
    ),
    pad = {"t": 50},
    steps = steps,
    len=1,
)]

layout = dict(
    sliders=sliders,
    autosize=True,
    yaxis=dict(
        title='value',
        automargin=True,
    ),
    legend=dict(
        x=0,
        y=1,
    ),
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='box_slider')

#%% [markdown]

# From the boxplot above, we can see that the color and the shape of the 
# cap are not an effective parameter to decide whether a mushroom is 
# poisonous or edible, because their plots are very similar (same median 
# and very close distribution). 
# The `odor` and the `population` columns, on the other hand, are more significant; 
# 
# In the `odor` field, all the edible mushrooms are squeezed into a single value
# with a few outliers, while the poisonous may have all the different values.

#%% [markdown]
# #### 6 - Bar graph 
# 
# A bar chart or bar graph is a chart or graph that presents categorical data with rectangular bars with heights or lengths proportional to the values that they represent.
# With a slider we can move along the different features, to better visualize the value distributions.

#%%

hist_features = [col for col in pre_data.columns if (col != 'class')]

active_index = 0
hist_edible = [(create_bar("edible", edible, col, False) if i != active_index 
               else create_bar("edible", edible, col, True)) 
              for i, col in enumerate(hist_features)]

hist_poisonous = [(create_bar("poisonous", poisonous, col, False) if i != active_index 
               else create_bar("poisonous", poisonous, col, True)) 
              for i, col in enumerate(hist_features)]

total_data = hist_edible + hist_poisonous
n_features = len(hist_features)
steps = []

for i in range(n_features):
    step = dict(
        method = 'restyle',  
        args = ['visible', [False] * len(total_data)],
        label = hist_features[i],
    )
    step['args'][1][i] = True # Toggle i'th trace to "visible"
    step['args'][1][i + n_features] = True # Toggle i'th trace to "visible"
    steps.append(step)
    
sliders = [dict(
    active = active_index,
    currentvalue = dict(
        prefix = "Feature: ", 
        xanchor= 'center',
    ),
    pad = {"t": 50},
    steps = steps,
)]

layout = dict(
    sliders=sliders,
    autosize=True,
    yaxis=dict(
        title='value',
        automargin=True,
    ),
    legend=dict(
        x=0,
        y=1,
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1
)

fig = dict(data=total_data, layout=layout)
py.iplot(fig, filename='bar_slider')

#%% [markdown]
# As we saw from the box plot, also here we can notice that
# the `cap-shape` and the `cap-color` are not that significant to classify a sample. 
# On the other hand, some fields like `odor` show a distinct 
# separation of the two classes; these ones will be the ones with more 
# impact on our classification algorithms.
# 
# From these bar graphs we can see that our dataset is pretty well 
# separated. This means that, in our classification task, we will be 
# able to achieve high accuracy even with some dimensionality reduction.

#%% [markdown]
# ### 7 - Correlation matrix
# A correlation matrix is a table showing correlation coefficients between sets of variables. 
# Each random variable (Xi) in the table is correlated with each of the other values in the table (Xj). 
# This allows you to see which pairs have the highest correlation. Correlation is any statistical association, 
# though in common usage it most often refers to how close two variables are to having 
# a linear relationship with each other.
# 
# We will use the **Pearson's correlation**, which is a measure of the linear correlation 
# between two variables X and Y. According to the Cauchy–Schwarz inequality it has a value 
# between +1 and −1, where 1 is total positive linear correlation, 0 is no linear correlation, 
# and −1 is total negative linear correlation.
# 
# The coefficient for a population is computed as:
# 
# $\rho_{(X,Y)} = \frac{cov(X,Y)}{\sigma_X\sigma_Y}$
# 
# Where:
# - $cov$ is the covariance:
#   - $cov(X,Y) = \frac{E[(X - \mu_X)(Y-\mu_Y)]}{\sigma_X\sigma_Y}$
#    - Where $\mu$ is the mean value  
# 
# - $\sigma_X$ is the standard deviation of X
#   - $\sigma_X^2 = E[X^2] - [E[X]]^2$
# - $\sigma_Y$ is the standard deviation of Y
#   - $\sigma_Y^2 = E[Y^2] - [E[Y]]^2$

#%%

def plot_correlation_matrix(matrix):

    z_text = np.around(matrix.values.tolist(), decimals=2)

    figure = ff.create_annotated_heatmap(z=matrix.values, 
                                         x=matrix.columns.tolist(), 
                                         y=matrix.index.tolist(),
                                         annotation_text=z_text,
                                         colorscale=COLORSCALE_HEATMAP,
                                         showscale=True)

    figure.layout.title = 'Heatmap of columns correlation'
    figure.layout.autosize = False
    figure.layout.width = 850
    figure.layout.height = 850
    figure.layout.margin = go.layout.Margin(l=140, r=100, b=200, t=80)
    figure.layout.xaxis.update(side='bottom')
    figure.layout.yaxis.update(side='left')

    for i in range(len(figure.layout.annotations)):
        figure.layout.annotations[i].font.size = 8
                                    
    return py.iplot(figure, filename='labelled-heatmap4')

def plot_correlation_row(matrix, key):
    
    matrix = pd.Series.to_frame(matrix.loc['class']).transpose()
    z_text = np.around(matrix.values.tolist(), decimals=2)

    figure = ff.create_annotated_heatmap(z=matrix.values, 
                                         x=matrix.columns.tolist(), 
                                         y=matrix.index.tolist(),
                                         annotation_text=z_text,
                                         colorscale=COLORSCALE_HEATMAP,
                                         showscale=False)

    figure.layout.title = "Heatmap of " + key + " correlation"
    figure.layout.autosize = False
    figure.layout.width = 850
    figure.layout.height = 220
    figure.layout.xaxis.update(side='bottom')
    figure.layout.yaxis.update(side='left')

    for i in range(len(figure.layout.annotations)):
        figure.layout.annotations[i].font.size = 8
                                    
    return py.iplot(figure, filename='labelled-heatmap4')
#%%

correlation_matrix = pre_data.corr(method='pearson')
plot_correlation_matrix(correlation_matrix)
#%% [markdown]
# But wait, does this really makes sense? Didn't we see form the boxplot that the odor
# was very important to determine the class? Here it doesn't seem so ...
#
# This is because we performed a geometrical analysis on a dataset that has been encoded
# with arbitrary values! Looking at the Pearson's correlation formula, how do we subtract 
# *green* from the *average* of colors?
#
# To obtain a consistent result, we could plot the correlation matrix using the One Hot Encoded dataset, but the problem is that
# we would obtain a huge matrix showing correlation between every combination of values that
# our features may have, which is not that significative for our analysis.
#
# Due to all these problems, we will need something that looks like the correlation,
# but works with categorical features. 
# We need then a *measure of association* between features,
# but on contrary of a simple Pearson's correlation, we would need something that is not symmetrical,
# because otherwise we risk to lose information such as:
#
# - feature *x* implies label *y* does not imply that label *y* implies features *x*
#
#
# **Theil's U**, also referred as **Uncertanty Coefficient**, suits our situation perfectly.
# It is based on the conditional entropy between the variables *x* and *y*; in other words, 
# given the value *x*, shows how many states does *y* have, and how often they occurr.
# Its output is between $[0,1]$ and it is asymmetric ( $U(x,y) \ne U(y,x)$ )
#
# The coefficient is defined as:
#
# - $U(X|Y) = \frac{H(X) - H(X|Y)}{H(X)}$
# - where:
#   - $H(X) = -\Sigma_x{P_X(x) logP_X(x)}$
#   - $H(X|Y) = -\Sigma_{x, y}{P_{X,Y}(x, y) log{P_{X,Y}(x, y)}}$
#

#%%

def conditional_entropy(x, y):
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
    return entropy

def theil_u(x, y):
    s_xy = conditional_entropy(x, y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x

def create_theil_matrix(data):

    columns = data.columns
    matrix = []
    for col in columns:
        row = []
        for j in range(0, len(columns)):
            u = np.around(theil_u(data[col].tolist(), data[columns[j]].tolist()), decimals=4)
            row.append(u)
        matrix.append(row)
    
    matrix = np.array(matrix)
    i_lower = np.tril_indices(len(columns), -1)
    matrix[i_lower] = matrix.T[i_lower]

    matrix = pd.DataFrame(data=matrix,
                 columns=columns,
                 index=columns)
    return matrix

corr_dataset = create_theil_matrix(pre_data)
plot_correlation_matrix(corr_dataset)

#%% [markdown]
# Now we can see that the `odor` is highly correlated with the class, as we expected.
# The `gill-color` and `spore-print-color` are also quite correlated to our label field.
#
# We can also notice that `gill-attachment` is highly correlate with three other features;
# maybe we can reduce dimensionality losing a few information.
#
#
# Being the `odor` highly correlated with the class, we can see how many mushrooms are we able to classify 
# just by smelling them. 

#%%
plot_correlation_row(corr_dataset, "class")

#%%
feature_histogram(dataset, "odor")

#%% [markdown]
# Looking at the bar chart above, if it has an odor (different from `n`, which means none) 
# and it is not `a` or `l`, then it is edible!
# But what about the mushrooms that have no odor? Let's do the same trick, but with the mushrooms with
# no `odor`.

#%%
no_odor = dataset[dataset["odor"] == "n"]
corr_data_no_odor = create_theil_matrix(no_odor)
plot_correlation_row(corr_data_no_odor, "class")
#%% [markdown]
# Now the most helpful feature is `spore-print-color`! Let's look at the 
# value distribution in this case:

#%%
feature_histogram(no_odor, "spore-print-color")

#%% [markdown]
# We can easily see that only mushrooms with a `w` spore print color 
# may leave us in doubt. Wonderful, let's count what percentage of the dataset we have 
# classified in this simple way:

#%%
no_odor_w = no_odor[no_odor["spore-print-color"] == "w"]
print("We can determine the class of: %.2f%% mushrooms\n" % ((len(dataset.index) - len(no_odor_w.index))/len(dataset.index)))

#%% [markdown]
# To sum up, we can determine the class of 92% of our dataset in an easy way.
# If you are alone in a forest, you can safely eat mushrooms with almond or anise smell;
# If it has no smell, you can look at their `spore-print-color`;
# you can savour every one of them apart from the ones with a red print, which are certainly poisonous, and 
# the ones with a white print (0,08% probability of being poisonous, but you need to take into account 
# the probability of finding one on that kind in nature).
#
#
# After this point, to be 100% sure of the class of a mushroom, we request the help of 
# classification models.
# The last step before the automatic classification, will be to plot a dendogram,
# to see if we can further reduce the dimension of the dataset.
#%% [markdown]
# ### 7 - Dendogram
# A **dendrogram** is a diagram representing a tree. This diagrammatic representation is frequently used in different contexts, but we will see the case representing hierarchical clustering. 
#
# It illustrates the arrangement of the clusters, and its objective is to analyze if we have any duplicate features. In order to reduce the dimensionality of our dataset, we can identify and remove duplicate features according to their pairwise correlation with others.
# 
# The linkage criterion determines the distance between sets of observations as a function of the pairwise distances between observations.
# We will use the between-group average linkage (UPGMA). Proximity between two clusters is the arithmetic mean of all the proximities between the objects of one, on one side, and the objects of the other, on the other side.
# The method is frequently set the default one in hierarhical clustering packages.

#%% 

names = pre_data.columns
inverse_correlation = 1 - corr_dataset # This is the 'dissimilarity' method

fig = ff.create_dendrogram(inverse_correlation, 
                           labels=names, 
                           colorscale=COLOR_PALETTE, 
                           linkagefun=lambda x: hc.linkage(x, 'average'))

fig['layout'].update(dict(
    title="Dendrogram of correlation among features",
    width=800, 
    height=600,
    xaxis=dict(
        title='Features',
    ),
    yaxis=dict(
        title='Distance',
    ),
))
iplot(fig, filename='dendrogram_corr_clustering')

#%% [markdown]
# From the above graph, the closest features are `class` and `odor`, as we saw in the correlation matrix.
# We do not have any low distance clusters, and due to this we are going to keep all the features,
# being the dataset not that large.

#%% [markdown]
# ### 8/9 - Scale and divide data

# Most of the times, datasets contain features highly varying in magnitudes, units and range. 
# But since, most of the machine learning algorithms use Euclidean distance between two 
# data points in their computations, this is a problem. If left alone, these algorithms 
# only take in the magnitude of features neglecting the units. The results would vary greatly 
# between different units, for example between 5kg and 5000gms. 
# 
# The features with high magnitudes will weigh in a lot more in the 
# distance calculations than features with low magnitudes.
# To supress this effect, we need to bring all features to the same level of magnitudes. 
# This can be acheived by scaling.
# 
# We will use the `StandardScaler`, which standardizes our data both with mean and 
# standard deviation.
#
# The operation performed will be:
#
# $$
# x' = \frac{x - \mu_x}{\sigma_x}
# $$

# After this step, we divide the dataset into an array of unclassified samples 
# and an array of labels, to use for the classification phase.
# 
# At the end, we decide to bring to the next step of our analysis 
# different datasets:
# 
# 1. The full dataset Label Encoded, where the missing values are encoded with an integer.
# 2. A reduced version of the dataset, where the rows with missing data are considered as 
#   incomplete samples and are dropped.
# 3. The full dataset Label Encoded, with `stalk-root`field dropped.
# 4. The dataset One Hot Encoded
#%%

def dataframe_to_array(data):
    y_data = data['class']
    X_data = data.drop(['class'], axis=1)
    return X_data, y_data

def scale_data(X_data):
    """
    Scales data using mean and std

    :param (array) pca: Array of data
    """
    scaler = StandardScaler(with_mean=True, with_std=True, copy=True)
    return scaler.fit_transform(X_data)

#%% [markdown]
# Before dropping the samples containing a missing value, we will check how many of them belong to a class or to the other;
# we perform this step because it is possible that we unintentionally remove most of the samples from one of the two classes
# creating some sort of imbalance within the data.
#%%
edible_removed = np.sum(dataset[dataset['stalk-root'] == "?"]['class'] == 'e')
poisonous_removed = np.sum(dataset[dataset['stalk-root'] == "?"]['class'] == 'p')
print("Will be removed:\n"
      f"\t- {edible_removed} edible samples\n"
      f"\t- {poisonous_removed} poisonous samples")

#%% [markdown]
# We can notice that our missing values are mainly in the poisonous samples. 
# Removing them may create an unbalance in the number of rows in the two classes, 
# creating issues in some classifiers (e.g. Naive Bayes). 
#%%

drop_data = pre_data[pre_data['stalk-root'] != le_mapping['stalk-root']['m']]
data_no_stalk = pre_data.drop(['stalk-root'], axis=1)

X_pre_data, y_data = dataframe_to_array(pre_data)
X_scaled_data = scale_data(X_pre_data)

X_drop_data, y_drop_data = dataframe_to_array(drop_data)
X_scaled_drop_data = scale_data(X_drop_data)

X_no_stalk, y_no_stalk = dataframe_to_array(data_no_stalk)
X_scaled_no_stalk = scale_data(X_no_stalk)

y_ohc_data = y_data
X_scaled_ohc = scale_data(pre_ohc_data)

#%% [markdown]
# <a id='pca'></a>
# ## Principal component analysis
# When our data are represented by a matrix too large (the number of dimensions is too high), it is 
# difficult to extract the most interesting features and find correlations among them; moreover the space 
# occupied is very high. PCA is a technique that allows to achieve dimensionality reduction while preserving 
# the most important differences among samples. 
# 
# This transformation is defined in such a way that the first principal component has the largest 
# possible variance (that is, accounts for as much of the variability in the data as possible), and 
# each succeeding component in turn has the highest variance possible under the constraint that it 
# is orthogonal to the preceding components. The resulting vectors (each being a linear combination 
# of the variables and containing n observations) are an uncorrelated orthogonal basis set.
#
# Please note that we can apply PCA only on the One Hot Encoded dataset, because values in the Label Encoded one
# does not have any geometrical meaning.
#
# Let's calculate the Principal Components and show their retained variance on a bar graph.
#%% 
def plot_cumulative_variance(pca):
    """
    Plots cumulative variance of all PC

    :param (pca object) pca: Pca object
    """   

    tot_var = np.sum(pca.explained_variance_)
    ex_var = [(i / tot_var) * 100 for i in sorted(pca.explained_variance_, reverse=True)]
    cum_ex_var = np.cumsum(ex_var)

    cum_var_bar = go.Bar(
        x=list(range(1, len(cum_ex_var) + 1)), 
        y=ex_var,
        name="Variance of each component",
        marker=dict(
            color=PLOTLY_COLORS[0],
        ),
        opacity=PLOTLY_OPACITY
        )

    variance_line = go.Scatter(
        x=list(range(1, len(cum_ex_var) + 1)),
        y=cum_ex_var,
        mode='lines+markers',
        name="Cumulative variance",
        marker=dict(
            color=PLOTLY_COLORS[1],
        ),
        opacity=PLOTLY_OPACITY,
        line=dict(
            shape='hv',
        ))
    data = [cum_var_bar, variance_line]
    layout = go.Layout(
        title='Individual and Cumulative Explained Variance',
        autosize=True,
        yaxis=dict(
            title='Explained variance (%)',
        ),
        xaxis=dict(
            title="Principal components",
            dtick=1,
            rangemode='nonnegative'
        ),
        legend=dict(
            x=0,
            y=1,
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    return iplot(fig, filename='basic-bar')

def compress_data(X_dataset, n_components, plot_comp=False):
    
    """
    Performs pca reduction of a dataset.

    :param (array of arrays) X_dataset: Dataset to reduce
    :param (int) n_components: N components to project on 
    :param (bool) plot_comp: Plot explained variance

    :returns (pandas dataframe) X_df_reduced: pandas dataframe with reduced dataset
    :return (iplot) p: Plot, only if plot_com equals True. To plot the graph,
                       simply call the return value.
    """   

    pca = PCA(random_state=RANDOM_SEED)
    projected_data = pca.fit_transform(X_dataset)

    if plot_comp:
        p = plot_cumulative_variance(pca)

    pca.components_ = pca.components_[:n_components]
    reduced_data = np.dot(projected_data, pca.components_.T)
    X_df_reduced = pd.DataFrame(reduced_data, columns=["PC#%d" % (x + 1) for x in range(n_components)])
    if plot_comp:
        return p, X_df_reduced
    else:
        return X_df_reduced
    
#%%
'''
plot, X_df_reduced = compress_data(X_dataset=X_scaled_data,
                             n_components=9,
                             plot_comp=True)
plot                       


X_df_drop_reduced = compress_data(X_dataset=X_scaled_drop_data,
                                  n_components=9,
                                  plot_comp=False)
'''
#%%
plot, X_df_ohc_reduced = compress_data(X_dataset=pre_ohc_data,
              n_components=20,
              plot_comp=True)
plot
#%% [markdown]
# We can see that on those many components, some of them can be excluded. From the graph we can see that the first 20 
# components retain almost 80% of total variance, while last 39 not even 1%.
#
# This allows us to work on a smaller dataset achieving similar results, 
# because most of the information is maintained.
#%%
X_df_ohc_reduced.head(4)

#%% [markdown]
#
# We will try now to use a scatte-plot to show if a clustering algorithm applied
# on the first two principal components is able to separate the samples in different clusters.
# 
# PCA tries to find combinations of features that lead to maximum separation between data points. 
# What this means is that, if we had a dimension in our dataset which was the same for all members, 
# then that would not be considered, alone or in combination, among the top principal components. 

# Only the features that vary a lot from data point to data point form a part of the top principal components. 
# As a result, the points should appear to be quite far apart from each other on the plot.
# 
# The plot of the PCA clusters may not make sense in a cuple of conditions:
#
# 1. There is a lot of variance in the dataset, so the first two components
# cannot significantly represent the data
# 2. The clustering algorithm focuses on features considered unimportant by the 
# PCA.
#
#
# Our dataset should not have that much variance, so the clustering
# algorithm should be able to decently separate data.
#
#%% 

values = pre_ohc_data.values
pca = PCA(n_components=2)
x = pca.fit_transform(values)

kmeans = KMeans(n_clusters=2, random_state=RANDOM_SEED)
X_clustered = kmeans.fit_predict(values)

c1_idx = np.where(X_clustered == 0)
c2_idx = np.where(X_clustered == 1)

p1 = go.Scatter(
    x=np.take(x[:,0], indices=c1_idx)[0],
    y=np.take(x[:,1], indices=c1_idx)[0],
    mode='markers',
    name="Cluster1",
    marker=dict(
        color=COLOR_PALETTE[0],
    ),
    opacity=PLOTLY_OPACITY)

p2 = go.Scatter(
    x=np.take(x[:,0], indices=c2_idx)[0],
    y=np.take(x[:,1], indices=c2_idx)[0],
    mode='markers',
    name="Cluster2",
    marker=dict(
        color=COLOR_PALETTE[1],
    ),
    opacity=PLOTLY_OPACITY)

data = [p1, p2]

layout = go.Layout(
    title='Data clustered using first two components',
    autosize=True,
    yaxis=dict(
        title='Second component',
    ),
    xaxis=dict(
        title="First component",
        dtick=1,
    ),
    legend=dict(
        x=0,
        y=1,
    ),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='clusters-scatter')
#%% [markdown]
# As expected, using K-means we are able to separate two groups of data using the 
# two components with maximum variance.
#
#%% [markdown]
# <a id='classification'></a>
## Classification
# Now we are going to explore different supervised learning classification methods, and see in the end the one that performs better. 
#
# Now, before starting the classification phase, let's see what kind of pre-processed data it is better to use to achieve the best classification possible.
# Due to the fact that our dataset is pretty small, probably the dimensionality reduction using PCA is not strictly necessary, but if we can achieve a similar score using just main principal components, it is definitely better.
#
# We are going to compare results of several classification methods on the different datasets. In this way we can choose the one 
# to pick for the next phase. 
# The current versions of the dataset are:
#
# 1. Full dataset Label Encoded.
# 2. Dataset Label Encoded with missing values removed.
# 3. Dataset Label Encoded with column containing missing values removed 
# 4. Dataset One Hot Encoded
# 5. Dataset One Hot Encoded compressed using PCA
#
#
# Our dataset is pretty balanced, so we do not strictly need any over or under-sampling technique.
# Let's start with splitting the datasets in train and test.

#%%

X_train, X_test, y_train, y_test = train_test_split(X_scaled_data, y_data, test_size=0.2, random_state=RANDOM_SEED)
X_train_drop, X_test_drop, y_train_drop, y_test_drop = train_test_split(X_scaled_drop_data, y_drop_data, test_size=0.2, random_state=RANDOM_SEED)
X_train_no_stalk, X_test_no_stalk, y_train_no_stalk, y_test_no_stalk = train_test_split(X_scaled_no_stalk, y_no_stalk, test_size=0.2, random_state=RANDOM_SEED)

X_train_ohc_pc, X_test_ohc_pc, y_train_ohc_pc, y_test_ohc_pc = train_test_split(X_df_ohc_reduced, y_data, test_size=0.2, random_state=RANDOM_SEED)
X_train_ohc, X_test_ohc, y_train_ohc, y_test_ohc = train_test_split(pre_ohc_data, y_data, test_size=0.2, random_state=RANDOM_SEED)

'''
train_pc_drop, X_test_pc_drop, y_train_pc_drop, y_test_pc_drop = train_test_split(X_df_drop_reduced, y_drop_data, test_size=0.2, random_state=RANDOM_SEED)
X_train_pc, X_test_pc, y_train_pc, y_test_pc = train_test_split(X_df_reduced, y_data, test_size=0.2, random_state=RANDOM_SEED)
'''
#%% [markdown]
# We will apply different classification models on our datasets without any tuning of the parameters. In this way we can 
# see which datasets gives an overall efficient tradeoff between accuracy and time.
# In the next phase we will analyze one classifier at a time optimizing its parameters to obtain the best
# accuracy.
# 
# Let's start defining the functions that we are going to use:

#%%

def print_gridcv_scores(grid_search, n=5):
    """
    Prints the best score achieved by a grid_search, alongside with its parametes

    :param (estimator) clf: Classifier object
    :param (int) n: Best n scores 
    """    
    
    t = PrettyTable()

    print("Best grid scores on validation set:")
    indexes = np.argsort(grid_search.cv_results_['mean_test_score'])[::-1][:n]
    means = grid_search.cv_results_['mean_test_score'][indexes]
    stds = grid_search.cv_results_['std_test_score'][indexes]
    params = np.array(grid_search.cv_results_['params'])[indexes]
    
    t.field_names = ['Score'] + [f for f in params[0].keys()] 
    for mean, std, params in zip(means, stds, params):
        row=["%0.3f (+/-%0.03f)" % (mean, std * 2)] + [p for p in params.values()]
        t.add_row(row)
    print(t)
               

def param_tune_grid_cv(clf, params, X_train, y_train, cv, execution_time=False):
    """
    Function that performs a grid search over some parameters

    :param (estimator) clf: Classifier object
    :param (dictionary) params: parameters to be tested in grid search
    :param (array-like) X_train: List of data to be trained with
    :param (array-like) y_train: Target relative to X for classification or regression
    :param (cross-validation generator) cv: Determines the cross-validation splitting strategy
    """ 
    if execution_time:
      start = time.perf_counter()
    pipeline = Pipeline([('clf', clf)])
    grid_search = GridSearchCV(estimator=pipeline, 
                               param_grid=params, 
                               cv=cv, 
                               n_jobs=-1,       # Use all processors
                               scoring='f1',    # Use f1 metric for evaluation
                               return_train_score=True)
    grid_search.fit(X_train, y_train)
    if execution_time:
      end = time.perf_counter()
      return grid_search, "%.4f" % (end-start)
    return grid_search
   

def score(clfs, datasets):
    """
    Function that scores a classifier on some data
    
    :param (array of estimator) clf: Array of classifiers
    :param (dictionary) params: Dictionary of test data, passed like [(X_test, y_test)]

    """  
    scores = []
    for c, (X_test, y_test) in zip(clfs, datasets):
        scores.append(c.score(X_test, y_test))

    return scores


def hexToRGBA(hex, alpha):

    """
    Function that returns an rgba value from an hex and an opacity value
    
    :param (String) clf: Hex value 
    :param (float) params: Value between 0 and 1 indicating opacity

    """  

    r = int(hex[1:3], 16)
    g = int(hex[3:5], 16)
    b = int(hex[5:], 16)

    if alpha:
        return "rgba(" + str(r) + ", " + str(g) + ", " + str(b) + ", " + str(alpha) + ")"
    else:
        return "rgb(" + str(r) + ", " + str(g) + ", " + str(b) + ")"


def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=-1, train_sizes=np.linspace(.008, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    
    train_sizes, train_scores, test_scores = learning_curve(estimator, 
                                                            X, 
                                                            y, 
                                                            cv=cv, 
                                                            n_jobs=n_jobs, 
                                                            train_sizes=train_sizes, 
                                                            scoring="f1", 
                                                            random_state=RANDOM_SEED)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Prints lower bound (mean - std) of train 
    trace1 = go.Scatter(
        x=train_sizes, 
        y=train_scores_mean - train_scores_std, 
        showlegend=False,
        mode="lines",
        name="",
        hoverlabel = dict(
            namelength=20
        ),
        line = dict(
            width = 0.1,
            color = hexToRGBA(PLOTLY_COLORS[0], 0.4),
        ),
    )
    # Prints upper bound (mean + std) of train
    trace2 = go.Scatter(
        x=train_sizes, 
        y=train_scores_mean + train_scores_std, 
        showlegend=False,
        fill="tonexty",
        mode="lines",
        name="",
        hoverlabel = dict(
            namelength=20
        ),
        line = dict(
            width = 0.1,
            color = hexToRGBA(PLOTLY_COLORS[0], 0.4),
        ),
    )
    
    # Prints mean train score line
    trace3 = go.Scatter(
        x=train_sizes, 
        y=train_scores_mean, 
        showlegend=True,
        name="Train score",
        line = dict(
            color = PLOTLY_COLORS[0],
        ),
    )
    
    # Prints lower bound (mean - std) of test 
    trace4 = go.Scatter(
        x=train_sizes, 
        y=test_scores_mean - test_scores_std, 
        showlegend=False,
        mode="lines",
        name="",
        hoverlabel = dict(
            namelength=20
        ),
        line = dict(
            width = 0.1,
            color = hexToRGBA(PLOTLY_COLORS[1], 0.4),
        ),
    )
        # Prints upper bound (mean + std) of test
    trace5 = go.Scatter(
        x=train_sizes, 
        y=test_scores_mean + test_scores_std, 
        showlegend=False,
        fill="tonexty",
        mode="lines",
        name="",
        hoverlabel = dict(
            namelength=20
        ),
        line = dict(
            width = 0.1,
            color = hexToRGBA(PLOTLY_COLORS[1], 0.4),
        ),
    )

    # Prints mean test score line 
    trace6 = go.Scatter(
        x=train_sizes, 
        y=test_scores_mean, 
        showlegend=True,
        name="Test score",
        line = dict(
            color = PLOTLY_COLORS[1],
        ),
    )
    
    data = [trace1, trace2, trace3, trace4, trace5, trace6]
    layout = go.Layout(
        title=title,
        autosize=True,
        yaxis=dict(
            title='F1 Score',
        ),
        xaxis=dict(
            title="#Training samples",
        ),
        legend=dict(
            x=0.8,
            y=0,
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    return iplot(fig, filename=title)


def print_confusion_matrix(gs, X_test, y_test):

    """
    Function that prints confusion matrix for a classifier
    
    :param (estimator) clf: Classifier object
    :param (array-like) X_test: List of data to be tested with
    :param (array-like) y_test: List of labels for test 
    """  

    gs_score = gs.score(X_test, y_test)
    y_pred = gs.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    t = PrettyTable()
    t.add_row(["True Edible", cm[0][0], cm[0][1]])
    t.add_row(["True Poisonous", cm[1][0], cm[1][1]])
    t.field_names = [" ", "Predicted Edible", "Predicted Poisonous"]
    print(t)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # normalize the confusion matrix
    cm_df = pd.DataFrame(cm.round(3), index=["True edible", "True Poisonous"], columns=["Predicted edible", "Predicted poisonous"])
    cm_df


def print_raw_score(clf, X_test, y_test):
    """
    Function that scores a classifier on some data
    
    :param (array of estimator) clf: Array of classifiers
    :param (array-like) X_test: List of data to be tested with
    :param (array-like) y_test: List of labels for test 

    """  
    print("Score achieved by NB: %0.3f" % (score([clf], [(X_test, y_test)])[0]))


def plot_feature_importance(feature_importance, title):
    """
    Function that plots feature importance for a decision tree or a random forest classifier
    
    :param (dictionary) feature_importance: Dictionary of most important features sorted
    :param (str) title: Title of the plot

    """ 
    
    trace1 = go.Bar(
        x=feature_importance[:, 0],
        y=feature_importance[:, 1],
        marker = dict(color = PLOTLY_COLORS[0]),
        opacity=PLOTLY_OPACITY,
        name='Feature importance'
    )
    data = [trace1]
    layout = go.Layout(
        title=title,
        autosize=True,
        margin=go.layout.Margin(l=50, r=100, b=150),
        xaxis=dict(
            title='feature',
            tickangle=30
        ),
        yaxis=dict(
            title='feature importance',
            automargin=True,
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    return iplot(fig, filename=title)


def print_performances(classifiers, classifier_names, auc_scores, X_test, y_test):
  
    """
    Function that scores a classifier on some data
    
    :param (array of estimator) clf: Array of classifiers
    :param (array-like) classifier_names: Title of the classifier
    :param (array-like) auc-score: Auc scores
    :param (array-like) X_test: List of data to be tested with
    :param (array-like) y_test: List of labels for test 

    """ 

    accs = []
    recalls = []
    precision = []
    results_table = pd.DataFrame(columns=["accuracy", "precision", "recall", "f1", "auc"])
    for (i, clf), name, auc in zip(enumerate(classifiers), classifier_names, auc_scores):
        y_pred = clf.predict(X_test)
        row = []
        row.append(accuracy_score(y_test, y_pred))
        row.append(precision_score(y_test, y_pred))
        row.append(recall_score(y_test, y_pred))
        row.append(f1_score(y_test, y_pred))
        row.append(auc)
        row = ["%.3f" % r for r in row]
        results_table.loc[name] = row
    return results_table


#%% [markdown]

# Let's start defining the Stratified K-Folds cross-validator; it provides train/test indices to split data in train/test sets.
#
# This cross-validation object is a variation of KFold that returns stratified folds. The folds are made by preserving the percentage of 
# samples for each class. Stratification is generally a better scheme, both in terms of bias and variance, when compared to 
# regular cross-validation.
#
# Then we define our classifiers, setting the `random_state` parameter to our usual seed value, 
# in such a way that we can classify each time in the same way.
#%% 

kf = StratifiedKFold(n_splits=5, random_state=RANDOM_SEED)
clf_nb = GaussianNB()
clf_knn = KNeighborsClassifier()
clf_rf = RandomForestClassifier(random_state=RANDOM_SEED)
clf_lr = LogisticRegression(random_state=RANDOM_SEED)
clf_svm = SVC(random_state=RANDOM_SEED)

clfs = [clf_nb, clf_knn, clf_rf, clf_lr, clf_svm]
clfs_ng = [clf_nb, clf_rf]
TEST_PARAMS = {}

#%% [markdown]
# We set `TEST_PARAMS` empty, because in this phase we are not interested in tuning parameters.
#%%
all_test_results = []
all_gss = []
times = np.zeros(2)
t = 0

for clf in clfs:
    gs_ohc_pc, t = param_tune_grid_cv(clf, TEST_PARAMS, X_train_ohc_pc, y_train_ohc_pc, kf, execution_time=True)
    times[0] = times[0] + float(t)

    gs_ohc, t = param_tune_grid_cv(clf, TEST_PARAMS, X_train_ohc, y_train_ohc, kf, execution_time=True)
    times[1] = times[1] + float(t)

    gss = [gs_ohc_pc, gs_ohc]
    all_gss.append(gss)
    test_results = score(gss, [(X_test_ohc_pc, y_test_ohc_pc),
                                (X_test_ohc, y_test_ohc)])
    all_test_results.append(test_results)

all_test_results_ng = []
all_gss_ng = []
times_ng = np.zeros(3)

for clf in clfs_ng:
    gs_full, t = param_tune_grid_cv(clf, TEST_PARAMS, X_train, y_train, kf, execution_time=True)
    times_ng[0] = times_ng[0] + float(t)
  
    #gs_pc, t = param_tune_grid_cv(clf, TEST_PARAMS, X_train_pc, y_train_pc, kf, execution_time=True)
    #times[1] = times[1] + float(t)
  
    gs_drop, t = param_tune_grid_cv(clf, TEST_PARAMS, X_train_drop, y_train_drop, kf, execution_time=True)
    times_ng[1] = times_ng[1] + float(t)
  
    #gs_pc_drop, t = param_tune_grid_cv(clf, TEST_PARAMS, X_train_pc_drop, y_train_pc_drop, kf, execution_time=True)
    #times[3] = times[3] + float(t)
  
    gs_no_stalk, t = param_tune_grid_cv(clf, TEST_PARAMS, X_train_no_stalk, y_train_no_stalk, kf, execution_time=True)
    times_ng[2] = times_ng[2] + float(t)

    gss_ng = [gs_full, gs_drop, gs_no_stalk]
    all_gss_ng.append(gss_ng)
    test_results = score(gss_ng, [(X_test, y_test),
                                  (X_test_drop, y_test_drop),
                                  (X_no_stalk, y_no_stalk)])
    all_test_results_ng.append(test_results)

#%% [markdown]
# This is the score of the different classifiers on the different datasets.

#%%

def print_results(column_names, row_names, values):
    t = PrettyTable()
    t.field_names = column_names
    
    all_rows = []
    result_row = []

    for name, results in zip(row_names, values):
        result_row.append(name)
        for r in results:
            result_row.append("%.3f" % r)
        all_rows.append(result_row)
        result_row = []

    all_rows = sorted(all_rows, key=lambda kv: kv[1], reverse=True)
    for k in all_rows:
        t.add_row(k)
    
    print(t)


dataset_strings = [" ",
                   "dataset ohc reduced on first 20 PC",
                   "dataset ohc"]

dataset_strings_ng = [" ", "full dataset",
                      "dataset with dropped missing values",
                      "dataset with stalk-root field dropped"]

row_names = ["Naive Bayes", "KNN", "Random Forest", "Logistic Regression", "SVM"]
row_names_ng = ["Naive Bayes", "Random Forest"]

print_results(dataset_strings, row_names, all_test_results)
print_results(dataset_strings_ng, row_names_ng, all_test_results_ng)


#%% [markdown]
# Looking at the second table we can notice that the Naive Bayes classifier performed very poorly on the 
# dataset where we dropped the rows containing the missing values. If we look deeper in its confusion matrix, 
# we can notice that almost all edible mushrooms are classified correcly, while almost all of the poisonous ones are 
# misclassified. This happens because the probability of finding an edible sample is much higher than the one of finding a 
# poisonous one, and the classifies picks almost always the edible class due to this reason. To address
# this problem, we should apply some over/under sampling methods (or a combination of them), such as
# SMOTE or RandomOver(Under)Sampler. 
#%%
print_confusion_matrix(all_gss_ng[0][np.argmin(all_test_results_ng[0])], X_test_drop, y_test_drop)

#%% [markdown]
# After have evaluated all the above results, we are going to select the one that have the best tradeoff
# between time and score. Let's calculate the mean score and the overall time taken to train all
# the different classifiers with respect to the same dataset.

#%%
means = np.mean(all_test_results, axis=0)
row_names = ["Total train time (s)", "Mean score for dataset"]
print_results(dataset_strings, row_names, [times, means])

#%%

means_ng = np.mean(all_test_results_ng, axis=0)
print_results(dataset_strings_ng, row_names, [times_ng, means_ng])

#%%
  
print("The dataset that gives the best overall performances is:")
print("\t- " + dataset_strings[means.argmax()] + ", with a score of " + str("%.3f" % means.max()))

#%% [markdown]
# As we can see, the most accurate on average is the dataset encoded with `OneHotEncoder`; The training time though, is really high 
# with respect to the other datasets. 
# If we look carefully, we achieve an optimal score using the Random Forest Classifier on the full dataset, without any parameter 
# tuning, in a short time. This indicates that the dataset is easily classified, even without the use of more complicated methods.
#
# For demonstrative purposes, for now on we will use the dataset One Hot Encoded, but the compressed version, to 
# see how the accuracy can improve by means of parameter tuning in a reasonable time. 
# This dataset allows us to use all the classification models mentioned above.


#%% [markdown]
# ### Logistic Regression
# The first classifier that we will analyze is the Logistic Regression classifier. It uses the 
# sigmoid function to classify our samples:
#
# - $P(y=0 | X;\theta) = g(w^T X) = \frac{1}{1+e^{w^T X}}$
# - $P(y=1 | X;\theta) = 1 - g(w^T X) = \frac{e^{w^T X}}{1+e^{w^T X}}$
# 
# 
# This model, with respect to linear regression, can model better the zone close to 
# 0 and 1. To learn the weights, the $MLE$ is found and then the gradient descent algorithm 
# is applied until the accuracy converges.
#
# They are:
# - `liblinear`. Solver, which is better for smaller datasets
# - `C`. Regularization strength, ranging from 0.01 to 100. A smaller value inidicates stronger regularization, like in svms.
# - `penalty`. "l1" and "l2" penalty for regularization, which are defined as:
#   - l1, it penalizes every mistake at the same way
#       - $S = \Sigma_{i=1}^{n}{|y_i - f(x_i)|}$
#    
#   - l2, it penalizes bigger values
#       - $S = \Sigma_{i=1}^{n}{(y_i - f(x_i))^2}$
#  
#   - Where $y_i$ is the true label and $f(x_i)$ is the assigned label.

#%%

clf_lr = LogisticRegression(random_state=RANDOM_SEED)
gs_pc_lr = param_tune_grid_cv(clf_lr, LOGISTIC_REGRESSION_PARAMS, X_train_ohc_pc, y_train_ohc_pc, kf)
print_gridcv_scores(gs_pc_lr, n=5)

#%%
plot_learning_curve(gs_pc_lr.best_estimator_, "Learning curve of Logistic Regression", 
                    X_train_ohc_pc,
                    y_train_ohc_pc,
                    cv=5)

#%% [markdown]
# From the graph we can see that the two lines approach themselves ans the test score
# tends to the train score. In this case, from the graph is evident that adding more
# samples is useless; we have more than enough, but the model is not complex enough to 
# anchieve higher scores.
#
# In any case, we achieved a 0.02 improvement with the tuning of the Logistic Regressor parameters.

#%% [markdown]
# From the learning curve we can see that, at the beginning 
#%%
print_confusion_matrix(gs_pc_lr, X_test_ohc_pc, y_test_ohc_pc)

#%% [markdown]
# ### Support vector machine
# A Support Vector Machine (SVM) is a discriminative classifier formally defined by a 
# separating hyperplane. In other words, given labeled training data (supervised learning), 
# the algorithm outputs an optimal hyperplane which categorizes new examples. In two dimentional space 
# this hyperplane is a line dividing a plane in two parts where in each class lay in either side.
# 
# We perform the grid search over:
# - `linear`. This is the simplest SVM, finds the hyperplane which separates in the best way our samples.
# - `C`. The C parameter tells the SVM optimization how much you want to avoid misclassifying 
# each training example. For large values of C, the optimization will choose a smaller-margin 
# hyperplane if that hyperplane does a better job of getting all the training points 
# classified correctly.
# - `rbf`. This parameter indicates that we are using a radial basis function kernel to perform the 
# scalar product. 
#   - `gamma`. Defines how far the influence of a single training example reaches, 
# with low values meaning ‘far’ and high values meaning ‘close’.

#%%
clf_svm = SVC(probability=True, random_state=RANDOM_SEED)
gs_pc_svm = param_tune_grid_cv(clf_svm, SVM_PARAMS, X_train_ohc_pc, y_train_ohc_pc, kf)
print_gridcv_scores(gs_pc_svm, n=5)

#%%
plot_learning_curve(gs_pc_svm.best_estimator_, "Learning curve of SVM", 
                    X_train_ohc_pc,
                    y_train_ohc_pc,
                    cv=5)

#%% [markdown]
# We can see from the learning curve that we can achieve optimal performances
# before running out of training samples. After approximately 1300 samples analyzed,
# we already achieve a test score around than 0.99 with very low standard deviation.
# 
# This is the best score possibly achievable.
# We could expect it, but we will print anyway the confusion matrix:
# 

#%%
print_confusion_matrix(gs_pc_svm, X_test_ohc_pc, y_test_ohc_pc)

#%% [markdown]

# ### Naive Bayes Classifier
# The naive bayes classifier is based on the Bayes theorem, which states that:
#
# - $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$
#
# Where:
# - $P(A)$, the prior, is the initial degree of belief in A.
# - $P(A|B)$, the posterior is the degree of belief having accounted for B.
# - $P(B|A$), is the likelyhood, the degree of belief in B, given that A is true.
#
#
# Using Bayes theorem, we can find the probability of A happening, 
# given that B has occurred. Here, B is the evidence and A is the hypothesis. The assumption made 
# here is that the predictors/features are independent. That is presence of one particular 
# feature does not affect the other. Hence it is called naive.

#%%
clf_nb = GaussianNB()
gs_pc_nb = param_tune_grid_cv(clf_nb, TEST_PARAMS, X_train_ohc_pc, y_train_ohc_pc, kf)
print_gridcv_scores(gs_pc_nb, n=5)

print_confusion_matrix(gs_pc_nb, X_test_ohc_pc, y_test_ohc_pc)

#%%
plot_learning_curve(clf_nb, "Learning curve of GaussianNB", 
                    X_train_ohc_pc, 
                    y_train_ohc_pc, 
                    cv=5)

#%% [markdown]
# To be a really simple classifier, it can achieve decent results. The independecy assumption
# does not work perfectly, because the different features are never completely independent from 
# each other, but nevertheless it manages to achieve a good accuracy even if we used only 
# the data projected on the principal components. 
# 
# With the naive bayes classifier, it is always 
# better to use the original dataset, being really fast itself to train and to classify samples.
# With the ohc dataset, if we look in the upper table, we achieve a score around 0.945, which is 
# a great score given our big assumption.


#%% [markdown]
# ### Random Forest Classifier
# It is a tree-based method, derived from bagging decision trees, to which
# it is added another small trick which decorrelates the trees, in order to 
# reduce further the variance. 
#
# As in bagging, we build a number of decision trees on bootstrapped training 
# samples. But when building these decision trees, each time a split in a tree 
# is considered, a random selection of m predictors is chosen as split 
# candidates from the full set of p predictors. The split is allowed to use only 
# one of those m predictors, which typically are $m \approx \sqrt{p}$.
#%%

clf_pc_rf = RandomForestClassifier(random_state=RANDOM_SEED)
gs_pc_rf = param_tune_grid_cv(clf_pc_rf, RANDOM_FOREST_PARAMS, X_train_ohc_pc, y_train_ohc_pc, kf)
print_gridcv_scores(gs_pc_rf, n = 5)

#%%
print_confusion_matrix(gs_pc_rf, X_test_ohc_pc, y_test_ohc_pc)

#%%
plot_learning_curve(gs_pc_rf.best_estimator_, "Learning curve of Random Forest Classifier", 
                    X_train_ohc_pc,
                    y_train_ohc_pc,
                    cv=5)

#%% [markdown]
# The training time is pretty high, but the accuracy as well. Looking at the learning curve we can notice that
# the test score steadly increases, approaching the training score, but it does not manage to reach it. 
# In this case, if we had more training samples, we could possibly achieve a better performance.
#
# Similarly to the case of the Naive Bayes Classifier, also in this case it is better to use the original dataset 
# with a simpler encoding, as we can see from the score achieved with it compared to this one. 
# 
# Now let's look deeper into the features of the Random Forest Classifier; let's see which of them weight more
# on the classification.
#%%
feature_importance = np.array(  sorted(zip(X_train_ohc_pc.columns, 
                                gs_pc_rf.best_estimator_.named_steps['clf'].feature_importances_),
                                key=lambda x: x[1], reverse=True))
plot_feature_importance(feature_importance, "Feature importance in the random forest")

#%% [markdown]
# Here we just see which are the components that weight more in the classification, but due to the 
# fact that we reduced the dataset by means of PCA, we only see which component weighted more in
# the node splitting phase. Why the last components weight more than the first, the ones
# which contain most of the variability?
#
# The first principal component is a linear combination of all our features. The fact that it explains almost all the 
# variability just means that most of the coefficients of the variables in the first principal component are significant.
# The classification trees we generate do binary splits on 
# continuous variables that best separate the categories we want to classify. That is not exactly 
# the same as finding orthogonal linear combinations of continuous variables that give the direction 
# of greatest variance. 

#%% [markdown]
# ### K-Nearest Neighbors Classifier
# K-NN is a type of instance-based learning, or lazy learning, where the function 
# is only approximated locally and all computation is deferred until classification. 
# The K-NN algorithm is among the simplest of all machine learning algorithms.
# 
# The training phase of the algorithm consists only of storing the feature vectors 
# and class labels of the training samples. In the classification phase, K is a user-defined constant, 
# and an unlabeled vector (a query or test point) is classified by assigning the label which 
# is most frequent among the k training samples nearest to that query point.
# 
# The parameters for cross validation are:
# - `n_neighbors`. Number of closes samples to analyze
# - `weights`. Indicates the weight function to use in prediction.
#   - `uniform`. All points in the neighborhood are weighted equally.
#   - `distance`. Weight points by the inverse of their distance. 
# In this case, closer neighbors of a query point will have a greater 
# influence than neighbors which are further away.
# - `p`. Power parameter for the Minkowski metric (generalization of Euclidean distance)
#   - p=1 uses `l1`
#   - p=2 uses `l2`
#   - p>2 minkowski_distance (l_p) is used.

#%%

clf_knn = KNeighborsClassifier()
gs_knn = param_tune_grid_cv(clf_knn, KNN_PARAMS, X_train_ohc_pc, y_train_ohc_pc, kf)
print_gridcv_scores(gs_knn, n=5)

#%%
print_confusion_matrix(gs_knn, X_train_ohc_pc, y_train_ohc_pc)

#%%

plot_learning_curve(gs_knn.best_estimator_, "Learning curve of k-NN Classifier", 
                    X_train_ohc_pc,
                    y_train_ohc_pc,
                    cv=5)


#%% [markdown]
# The accuracy achieved is really high.
# The K-NN classification with the whole dataset gives almost the same result but it takes a lot more time.
# If we look at the curves, also here the test score steadily increased;
# if we had more training samples, we could probably achieve the score of 1.

#%% [markdown]
# ### ROC curve
# At this point we need to evaluate the performances of the different classifiers and
# compare them. 
# We are going to plot the ROC curve and the Area Under Curve for all our models.
# classifiers. 
# The ROC curve is plotted with TPR against the FPR where TPR is on y-axis and FPR is on the x-axis.
# Specifically, these parameters are:
# 
# - $TRP/Recall/Sensitivity = \frac{TP}{TP+FN}$
# - $FPR = \frac{FP}{TN+FP}$
#   
#
# An excellent model has AUC near to the 1 which means it has good measure of 
# separability. A poor model has AUC near to the 0 which means it has worst measure 
# of separability.
#%%

def plot_roc_curve(classifiers, legend, title, X_test, y_test):
    t1 = go.Scatter(
        x=[0, 1], 
        y=[0, 1], 
        showlegend=False,
        mode="lines",
        name="",
        line = dict(
            color = COLOR_PALETTE[0],
        ),
    )
    
    data = [t1]
    aucs = []
    for clf, string, c in zip(classifiers, legend, COLOR_PALETTE[1:]):
        y_test_roc = np.array([([0, 1] if y else [1, 0]) for y in y_test])
        y_score = clf.predict_proba(X_test)
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(2):
            fpr[i], tpr[i], _ = roc_curve(y_test_roc[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_roc.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        aucs.append(roc_auc['micro'])

        trace = go.Scatter(
            x=fpr['micro'], 
            y=tpr['micro'], 
            showlegend=True,
            mode="lines",
            name=string + " (area = %0.2f)" % roc_auc['micro'],
            hoverlabel = dict(
                namelength=30
            ),
            line = dict(
                color = c,
            ),
        )
        data.append(trace)

    layout = go.Layout(
        title=title,
        autosize=False,
        width=550,
        height=550,
        yaxis=dict(
            title='True Positive Rate',
        ),
        xaxis=dict(
            title="False Positive Rate",
        ),
        legend=dict(
            x=0.4,
            y=0.06,
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    return aucs, iplot(fig, filename=title)

#%%

classifiers = [gs_pc_lr, gs_pc_svm, gs_pc_nb, gs_pc_rf, gs_knn]
classifier_names = ["Logistic Regression", "SVM", "GaussianNB", "Random Forest", "KNN"]
auc_scores, roc_plot = plot_roc_curve(classifiers, classifier_names, "ROC curve", X_test_ohc_pc, y_test_ohc_pc)
roc_plot

#%% [markdown]
# Now let's look at some additional parameters that evaluate the goodness of our models:
# 
# 1. Accuracy
# 2. Precision
#       - $P = \frac{TP}{TP+FP}$
# 3. Recall
#       - $R = \frac{TP}{TP+FN}$
# 4. F1 (weighted average of the precision and recall)
#       - $F1= 2*\frac{P * R}{P + R}$
# 5. Auc, area under the ROC curve
#%%
print_performances(classifiers, classifier_names, auc_scores, X_test_ohc_pc, y_test_ohc_pc)

#%% [markdown]
# The table shows that overall all classifiers performed well. The Logistic Regression Classifier
# is the one that performed worse, even worse than the Naive Bayes Classifier, which is the simplest one of them. 
# The NB, being really fast, gives better results in a very short time with the whole dataset. Infact 
# the one OneHotEncoded anchieves a really high score, namely 0,945 (see score table in the choice
# of the dataset phase).
# The SVM is the classifier that achieved the best score, hitting the 1 value on every parameter, while the
# k-NN almost reaches that score due to the shortage of training samples.
# The the Random Forest classifier achieved a similar score, even if slightly 
# worse in accuracy and in recall.
#%% [markdown]

# ## What NOT to do in the woods
#
# To have a little bit more fun we will try one more thing.
# If we are in a wood, how can we survive without the help of an SVM? 
# Let's find out what are the peculiar traits of a poisonous mushroom 
#
# Firstly, we create a KNN classifier and we iterate on all the columns,
# to see which of them gives a more accurate classification; 
# In this way we can see which ones are the most important charateristics to
# classify a mushroom.

#%%

n_features = pre_ohc_data.shape[1]
clf = KNeighborsClassifier()
feature_score = []
t = PrettyTable()
t.field_names = ["Feature", "Score"]

for i in range(n_features):
    X_feature= np.reshape(pre_ohc_data.iloc[:,i:i+1],-1,1) # One column at a time
    scores = cross_val_score(clf, X_feature, y_data)
    feature_score.append(scores.mean())
    t.add_row([pre_ohc_data.columns[i], "{0:0.4f}".format(scores.mean())])

print(t)

#%% [markdown]
# Let's now select all the features that are more significant; we will 
# pick the ones with a score greater than 0.7
#%%

f_importance = pd.Series(data = feature_score, index = pre_ohc_data.columns)
f_importance.sort_values(ascending=False, inplace=True)
f_importance[f_importance > 0.7]

#%% [markdown]
# Now we merge the unlabelled dataset with the labels, in such a way that
# we can group our samples using the class.

#%%

col_importance = f_importance[f_importance>0.7].index.values
pre_ohc_Xy = pd.concat([pre_ohc_data, pd.DataFrame(y_data, columns=['class'])], axis=1)
grouped = pre_ohc_Xy.groupby('class')

#%%
feat_edible = grouped.get_group(0)[col_importance].sum()
feat_edible
#%%
feat_poisonous = grouped.get_group(1)[col_importance].sum()
feat_poisonous

#%% [markdown]
# <a class="anchor-link" href="#conclusions">¶</a>
# ## Conclusions
# Our goal was to predict if a mushroom was poisonous or edible from its features.
#  
# We understood that they are well separated and our classifiers can anchieve optimal performances.
#  
# 
# ![](https://infovisual.info/storage/app/media/01/img_en/024%20Mushroom.jpg)
#
#
# Looking at the KNN score using the different values for features, we understood that you should 
# not eat a mushroom if it has:
#
# 1. fishy odor
# 2. stalk surface above ring silky
# 3. stalk surface below ring silky
# 4. gill size narrow
# 5. spore print color chocolatey

#%%
