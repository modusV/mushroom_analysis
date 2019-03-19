#%% [markdown]

# # Safe to eat or deadly poisonous?
# ### An analysis on mushroom classification by Lorenzo Santolini

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

LOGISTIC_REGRESSION_PARAMS = {
    'clf__solver': ['liblinear'],  # best for small datasets
    'clf__C': [0.01, 0.1, 1, 10, 100], # smaller value, stronger regularization, like svm
    'clf__penalty': ['l2', 'l1']
}

SVM_PARAMS = [
{
    'clf__kernel': ['linear'],
    'clf__C': [0.1, 1, 10, 100],
}, 
{
    'clf__kernel': ['rbf'],
    'clf__C': [0.01, 0.1, 1, 10, 100],
    'clf__gamma': [0.01, 0.1, 1, 10, 100],
}]

RANDOM_FOREST_PARAMS = {
    'clf__max_depth': [25, 50, 75],
    'clf__max_features': ["sqrt", "log2"], # sqrt is the same as auto
    'clf__criterion': ['gini', 'entropy'],
    'clf__n_estimators': [100, 300, 500, 1000]
}

KNN_PARAMS = {
    'clf__n_neighbors': [5, 15, 25, 35, 45, 55, 65],
    'clf__weights': ['uniform', 'distance'],
    'clf__p': [1, 2, 10]
}


#%% [markdown]
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

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, learning_curve
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

from imblearn.pipeline import make_pipeline, Pipeline
from imblearn.over_sampling import SMOTE

import warnings
from collections import defaultdict
from prettytable import PrettyTable
from functools import wraps
import time

plotly.tools.set_credentials_file(username='modusV', api_key='OBKKnTR2vYTeKIOKtRU6')
warnings.filterwarnings("ignore")
#%%

# Wrapper to calculate functions speed

def watcher(func):
    """
    Decorator for dumpers.
    Shows how much time it
    takes to create/retrieve
    the blob.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f" ===> took {end-start} seconds")
        return result
    return wrapper

#%% 
'''
# Define classes 

class Dataset:
    
    def __init__(self, data, seed, name):
        self.dataset = data
        self.seed = seed
        self.name = name
        
    def set_name(self, name):
        self.name = name
    
    @property
    def name(self):
        return self.__name

    def import_data(path):
        self.dataset = pd.read_csv(path)

    def count_classes():
        self.n_classes = self.dataset['class'].unique().size
        print(f"There are {self.n_classes} different classes:"
              f"\n {self.dataset['class'].unique().tolist()}")



class Classifier:

    def __init__(self, classifier, params, dataset, seed, name):
        self.classifier = classifier
        self.params = params
        self.dataset = dataset
        self.seed = seed
        self.name = name

'''
#%% [markdown]
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
## Preprocessing

# Before starting the classification phase, we need to preprocess the dataset, in 
# such a way that our classifiers will score with more accuracy and reliability. 
# This is the most important step, if data are messy the classification will perform 
# poorly.

# The steps that we will go trough are:  
# 
# 1. Check data types  

# 2. Remove not significat columns, if any  
# 3. Remove null values, if any
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
print(f"Data types: \n{dataset.head(5)}")

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
#%% 
n_columns_original = len(dataset.columns)
to_drop = [col for col in dataset.columns if dataset[col].nunique() == 1]
dataset.drop(to_drop, axis=1, inplace=True)

for d in to_drop:
    print(str(d) + " ", end="")
print("have been removed because zero variance")
print(f"{n_columns_original - len(dataset.columns)} not significant columns have been removed")

#%% [markdown]
# As we can notice, only one field was removed. 

#%% [markdown]
# ### 3 - Handling missing values
# When we find missing values in a dataset, there are some of the approaches that can be 
# considered:  
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
    t = PrettyTable()
    field_names = []
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
# But this isn’t the case at all. To overcome this problem, we use One Hot Encoder.
#
# What one hot encoding does is, it takes a column which has categorical data, 
# which has been label encoded, and then splits the column into multiple columns. 
# The numbers are replaced by 1s and 0s, depending on which column has what value.
# 
# We will obtain two datasets; we will continue the analysis on the first one, but
# meanwhile we will keep this one for the classification phase, to check if we 
# may have an improvement.
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

TODO: PCA THE SHIT OUT OF THIS AND TRY THE TWO CLASSIFICATIONS
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
# This is the class distribution plotten on a histogram. As we already saw before the class distribution is pretty balanced. 
# 
# In this report the plotly library will be used. 
# 
# *Plotly.py* is an interactive, open-source, and browser-based graphing library for Python, which allows you to create interactive plots in a few steps.

#%%
data = [go.Bar(
            x=class_dict,
            y=y,
            marker=dict(
            color=PLOTLY_COLORS),
            opacity=PLOTLY_OPACITY,
    )]

layout = go.Layout(title="Class distribution",
                   autosize=True,
                   yaxis=dict(
                        title='N. samples',
                    ),
                   )
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='distribution-bar')

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

# From the boxplot above, we can see that the color and the shape of the cap are not an effective parameter to decide whether a mushroom is poisonous or edible, because their plots are very similar (same median and very close distribution). 
# The odor and the population columns, on the other hand, are more significant; 
# 
# In the odor field, all the edible mushrooms are squeezed into a single value
# with a few outliers, while the poisonous may have all the different values.

#%% [markdown]
# #### 6 - Bar graph 
# 
# A bar chart or bar graph is a chart or graph that presents categorical data with rectangular bars with heights or lengths proportional to the values that they represent.
# With a slider we can move along the different features, to better visualize the value distributions.
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
# From the bar graph we can see that the `cap-shape` and the `cap-color` are not that significant to classify a sample. On the other hand, some fields like `odor` show a distinct separation of the two classes; these ones will be the ones with more impact on our classification algorithms.
# 
# From these bar graphs we can see that our dataset is pretty well separated. This means that, in our classification task, we will be able to achieve high accuracy even with some dimensionality reduction.

#%% [markdown]
# ### 7 - Correlation matrix
# A correlation matrix is a table showing correlation coefficients between sets of variables. 
# Each random variable (Xi) in the table is correlated with each of the other values in the table (Xj). 
# This allows you to see which pairs have the highest correlation. Correlation is any statistical association, though in common usage it most often refers to how close two variables are to having a linear relationship with each other.
# 
# We will use the **Pearson's correlation**, which is a measure of the linear correlation between two variables X and Y. According to the Cauchy–Schwarz inequality it has a value between +1 and −1, where 1 is total positive linear correlation, 0 is no linear correlation, and −1 is total negative linear correlation.
# 
# The coefficient for a population is computed as:
# $$
# \rho_{(X,Y)} = \frac{cov(X,Y)}{\sigma_X\sigma_Y}
# $$
# Where:
# - $cov$ is the covariance:
#    $$cov(X,Y) = \frac{E[(X - \mu_X)(Y-\mu_Y)]}{\sigma_X\sigma_Y}$$
#    - Where $\mu$ is the mean value  
# 
# - $\sigma_X$ is the standard deviation of X
#     $$\sigma_X^2 = E[X^2] - [E[X]]^2$$
# - $\sigma_Y$ is the standard deviation of Y
#    $$\sigma_Y^2 = E[Y^2] - [E[Y]]^2$$


#%%
correlation_matrix = pre_data.corr(method='pearson')

trace = go.Heatmap(
    z=correlation_matrix.values.tolist(), 
    x=correlation_matrix.columns, 
    y=correlation_matrix.columns, 
    colorscale=COLORSCALE_HEATMAP,
    opacity=0.95,
    zmin=-1,
    zmax=1)
    

data=[trace]

layout = go.Layout(
    title='Heatmap of columns correlation',
    autosize=False,
    width=850,
    height=700,
    yaxis=go.layout.YAxis(automargin=True),
    xaxis=dict(tickangle=40),
    margin=go.layout.Margin(l=0, r=200, b=200, t=80)
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='labelled-heatmap4')

#%% [markdown]
# From the matrix, we can see that the most correlated columns to the class are `gill-color`, `gill-size` and `bruises`.
#
# The diagonal has correlation 1 because every class has maximum correlation with itself.
#
# We can also see that `veil-color` and `gill-attachment`are highly correlated.

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
inverse_correlation = 1 - abs(pre_data.corr()) # This is the 'dissimilarity' method

fig = ff.create_dendrogram(inverse_correlation.values, 
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
# From the above graph, the closest features are `veil-color` and `gill-attachment`.
# Because that their distance is still far from zero, I choose not to remove any of the two features; moreover our dataset is pretty small, so we should not have great performance issues.

#%% [markdown]
# ### 8/9 - Scale and divide data

# Most of the times, datasets contain features highly varying in magnitudes, units and range. 
# But since, most of the machine learning algorithms use Eucledian distance between two data points in their computations, this is a problem. If left alone, these algorithms only take in the magnitude of features neglecting the units. The results would vary greatly between different units, for example between 5kg and 5000gms. 
# 
# The features with high magnitudes will weigh in a lot more in the distance calculations than features with low magnitudes.
# To supress this effect, we need to bring all features to the same level of magnitudes. This can be acheived by scaling.
# 
# We will use the `StandardScaler`, which standardizes our data both with mean and standard deviation.
#
# The operation performed will be:
#
# $$
# x' = \frac{x - \mu_x}{\sigma_x}
# $$

# After this step, we divide the dataset into an array of unclassified samples and an array of labels, to use for the classification phase.
# 
# At this point, I decide to bring to the next phase two different datasets:  
# 
# 1. The full dataset, where the missing values are encoded with an integer.
# 2. A reduced version of the dataset, where the rows with missing data are considered as incomplete samples and are dropped. 
#%%

def dataframe_to_array(data):
    y_data = data['class']
    X_data = data.drop(['class'], axis=1)
    return X_data, y_data

def scale_data(X_data):
    """

    """
    scaler = StandardScaler(with_mean=True, with_std=True, copy=True)
    return scaler.fit_transform(X_data)

#%%

drop_data = pre_data[pre_data['stalk-root'] != le_mapping['stalk-root']['m']]
data_no_stalk = pre_data.drop(['stalk-root'], axis=1)

X_pre_data, y_data = dataframe_to_array(pre_data)
X_scaled_data = scale_data(X_pre_data)

X_drop_data, y_drop_data = dataframe_to_array(drop_data)
X_scaled_drop_data = scale_data(X_drop_data)

X_no_stalk, y_no_stalk = dataframe_to_array(data_no_stalk)
X_scaled_no_stalk = scale_data(X_no_stalk)

#%% [markdown]

# ## Principal component analysis
# When our data are represented by a matrix too large (the number of dimensions is too high), it is difficult to extract the most interesting features and find correlations among them; moreover the space occupied is very high. PCA is a technique that allows to achieve dimensionality reduction while preserving the most important differences among samples. 
# 
# This transformation is defined in such a way that the first principal component has the largest possible variance (that is, accounts for as much of the variability in the data as possible), and each succeeding component in turn has the highest variance possible under the constraint that it is orthogonal to the preceding components. The resulting vectors (each being a linear combination of the variables and containing n observations) are an uncorrelated orthogonal basis set.
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
        ),
        legend=dict(
            x=0,
            y=1,
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename='basic-bar')

def compress_data(X_dataset, n_components, plot_comp=False):
    
    """
    Performs pca reduction of a dataset.

    :param (array of arrays) X_dataset: Dataset to reduce
    :param (int) n_components: N components to project on 
    :param (bool) plot_comp: Plot explained variance

    :returns (pandas dataframe) X_df_reduced: pandas dataframe with reduced dataset
    """   

    pca = PCA(random_state=RANDOM_SEED)
    projected_data = pca.fit_transform(X_dataset)

    if plot_comp:
        plot_cumulative_variance(pca)

    n_comp = n_components
    pca.components_ = pca.components_[:n_comp]
    reduced_data = np.dot(projected_data, pca.components_.T)
    X_df_reduced = pd.DataFrame(reduced_data, columns=["PC#%d" % (x + 1) for x in range(n_comp)])
    return X_df_reduced
    
#%%

X_df_reduced = compress_data(X_dataset=X_scaled_data,
                             n_components=9,
                             plot_comp=True)

X_df_drop_reduced = compress_data(X_dataset=X_scaled_drop_data,
                                  n_components=9,
                                  plot_comp=False)

X_df_reduced.head(4)
#%% [markdown]
# From the graph we can see that the first 9 components retain almost 80% of total variance, while last 5 not even 2%. We then choose to select first nine of them. 
# 
# This allows us to work on a smaller dataset achieving similar results, because most of the informaion is maintained.
# 
# Let's now project our samples on the found components.
#%% 
'''
N=pre_data.values
pca = PCA(n_components=2)
x = pca.fit_transform(N)

kmeans = KMeans(n_clusters=2, random_state=RANDOM_SEED)
X_clustered = kmeans.fit_predict(N)
print(len(np.where(X_clustered == 0)[0]))
print(len(np.where(X_clustered == 1)[0]))

ed_idx = np.where(X_clustered == 0)
po_idx = np.where(X_clustered == 1)

p1 = go.Scatter(
    x=np.take(x[:,0], indices=ed_idx)[0],
    y=np.take(x[:,1], indices=ed_idx)[0],
    mode='markers',
    name="Edible",
    marker=dict(
        color=PLOTLY_COLORS[0],
    ),
    opacity=PLOTLY_OPACITY)

p2 = go.Scatter(
    x=np.take(x[:,0], indices=po_idx)[0],
    y=np.take(x[:,1], indices=po_idx)[0],
    mode='markers',
    name="Poisonous",
    marker=dict(
        color=PLOTLY_COLORS[1],
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
'''
#%% [markdown]

# Using K-means we are able to separate two classes using the two components with maximum variance.
#%% [markdown]

## Classification
# Now we are going to explore different classification methods, and see in the end the one that performs better. 
#
# Now, before starting the classification phase, let's see what kind of pre-processed data it is better to use to achieve the best classification possible.
# Due to the fact that our dataset is pretty small, probably the dimensionality reduction using PCA is not strictly necessary, but if we can achieve a similar score using just main principal components, it is definitely better.
#
# We are going to compare results of a classification method on the different datasets. In this way we can choose the one to pick for the next phase. 
# The current versions of the dataset are:
#
# 1. Full dataset.
# 2. Dataset with missing values removed.
# 3. Reduced dataset by means of PCA.
#
#
# Our dataset is pretty balanced, so we do not need any over or under-sampling technique. If we will perform poorly in classification, we could try to use some ensemble learning methods, but they should not be necessary.
# Let's start with splitting the datasets in train and test.

#%%

X_train, X_test, y_train, y_test = train_test_split(X_scaled_data, y_data, test_size=0.2, random_state=RANDOM_SEED)
X_train_pc, X_test_pc, y_train_pc, y_test_pc = train_test_split(X_df_reduced, y_data, test_size=0.2, random_state=RANDOM_SEED)
X_train_drop, X_test_drop, y_train_drop, y_test_drop = train_test_split(X_scaled_drop_data, y_drop_data, test_size=0.2, random_state=RANDOM_SEED)
X_train_pc_drop, X_test_pc_drop, y_train_pc_drop, y_test_pc_drop = train_test_split(X_df_drop_reduced, y_drop_data, test_size=0.2, random_state=RANDOM_SEED)
X_train_no_stalk, X_test_no_stalk, y_train_no_stalk, y_test_no_stalk = train_test_split(X_scaled_no_stalk, y_no_stalk, test_size=0.2, random_state=RANDOM_SEED)

#%% [markdown]
# The method used to pick the "best" dataset will be Logistic Regression, and we will tune its parameters using a grid search cross validation. 
# 
# Let's start defining the functions that we are going to use:

#%%

def print_gridcv_scores(grid_search, n=5):
    """
    Prints the best score achieved by a grid_search, alongside with its parametes

    :param (estimator) clf: Classifier object
    :param (int) n: Best n scores 
    """    

    if not hasattr(grid_search, 'best_score_'):
        raise KeyError('grid_search is not fitted.')
    
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
               
@watcher
def param_tune_grid_cv(clf, params, X_train, y_train, cv):
    """
    Function that performs a grid search over some parameters

    :param (estimator) clf: Classifier object
    :param (dictionary) params: parameters to be tested in grid search
    :param (array-like) X_train: List of data to be trained with
    :param (array-like) y_train: Target relative to X for classification or regression
    :param (cross-validation generator) cv: Determines the cross-validation splitting strategy
    """   
    pipeline = Pipeline([('clf', clf)])
    grid_search = GridSearchCV(estimator=pipeline, 
                               param_grid=params, 
                               cv=cv, 
                               n_jobs=-1,       # Use all processors
                               scoring='f1',    # Use f1 metric for evaluation
                               return_train_score=True)
    grid_search.fit(X_train, y_train)
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
    trace3 = go.Scatter(
        x=train_sizes, 
        y=train_scores_mean, 
        showlegend=True,
        name="Train score",
        line = dict(
            color = PLOTLY_COLORS[0],
        ),
    )
    
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
# This cross-validation object is a variation of KFold that returns stratified folds. The folds are made by preserving the percentage of samples for each class. Stratification is generally a better scheme, both in terms of bias and variance, when compared to regular cross-validation.
#
# Then we define our LogisticRegression classifier, setting the `random_state` parameter to our usual seed value, in such a way that we can classify each time in the same way.
#%% 
kf = StratifiedKFold(n_splits=5, random_state=RANDOM_SEED)
clf_lr = LogisticRegression(random_state=RANDOM_SEED)

#%% [markdown]
# Then we perform a grid search over all the parameters of the LogisticRegressor model.
#
# They are:
# - `liblinear` for solver, which is better for smaller datasets
# - `C` different values of this, ranging from 0.01 to 100. A smaller value inidicates stronger regularization, like in svms.
# - `penalty` "l1" and "l2" penalty for regularization, which are defined as:
#   - l1, it penalizes every mistake at the same way
#     $$ 
#     S = \Sigma_{i=1}^{n}{|y_i - f(x_i)|}
#     $$
#   - l2, it penalizes bigger values
#     $$ 
#     S = \Sigma_{i=1}^{n}{(y_i - f(x_i))^2}
#     $$
#     - Where $y_i$ is the true label and $f(x_i)$ is the assigned label


#%%

X_train, y_train = shuffle(X_train, y_train, random_state=RANDOM_SEED)
X_train_drop, y_train_drop = shuffle(X_train_drop, y_train_drop, random_state=RANDOM_SEED)
X_train_no_stalk, y_train_no_stalk = shuffle(X_train_no_stalk, y_train_no_stalk, random_state=RANDOM_SEED)

X_train_pc, y_train_pc = shuffle(X_train_pc, y_train_pc, random_state=RANDOM_SEED)
X_train_pc_drop, y_train_pc_drop = shuffle(X_train_pc_drop, y_train_pc_drop, random_state=RANDOM_SEED)



#%%

print("Full dataset cv:")
gs_full = param_tune_grid_cv(clf_lr, LOGISTIC_REGRESSION_PARAMS, X_train, y_train, kf)
print("\nDataset projected on first 9 pc cv:")
gs_pc = param_tune_grid_cv(clf_lr, LOGISTIC_REGRESSION_PARAMS, X_train_pc, y_train_pc, kf)
print("\nFull dataset with dropped values took:")
gs_drop = param_tune_grid_cv(clf_lr, LOGISTIC_REGRESSION_PARAMS, X_train_drop, y_train_drop, kf)
print("\nProjected dataset with dropped values took:")
gs_pc_drop = param_tune_grid_cv(clf_lr, LOGISTIC_REGRESSION_PARAMS, X_train_pc_drop, y_train_pc_drop, kf)
print("\nFull dataset without stalk-root column:")
gs_no_stalk = param_tune_grid_cv(clf_lr, LOGISTIC_REGRESSION_PARAMS, X_train_no_stalk, y_train_no_stalk, kf)

gss = [gs_full, gs_pc, gs_drop, gs_pc_drop, gs_no_stalk]

test_results = score(gss, [(X_test, y_test), 
                           (X_test_pc, y_test_pc), 
                           (X_test_drop, y_test_drop), 
                           (X_test_pc_drop, y_test_pc_drop),
                           (X_test_no_stalk, y_test_no_stalk)])

#%% 
X_train.shape

#%% [markdown]
# This is the score of the different classification on the test set:
#%%
dataset_strings = ["full dataset", 
                   "dataset with first 9 principal components", 
                   "dataset with dropped missing values",
                   "dataset with dropped missing value reduced with first 9 principal components",
                   "dataset with stalk-root field dropped"]
method_strings = ["without any method"]

t = PrettyTable()
t.field_names = ["Score", "Dataset", "Type"]

result_row = []
for ms, results in zip(method_strings, [test_results]):
    for ds, res in zip(dataset_strings, results):
        result_row.append(["%.3f" % res, ds, ms])
        
result_row = sorted(result_row, key=lambda kv: kv[0], reverse=True)

for k in result_row:
    t.add_row(k)

print(t)

#%% [markdown]
# We can notice that the classfication score achieved with the dataset in which the missing values were
# removed is the best one. Probably beacuse, with the full dataset, the classifier learnt some 
# not existent correlations between the missing value encoded and the other features. 
# Moreover, being the dataset easy to classify overall, reducing the amount of samples does not
# cause issues on the training of our model.
# 
# In any case, looking at the performances, there is a big difference between the full dataset
# and the one containing only principal components; due to this, in the next steps we are going to
# use the reduced dataset, beacuse we can achieve an high score saving a lot of time.

#%%
print_gridcv_scores(gs_drop)

#%%
print_confusion_matrix(gs_drop, X_test_drop, y_test_drop)

#%% 
plot_learning_curve(gs_drop.best_estimator_, "Learning Curve of Logistic Regression", 
                    np.concatenate((X_train_drop, X_test_drop)),
                    np.concatenate((y_train_drop, y_test_drop)), 
                    cv=5)

#%% [markdown]
# ### Support vector machine

#%%
clf_svm = SVC(probability=True, random_state=RANDOM_SEED)
gs_pc_svm = param_tune_grid_cv(clf_svm, SVM_PARAMS, X_train_pc, y_train_pc, kf)
print_gridcv_scores(gs_pc_svm, n=5)

#%%
plot_learning_curve(gs_pc_svm.best_estimator_, "Learning curve of SVM", 
                    np.concatenate((X_train_pc, X_test_pc)),
                    np.concatenate((y_train_pc, y_test_pc)),
                    cv=5)

#%%
print_confusion_matrix(gs_pc_svm, X_test_pc, y_test_pc)
#%% [markdown]
# We can notice that most of the times, the only mistakes are poisonous mushrooms classified as edible. These mistakes weight must be much higher with respect to an edible mushroom classified as poisonous, because there is not any danger in that case.

#%%
clf_nb = GaussianNB()
clf_nb.fit(X_train, y_train)
print_raw_score(clf_nb, X_test, y_test)
print_confusion_matrix(clf_nb, X_test, y_test)


#%%
plot_learning_curve(clf_nb, "Learning curve of GaussianNB", 
                    np.concatenate((X_train, X_test), axis=0), 
                    np.concatenate((y_train, y_test), axis=0), 
                    cv=5)

#%%

clf_pc_rf = RandomForestClassifier(random_state=RANDOM_SEED)
gs_pc_rf = param_tune_grid_cv(clf_pc_rf, RANDOM_FOREST_PARAMS, X_train_pc, y_train_pc, kf)
print_gridcv_scores(gs_pc_rf, n = 5)

#%%
print_confusion_matrix(gs_pc_rf, X_test_pc, y_test_pc)

#%%
plot_learning_curve(gs_pc_rf.best_estimator_, "Learning curve of Random Forest Classifier", 
                    np.concatenate((X_train_pc, X_test_pc)),
                    np.concatenate((y_train_pc, y_test_pc)), 
                    cv=5)

#%%
feature_importance = np.array(  sorted(zip(X_train_pc.columns, 
                                gs_pc_rf.best_estimator_.named_steps['clf'].feature_importances_),
                                key=lambda x: x[1], reverse=True))
plot_feature_importance(feature_importance, "Feature importance in the random forest")


#%%
'''
print("Full dataset cv:")
gs_full = param_tune_grid_cv(clf_svm, SVM_PARAMS, X_train, y_train, kf)
print("\nDataset projected on first 9 pc cv:")
gs_pc = param_tune_grid_cv(clf_svm, SVM_PARAMS, X_train_pc, y_train_pc, kf)
print("\nFull dataset with dropped values took:")
gs_drop = param_tune_grid_cv(clf_svm, SVM_PARAMS, X_train_drop, y_train_drop, kf)
gss = [gs_full, gs_pc, gs_drop]

test_results = score(gss, [(X_test, y_test), (X_test_pc, y_test_pc), (X_test_drop, y_test_drop)])
'''
#%%

clf_knn = KNeighborsClassifier()
gs_knn = param_tune_grid_cv(clf_knn, KNN_PARAMS, X_train_pc, y_train_pc, kf)
print_gridcv_scores(gs_knn, n=5)

'''
clf_knn = KNeighborsClassifier()
gs_knn = param_tune_grid_cv(clf_knn, KNN_PARAMS, X_train, y_train, kf)
print_gridcv_scores(gs_knn, n=5)
'''
#%%
print_confusion_matrix(gs_knn, X_train_pc, y_train_pc)

#%%

plot_learning_curve(gs_knn.best_estimator_, "Learning curve of Random Forest Classifier", 
                    np.concatenate((X_train_pc, X_test_pc)),
                    np.concatenate((y_train_pc, y_test_pc)), 
                    cv=5)


#%% [markdown]
# The K-NN classification with the whole dataset gives the same result but it takes more than 7 times more time

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

classifiers = [gs_drop, gs_pc_svm, clf_nb, gs_pc_rf, gs_knn]
classifier_names = ["Logistic Regression", "SVM", "GaussianNB", "Random Forest", "KNN"]
auc_scores, roc_plot = plot_roc_curve(classifiers, classifier_names, "ROC curve", X_test, y_test)
roc_plot

#%%
print_performances(classifiers, classifier_names, auc_scores, X_test_pc, y_test_pc)

#%% [markdown]

# To have a little bit more fun we will try one more thing. Due to the fact that some charateristics of mushrooms are subjective, like odor for example, and some others need more advanced analysis, like spore print, we will try to use our best-performing classification algorithm on a reduced version of the dataset, keeping only the features understendable by every person without the need of any specific equipment of knowledge.

#%%
pre_data.columns


#%%
data_vis = pre_data.drop(['odor', 'spore-print-color'], axis=1)
data_vis = data_vis[data_vis['stalk-root'] != le_mapping['stalk-root']['?']]

data_vis.shape

X_data_vis, y_data_vis = dataframe_to_array(data_vis)
X_data_vis = scale_data(X_data_vis)

pca = PCA(random_state=RANDOM_SEED)
proj_data = pca.fit_transform(X_data_vis)
tot_var = np.sum(pca.explained_variance_)
ex_var = [(i / tot_var) * 100 for i in sorted(pca.explained_variance_, reverse=True)]
cum_ex_var = np.cumsum(ex_var)
n_comp = 9
pca.components_ = pca.components_[:n_comp]
reduced_data = np.dot(proj_data, pca.components_.T)
# pca.inverse_transform(projected_data)
X_vis_reduced = pd.DataFrame(reduced_data, columns=["PC#%d" % (x + 1) for x in range(n_comp)])

#%%
X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(X_vis_reduced, y_data_vis, test_size=0.2, random_state=RANDOM_SEED)

clf_svm = SVC(probability=True, random_state=RANDOM_SEED)
gs_pc_svm = param_tune_grid_cv(clf_svm, SVM_PARAMS, X_train_vis, y_train_vis, kf)
print_gridcv_scores(gs_pc_svm, n=5)

#%%
plot_learning_curve(gs_pc_svm.best_estimator_, "Learning curve of SVM", 
                    np.concatenate((X_train_vis, X_test_vis)),
                    np.concatenate((y_train_vis, y_test_vis)),
                    cv=5)

#%%
print_confusion_matrix(gs_pc_svm, X_test_vis, y_test_vis)

#%% [markdown]
# SVM performs really well also in this case. Those mushrooms are easily classified by our model.