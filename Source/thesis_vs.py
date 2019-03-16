#%% [markdown]
# Safe to eat or deadly poisonous?
### An analysis on mushroom classification by Lorenzo Santolini

#%% [markdown]
This is a little code to import automatically the dataset into google colab. 
Provide your kaggle's API key (profile section) when file requested

#### Code snippet for google colab

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
# Constants
PLOTLY_COLORS = ['#140DFF', '#FF0DE2']
COLOR_PALETTE = ['#FF0DE2', '#140DFF', '#CAFFD0', '#C9E4E7', '#B4A0E5', '#904C77']
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
    'clf__C': [0.01, 0.1, 1, 10, 100], # smaller value, stronger regularization
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
    'clf__max_features': ["sqrt", "log2"], # same as auto
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
This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom. Each species is identified as edible or poisonous. Rows are composed by 23 different fields, each one of them identifying a specific charateristic:

- Class: poisonous=p, edible=e
- Cap-shape: bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s
- Cap-surface: fibrous=f, grooves=g, scaly=y, smooth=s
- Cap-color: brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y
- Bruises: bruises=t, no=f
- Odor: almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s
- Gill-attachment: attached=a, descending=d, free=f, notched=n
- Gill-spacing: close=c, crowded=w, distant=d
- Gill-size: broad=b, narrow=n
- Gill-color: black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o, pink=p, purple=u,red=e, white=w, yellow=y
- Stalk-shape: enlarging=e, tapering=t
- Stalk-root: bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=?
- Stalk-surface-above-ring: fibrous=f, scaly=y, silky=k, smooth=s
- Stalk-surface-below-ring: fibrous=f, scaly=y, silky=k, smooth=s
- Stalk-color-above-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y
- Stalk-color-below-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y
- Veil-type: partial=p, universal=u
- Veil-color: brown=n, orange=o, white=w, yellow=y
- Ring-number: none=n, one=o, two=t
- Ring-type: cobwebby=c, evanescent=e, flaring=f, large=l, none=n, pendant=p, sheathing=s, zone=z
- Spore-print-color: black=k, brown=n, buff=b, chocolate=h, green=r, orange=o, purple=u, white=w, yellow=y
- Population: abundant=a, clustered=c, numerous=n, scattered=s, several=v, solitary=y
- Habitat: grasses=g, leaves=l, meadows=m, paths=p, urban=u, waste=w, woods=d

This analysis was conducted in Python 3.7.1 using Jupyter Notebook allows you to combine code, comments, multimedia, and visualizations in an interactive document — called a notebook, naturally — that can be shared, re-used, and re-worked. In addition, the following packages were used:

- sklearn
- pandas
- numpy
- plotly

#%%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

import plotly
import plotly.plotly as py
from plotly.plotly import plot, iplot
import plotly.graph_objs as go
import plotly.figure_factory as ff

from scipy.cluster import hierarchy as hc

from imblearn.pipeline import make_pipeline, Pipeline
from imblearn.over_sampling import SMOTE

from prettytable import PrettyTable
from functools import wraps
import time

plotly.tools.set_credentials_file(username='modusV', api_key='OBKKnTR2vYTeKIOKtRU6')

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

#%% [markdown]
## Dataset analysis and preprocessing
Let's start importing the data:
#%%
# Load the dataset
dataset = pd.read_csv("./Input/mushrooms.csv")
# dataset = pd.read_csv("./mushrooms.csv")
print("The dataset has %d rows and %d columns." % dataset.shape)
#%% [markdown]
Now we will look at the dataset to understand what are the different fields and their types:
#%%
# Count number of classes for classification
print(f"There are {dataset['class'].unique().size} different classes:"
      f"\n {dataset['class'].unique().tolist()}")

# Count number of unique data for every column
print(f"Unique values for every field: \n{dataset.nunique()}")

#%%
# See data types 
print(f"Data types: \n{dataset.head(5)}")

# All columns 
print(", ".join(str(a) for a in dataset.columns))
#%% [markdown]
From the above snippet we can notice that the fields are all string values; converting them to numeric values can make our analysis much easier. We will take care of this in the next phase.
#%% [markdown]
## Preprocessing
We need to pre-process our data to 

#%% 
n_columns_original = len(dataset.columns)
to_drop = [col for col in dataset.columns if dataset[col].nunique() == 1]
dataset.drop(to_drop, axis=1, inplace=True)

print(f"{n_columns_original - len(dataset.columns)} not significant columns have been removed")


#### Handling missing values
#%%
# Check if any field is null
if dataset.isnull().any().any():
    print("There are some null values")
else:
    print("There are no null values")
#%% [markdown]
It may seem that we have no missing value from the previous analysis, but if we look better,from the data description we can notice that in the field stalk-root there are some missing values, marked with the question mark; let's count how many of them there are:

#%%
print("There are " + str((dataset['stalk-root'] == "?").sum()) + " missing values in stalk-root column")
# df_drop = dataset[dataset['stalk-root'] != "?"]

#%% [markdown]
When we find missing values in a dataset, there are some of the approaches that can be considered:

1. Delete all rows containing a missing value
2. Substitute with a constant value that has meaning within the domain, such as 0, distinct from all other values.
3. Substitute with a value from another randomly selected record.
4. Substitute with mean, median or mode value for the column.
5. Substitute with a value estimated by another predictive model.

It is evident from the `dataset.head()` function that our fileds are composed by all string values. Given the fact that we would need to translate in any case every field to a numeric one, to better display them in graphs, a simple approach is to keep the missing data as a peculiar number different from the others, and simply apply the transformation as they were present.

We will use the LabelEncoder from the sklearn library, which allows us to perform this mapping:


#%%
def preprocess(dataset):
    mapping = {}  
    d = dataset.copy()
    labelEncoder = LabelEncoder()
    for column in dataset.columns:
        labelEncoder.fit(dataset[column])
        mapping[column] = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
        d[column] = labelEncoder.transform(dataset[column])
        
    return d, labelEncoder, mapping

le = 0
pre_data, l_encoder, le_mapping = preprocess(dataset)

# Check mapping
print(le_mapping)

# Check new data
pre_data.head(5)

#%% [markdown]
Now let's check if there is any column not useful for classification, 
where the values are all the same (zero variance)


#%%
# Check new labels
print(pre_data.groupby('class').size())
#%% [markdown]
We can notice that data have been transformed, and now the labels are represented with a 0/1 integer value. 
Now we can look deeper into some statistical details about the dataset, using the `dataset.describe` command on our pandas DataFrame dataset. The output describes:

- count: number of samples (rows)
- mean: the mean of the attribute among all samples
- std: the standard deviation of the attribute
- min: the minimal value of the attribute
- 25%: the lower percentile
- 50%: the median
- 75%: the upper percentile
- max: the maximal value of the attribute
#%%
# Get insights on the dataset
dataset.describe()

#%%
y = dataset["class"].value_counts()
print(y)
class_dict = ["edible", "poisonous"]

#%% [markdown]
This is the class distribution:
#%%
data = [go.Bar(
            x=class_dict,
            y=y,
            marker=dict(
            color=PLOTLY_COLORS),
            opacity=PLOTLY_OPACITY,
    )]

layout = go.Layout(title="Class distribution",
                   autosize=False,
                   width=400,
                   height=400,
                   yaxis=dict(
                        title='N. samples',
                    ),
                   )
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='color-bar')

#%% [markdown]

At this point we can analyze the distribution of our data using a boxplot. 
A boxplot is a standardized way of displaying the distribution of data based on a 
five number summary (“minimum”, first quartile (Q1), median, third quartile (Q3), and “maximum”). 
It can tell you about your outliers and what their values are. 
It can also tell you if your data is symmetrical, how tightly your data is grouped, 
and if and how your data is skewed.
The information that we can find in a box plot are:

- **median** (Q2/50th Percentile): the middle value of the dataset.
- **first quartile** (Q1/25th Percentile): the middle number between the smallest number (not the “minimum”) and the median of the dataset.
- **third quartile** (Q3/75th Percentile): the middle value between the median and the highest value (not the “maximum”) of the dataset.
- **interquartile range** (IQR): 25th to the 75th percentile.
- **outliers** (shown as green circles)
- **maximum**: Q3 + 1.5*IQR
- **minimum**: Q1 -1.5*IQR

It makes no sense showing binary or with few different values fields, so we are going to filter them before plotting.

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

From the boxplot above, we can see that the colour and the shape of the 
cap are not an effective parameter to decide whether a mushroom is poisonous or edible, 
because their plots are very similar (same median and very close distribution). 
The gill color instead, is more significant; 

#%% [markdown]

Let's investigate the correlation between variables:

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

TODO: Give some impressions on the heatmap

#%%

def create_hist(type, data, col, visible=False):
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
hist_edible = [(create_hist("edible", edible, col, False) if i != active_index 
               else create_hist("edible", edible, col, True)) 
              for i, col in enumerate(hist_features)]

hist_poisonous = [(create_hist("poisonous", poisonous, col, False) if i != active_index 
               else create_hist("poisonous", poisonous, col, True)) 
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
py.iplot(fig, filename='hist_slider')

#%% [markdown]
TODO give impressions on the histogram

#%% 

X = np.random.rand(10, 10)
names = pre_data.columns
inverse_correlation = 1 - abs(pre_data.corr()) # This is the 'dissimilarity' method
fig = ff.create_dendrogram(inverse_correlation.values, 
                           labels=names, 
                           colorscale=PLOTLY_COLORS, 
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
TODO: comment the graph, every feature is useful

Now we standardize the data 

#%%

from sklearn.preprocessing import StandardScaler


drop_data = pre_data[pre_data['stalk-root'] != le_mapping['stalk-root']['?']]

y_data = pre_data['class']
y_drop_data = drop_data['class']

X_pre_data = pre_data.drop(['class'], axis=1)
X_drop_data = drop_data.drop(['class'], axis=1)

scaler = StandardScaler()

X_scaled_data = scaler.fit_transform(X_pre_data)
X_scaled_drop_data = scaler.fit_transform(X_drop_data)


#%% [markdown]
TODO: introduce PCA

#%% 

from sklearn.decomposition import PCA


pca = PCA(random_state=RANDOM_SEED)
projected_data = pca.fit_transform(X_scaled_data)

tot_var = np.sum(pca.explained_variance_)
ex_var = [(i / tot_var) * 100 for i in sorted(pca.explained_variance_, reverse=True)]
cum_ex_var = np.cumsum(ex_var)



#%%
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

#%% [markdown]
From the graph we can see that the first 9 components retain almost 80% of 
total variance, while last 5 not even 2%. We then choose to select first 
nine of them.

This is the reduced dataset:

#%%

n_comp = 9
pca.components_ = pca.components_[:n_comp]
reduced_data = np.dot(projected_data, pca.components_.T)
# pca.inverse_transform(projected_data)
X_df_reduced = pd.DataFrame(reduced_data, columns=["PC#%d" % (x + 1) for x in range(n_comp)])

#%% 

from sklearn.cluster import KMeans

N=pre_data.values
pca = PCA(n_components=2)
x = pca.fit_transform(N)

kmeans = KMeans(n_clusters=2, random_state=RANDOM_SEED)
X_clustered = kmeans.fit_predict(N)
print(kmeans.score)

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

#%% [markdown]

TODO: give some impressions on the clustered data

Using K-means we are able to separate two classes using the two components with maximum variance.

#%% [markdown]

## Classification
Now we are going to explore different classification methods, and see in the end the one that performs better

#%% [markdown]
Now, before starting the classification phase, let's see what kind of pre-processed
data it is better to use to achieve the best classification possible.
Due to the fact that our dataset is pretty small, probably the dimensionality reduction
using PCA is not necessary.

We are going to compare results of a classification method on the different datasets.
In this way we can choose the one to pick for the 
next phase. The  current versions of the dataset are:

- Full dataset
- Dataset reduced using first 9 principal components
- Full dataset with missing values removed (question marks in the stalk-root field)

Our dataset is pretty balanced, so we do not need any over or under-sampling
technique. If we will perform poorly in classification, we could try to use
ensemble learning methods.
Let's start with splitting the datasets in train and test.

Bagging, boosting and ensemble learning models
> https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/
> MSMOTE
> Over-Under sampling 
> SMOTE

#%%

X_train, X_test, y_train, y_test = train_test_split(X_scaled_data, y_data, test_size=0.2, random_state=RANDOM_SEED)
X_train_pc, X_test_pc, y_train_pc, y_test_pc = train_test_split(X_df_reduced, y_data, test_size=0.2, random_state=RANDOM_SEED)
X_train_drop, X_test_drop, y_train_drop, y_test_drop = train_test_split(X_scaled_drop_data, y_drop_data, test_size=0.2, random_state=RANDOM_SEED)

#%% [markdown]
The first method used will be Logistic Regression, and we will tune its parameters
using a grid search cross validation. Let's start defining the functions
that we are going to use:

#%%

def print_gridcv_scores(grid_search, n=5):
    
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
    scores = []
    for c, (X_test, y_test) in zip(clfs, datasets):
        scores.append(c.score(X_test, y_test))

    return scores

#%%
def hexToRGBA(hex, alpha):
    r = int(hex[1:3], 16)
    g = int(hex[3:5], 16)
    b = int(hex[5:], 16)

    if alpha:
        return "rgba(" + str(r) + ", " + str(g) + ", " + str(b) + ", " + str(alpha) + ")"
    else:
        return "rgb(" + str(r) + ", " + str(g) + ", " + str(b) + ")"


# partially based on https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
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
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring="f1", random_state=RANDOM_SEED)
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
    print("Score achieved by NB: %0.3f" % (score([clf], [(X_test, y_test)])[0]))


def plot_feature_importance(feature_importance, title):
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
#%% 
kf = StratifiedKFold(n_splits=5, random_state=RANDOM_SEED)
clf_lr = LogisticRegression(random_state=RANDOM_SEED)
clf_lr_balanced = LogisticRegression(random_state=RANDOM_SEED, class_weight="balanced")


#%%

print("Full dataset cv:")
gs_full = param_tune_grid_cv(clf_lr, LOGISTIC_REGRESSION_PARAMS, X_train, y_train, kf)
print("\nDataset projected on first 9 pc cv:")
gs_pc = param_tune_grid_cv(clf_lr, LOGISTIC_REGRESSION_PARAMS, X_train_pc, y_train_pc, kf)
print("\nFull dataset with dropped values took:")
gs_drop = param_tune_grid_cv(clf_lr, LOGISTIC_REGRESSION_PARAMS, X_train_drop, y_train_drop, kf)
gss = [gs_full, gs_pc, gs_drop]

test_results = score(gss, [(X_test, y_test), (X_test_pc, y_test_pc), (X_test_drop, y_test_drop)])

#%%

print("Full dataset cv:")
gs_full_balanced = param_tune_grid_cv(clf_lr_balanced, LOGISTIC_REGRESSION_PARAMS, X_train, y_train, kf)
print("\nDataset projected on first 9 pc cv:")
gs_pc_balanced = param_tune_grid_cv(clf_lr_balanced, LOGISTIC_REGRESSION_PARAMS, X_train_pc, y_train_pc, kf)
print("\nFull dataset with dropped values took:")
gs_drop_balanced = param_tune_grid_cv(clf_lr_balanced, LOGISTIC_REGRESSION_PARAMS, X_train_drop, y_train_drop, kf)
gss_balanced = [gs_full_balanced, gs_pc_balanced, gs_drop_balanced]

test_results_balanced = score(gss_balanced, [(X_test, y_test), (X_test_pc, y_test_pc), (X_test_drop, y_test_drop)])

#%% 
X_train.shape

#%%
dataset_strings = ["full dataset", "dataset with first 9 principal components", "dataset with dropped missing values"]
method_strings = ["without any balancing", "using balanced class weights"]

t = PrettyTable()
t.field_names = ["Score", "Dataset", "Type"]

result_row = []
for ms, results in zip(method_strings, [test_results, test_results_balanced]):
    for ds, res in zip(dataset_strings, results):
        result_row.append(["%.3f" % res, ds, ms])
        
result_row = sorted(result_row, key=lambda kv: kv[0], reverse=True)

for k in result_row:
    t.add_row(k)

t.title = "F1 score  dataset and method"
print(t)

#%% [markdown]
So the dataset we will use for all the classification methods will be 
the one with missing values removed.

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
### Support vector machine

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
We can notice that most of the times, the only mistakes are
poisonous mushrooms classified as edible. These mistakes weight
must be much higher with respect to an edible mushroom classified as
poisonous, because there is not any danger in that case.

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

#%%
print_confusion_matrix(gs_knn, X_train_pc, y_train_pc)

#%%

plot_learning_curve(gs_knn.best_estimator_, "Learning curve of Random Forest Classifier", 
                    np.concatenate((X_train_pc, X_test_pc)),
                    np.concatenate((y_train_pc, y_test_pc)), 
                    cv=5)



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

#%% [markdown]

To have a little bit more fun we will try one more thing. Due to the fact that some 
charateristics of mushrooms are subjective, like odor for example, and some others need 
more advanced analysis, like spore print, we will try to use our best-performing 
classification algorithm on a reduced version of the dataset, keeping only the features 
understendable by every person without the need of any specific equipment of knowledge.

#%%
pre_data.columns


#%%
data_vis = pre_data.drop(['odor', 'spore-print-color'], axis=1)
data_vis = data_vis[data_vis['stalk-root'] != le_mapping['stalk-root']['?']]
data_vis.shape


#%%

accs = []
recalls = []
precision = []
results_table = pd.DataFrame(columns=["accuracy", "precision", "recall", "f1", "auc"])
for (i, clf), name, auc in zip(enumerate(classifiers), classifier_names, auc_scores):
    y_pred = clf.predict(X_test_pc)
    row = []
    row.append(accuracy_score(y_test_pc, y_pred))
    row.append(precision_score(y_test_pc, y_pred))
    row.append(recall_score(y_test_pc, y_pred))
    row.append(f1_score(y_test_pc, y_pred))
    row.append(auc)
    row = ["%.3f" % r for r in row]
    results_table.loc[name] = row
results_table
