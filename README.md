# Mushroom Analysis
Data analysis of https://www.kaggle.com/uciml/mushroom-classification for the Data Spaces course at
Polytechnic of Turin.

## Preprocessing
In the first phase I performed different steps to prepare data for the other phases:

  - Encoded or removed missing values.
  - Removed useless features.
  - Use of the plotly library for box plots, bar charts, heatmap, dendogram.
  - Data scaling.

## Dimensionality Reduction

  - Use of PCA, selection and projection of data on principal components
  - Cumulative variance plot
  
## Classification

  - Use of K-fold cross validation on different models to choose the best one
  - Parameter tuning, theory, ROC curve, Model evaluation, learning curve for:
    - k-NN 
    - SVM 
    - Random Forest 
    - Logistic Regression
