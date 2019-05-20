# Mushroom Analysis
Data analysis of https://www.kaggle.com/uciml/mushroom-classification for the Data Spaces course at
Polytechnic of Turin.

Notebook with output viewable at https://colab.research.google.com/drive/19ZdQ7dduPIvG0k5F607FTLf_vkhQdp37

# Description

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

# Usage
  
If your are running the notebook locally/on remote jupyter server:

  - Clone the repository
  - Download the dataset from kaggle's website (link above)
  - Enter your *plotly* username and api_key (you can find it in your profile section in plotly)
  - Run and enjoy!
  
If you are running the notebook on google colab:

  - uncomment the *Code snippet for google colab* section
  - upload your kaggle.json file (you can download it from your profile section in kaggle
  - In the *load the dataset* part follow comments innstructions and comment out the dataset load command
  - Run and enjoy!


Feel free to fork the notebook and to improve it! Just insert a reference to the original.
If you liked the work, just leave a star! Thank you :sparkles:
