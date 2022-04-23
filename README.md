<h1 align=center> Wine Quality </h1>
<p align=center> Miniproject for NTU SC1015 </p>
<p align=center> Group 9, Lab Group SC2 </p>

<p align="center">
  <a>Zhang Danxu</a> •
  <a>Lohia Vardhan</a> •
  <a>Sannabhadti Shikha Deepak</a>
</p>

--- 

## Overview 💻
This miniproject focused on [UCI Machine Learning Repository Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/Wine+Quality), trying to use various properties of a wine to predict its quality.

## Libraries 📚
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `sklearn`
- `tabulate`
- `torch`

## File Structure 📂
- `data/`: datasets
    - `raw/winequality-red.csv`: raw dataset downloaded from UCI Website
    - `refined_wine.csv`: dataset after dopping null values, duplicated values and outliers
    - `reclassified_wine.csv`: dataset after reclassification
    - `predictors.csv`: cleaned and scaled predictor variables
    - `response.csv`: cleaned response variable
    - `winequality.txt`: metadata for dataset

-  `src/`: jupyter notebooks and python scripts
    - `data-clean-up.ipynb`: notebook for data cleaning;
    - `eda.ipynb`: notebook for exploratory data analysis
    - `ml.ipynb`: notebook for machine learning;
    - `SGD.py`: script for Stochastic Gradient Descent;
    - `loss_history.png`: learning curve of SGD, number of iterations w.r.t. loss function
- `.gitigore`: files to be ignored by git
- `LICENSE`: lincensing information
- `README.md`: basic information about our project

## Data Clean-Up 🧹
In [data clean-up](src/data-clean-up.ipynb), we first dropped null values, duplicated rows, and outliers. Since our data was imbalanced, we reclassifed our response variable `quality` to make it more balanced. Then we did feature scaling to make our predictor variables of similar scales.

## Exploratory Data Analysis 🔎
[Exploratory data anaysis](src/eda.ipynb) includes histogram with kde plots, boxplots for data before and after reclassificatin, finding corrleation between variables, and preforming Point Biserial Correlation.

## Machine Learning 🤖
In [machine learning](src/ml.ipynb), we used three algorighms: decision tree, stochastic gradient descent, and support vector machine with rbf kernel.  
We first built a decision tree of depth 4, and the training accuracy is 91% and testing accuracy is 86%. But since we had a relative large number of features, the model is a bit overfitting.  
The second model we built is stochastic gradient descent. We set learning rate to be $10^{-3}$ and do $10^5$ times iterations, and the loss funcion has reached a minimal value. [Here](src/loss_history.png) is how loss value gradually decrease as number of iterations increases. The training accuray is 89% and testing accuracy is 88%.  
The stochastic gradient descent model used a simple linear layer, but since we have a relative large number of features, it could have some non-linear relationship. So the third model we built is support vector machine with rbf kernel. To avoid overfitting, we set parameter gamma to a small value. The training accuracy is 90% and testing accuracy is 89%.  





