# Credit card fraud detection

Project for my Intelligent Systems course at university. I took a dataset of 1 million credit card transactions and tried to predict which ones are fraud, and to find out what makes a transaction look suspicious.

The data is from Kaggle: https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud
Download `card_transdata.csv` and put it in the same folder as the notebook (I didn't upload it because of its size).

## What's in the notebook
- Looking at the data first. About 8.7% of the transactions are fraud. Online orders are much riskier than in-person ones (12.7% fraud vs 1.3%), and paying with a PIN almost removes the problem (0.27%).
- A 70/30 train/test split that keeps the fraud ratio the same in both parts.
- Models: logistic regression, decision tree, random forest and XGBoost. I trained each tree-based one with default settings and again with tuned settings (grid search or random search), plus a stacking model on top of the three.
- Evaluation with sensitivity, specificity, F1, balanced accuracy and AUC. Plain accuracy doesn't say much here because 91% of the transactions are not fraud.
- Feature importance and SHAP plots to see what drives the predictions, and a look at one legitimate and one fraudulent transaction.

## Results (test set)
| Model | Sensitivity | Specificity | AUC |
|---|---|---|---|
| Logistic regression | 0.5983 | 0.9930 | 0.9681 |
| XGBoost (tuned) | 0.9904 | 0.9991 | 1.0000 |
| Random forest (tuned) | 0.9998 | 1.0000 | 1.0000 |
| Stacking | 0.9998 | 1.0000 | 1.0000 |

The most important factors were the ratio of the purchase to the cardholder's usual spending, whether the order was online, and the distance from home.

## Notes
- The scores are almost perfect because the dataset is synthetic. Real fraud data is much messier, so I wouldn't expect numbers like this outside of a course project.
- Logistic regression did much worse than the tree-based models and missed about 40% of the fraud.

## Running it
```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap jupyter
jupyter notebook credit_card_fraud.ipynb
```
