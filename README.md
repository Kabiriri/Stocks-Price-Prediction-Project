# Stock Price Movement Prediction

## Overview

This project leverages historical stock market data and machine learning techniques to predict stock price movements. 

Our goal is to build a robust classification model that distinguishes between upward and downward price movements, thereby providing actionable insights for investors.

![Professional Trading Floor](Images/imagestocks.PNG)




## Business Problem & Objectives

**Business Problem:**  
Investors face challenges in forecasting stock price movements due to market volatility and complex market signals. Traditional technical analysis often falls short, so there is a need for a data-driven approach that can outperform simple heuristics.

**Project Objectives:**
- **Model Stock Price Movements:** Build a machine learning classifier to predict whether stock prices will move upward or downward.
- **Compare Models:** Evaluate and compare different models (Logistic Regression, Decision Tree, Random Forest) against baselines and traditional technical analysis.
- **Optimal Time Horizon:** Investigate whether daily, weekly, or monthly predictions yield the best results.
- **Generalizability:** Assess if the model generalizes across various stocks and sectors.
- **Financial Impact:** Determine how model predictions can translate into improved trading strategies and profitability.

## Dataset

- **Source:** Historical stock data from Yahoo Finance.
- **Time Span:** 25 years of data.
- **Stocks:** 20 diverse stocks from various sectors.
- **Features:**  
  - Price data: Open, High, Low, Close, Adjusted Close, Volume.
  - Technical indicators: Moving Averages (EMA, MA), RSI, Bollinger Bands, etc.
- **Target:** A binary indicator (0 for downward movement, 1 for upward movement).

> **Image Recommendation:**  
> Display a sample snapshot of the dataset or an annotated diagram showing key features and the target variable.

---

## Data Preprocessing & Feature Engineering

We applied rigorous preprocessing steps to ensure data quality:
- **Data Cleaning:** Removed non-numeric columns such as "Date" and "Ticker"; handled missing values.
- **Feature Engineering:** Calculated technical indicators (EMA, RSI, Bollinger Bands) and generated shifted features (e.g., Close_Day_1).
- **Feature Selection:** Utilized Recursive Feature Elimination (RFE) to identify the most predictive features.
- **Train-Test Split:** Divided the dataset into 80% training and 20% testing sets, maintaining class balance.

## Exploratory Data Analysis (EDA)

We performed extensive EDA to understand the data patterns:
- **Univariate Analysis:**  
  - Histograms and density plots of closing prices, trading volumes, and technical indicators.
- **Bivariate Analysis:**  
  - Scatter plots and box plots examining relationships between trading volume and price, and between technical signals and subsequent price changes.
- **Multivariate Analysis:**  
  - Correlation heatmaps and pairplots that reveal complex interactions among features.

> **Image Recommendation:**  
> Include screenshots of key plots such as a correlation heatmap, a time series chart of prices, and a scatter plot of volume vs. price.

---

## Modeling Approach

We experimented with several models:
- **Logistic Regression:**  
  - Achieved ~88.49% accuracy with balanced precision and recall.
- **Decision Tree:**  
  - The unpruned tree showed high accuracy (~93.06%) but suffered from overfitting.
- **Pruned Decision Tree:**  
  - Reduced overfitting achieved ~86.14% accuracy.
- **Random Forest:**  
  - Achieved ~74.59% accuracy.
  
Based on our results, the basic logistic regression model currently offers the best balance between accuracy, generalization, and interpretability.

> **Image Recommendation:**  
> Provide a comparative chart (e.g., a bar chart) summarizing accuracy and key metrics for each model.

---

## Model Evaluation

### Logistic Regression Results
- **Accuracy:** ~88.49%
- **Metrics:** Precision, recall, and f1-scores are around 0.88 for both classes.
  
### Pruned Decision Tree Results
- **Accuracy:** ~86.14%
- **Metrics:** Balanced performance, but with lower accuracy compared to logistic regression.
  
### Random Forest Results
- **Accuracy:** ~74.59%
- **Metrics:** Lower performance overall compared to logistic regression and pruned decision trees.

> **Image Recommendation:**  
> Include visualizations such as confusion matrices, classification reports, and feature importance plots.

---

## Future Work & Improvements

- **Advanced Modeling:**  
  - Explore ensemble methods (Gradient Boosting, XGBoost) and deep learning (LSTM, Transformers) for capturing non-linear dynamics.
- **Enhanced Feature Engineering:**  
  - Incorporate additional data (e.g., sentiment analysis, macroeconomic indicators).
- **Backtesting:**  
  - Simulate trading strategies using historical data to quantify the financial impact.
- **Optimization:**  
  - Fine-tune hyperparameters further and explore additional techniques for class imbalance resolution (e.g., SMOTE).

> **Image Recommendation:**  
> Use a roadmap or timeline graphic to illustrate planned future enhancements and next steps.

---

## How to Run the Project

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/stock-price-prediction.git
   cd stock-price-prediction
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Preprocess Data:**
   ```bash
   python preprocess_data.py
   ```
4. **Train Models:**
   ```bash
   python train_model.py
   ```
5. **Evaluate Models:**
   ```bash
   python evaluate_model.py
   ```
6. **View Visualizations:**
   - Plots are saved in the `images/` folder or can be viewed by running the visualization scripts.

---

## Conclusion

This project demonstrates an end-to-end approach to predicting stock price movements using machine learning. Our analysis shows that the logistic regression model currently offers the best balance between accuracy and generalization. Future work will focus on enhancing feature engineering, exploring more advanced models, and quantifying the financial impact through backtesting. 
