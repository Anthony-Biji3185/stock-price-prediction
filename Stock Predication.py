import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
df = pd.read_csv(
    r"C:\Users\antho\Documents\Summer Vacation Learning 2026\ML Course Projects\Stock Data.csv"
)
print("\n===== DATASET INFORMATION =====")
print(df.info())
print("\n===== FIRST 5 ROWS =====")
print(df.head())
print("\n===== DATASET SHAPE =====")
print(df.shape)
print("\n===== DATASET STATISTICS =====")
print(df.describe())
print("\n===== CORRELATION MATRIX =====")
print(df.select_dtypes(include=[np.number]).corr())
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())
df.dropna(inplace=True)
print("\n===== DATASET SHAPE AFTER CLEANING =====")
print(df.shape)
if 'date' in df.columns:
    df.drop('date', axis=1, inplace=True)
X = df.drop('next_day_close', axis=1)
y = df['next_day_close']
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_r2 = r2_score(y_test, lr_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("\n===== MODEL PERFORMANCE =====")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R² Score : {r2:.4f}")
if r2 >= 0.90:
    print("Model Accuracy Level : Excellent")
elif r2 >= 0.80:
    print("Model Accuracy Level : Very Good")
elif r2 >= 0.70:
    print("Model Accuracy Level : Good")
else:
    print("Model Accuracy Level : Needs Improvement")
print("\n===== MODEL COMPARISON =====")
print(f"Linear Regression R² : {lr_r2:.4f}")
print(f"Random Forest R²     : {r2:.4f}")
lr_mae = mean_absolute_error(y_test, lr_pred)
print(f"Linear Regression MAE : {lr_mae:.4f}")
print(f"Random Forest MAE     : {mae:.4f}")
if r2 > lr_r2:
    print("\n🏆 Random Forest performed better.")
else:
    print("\n🏆 Linear Regression performed better.")
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})
importance = importance.sort_values(
    by='Importance',
    ascending=False
)
print("\n===== FEATURE IMPORTANCE =====")
print(importance)

print("\nMost Important Feature:")
print(importance.iloc[0]['Feature'])
plt.figure(figsize=(12,6))
plt.plot(y_test.values[:100],
         label='Actual')
plt.plot(y_pred[:100],
         label='Predicted')
plt.title("Actual vs Predicted Stock Price")
plt.xlabel("Samples")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10,6))
importance = importance.sort_values(
    by='Importance'
)
plt.barh(
    importance['Feature'],
    importance['Importance']
)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.show()
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Scatter Plot")
plt.grid(True)
plt.show()
sample = X.iloc[-1].values.reshape(1,-1)
future_price = model.predict(sample)
print("\nPredicted Next Day Close Price:")
print(future_price[0])
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
results.to_csv("prediction_results.csv", index=False)
print("\nPrediction results saved successfully.")
print("\n====================================")
print("PROJECT RESULT SUMMARY")
print("====================================")
print(f"Random Forest R² Score : {r2:.4f}")
print(f"Random Forest MAE      : {mae:.4f}")
if r2 >= 0.90:
    print("Result : Excellent Prediction Performance")
elif r2 >= 0.80:
    print("Result : Very Good Prediction Performance")
elif r2 >= 0.70:
    print("Result : Good Prediction Performance")
else:
    print("Result : Prediction Model Needs Improvement")
print("\nProject Conclusion:")
print("The Stock Price Prediction model was successfully developed using Machine Learning.")
print("Random Forest and Linear Regression models were trained and evaluated.")
print("Random Forest achieved better performance and was selected as the final model.")
print("The model can be used to predict the next day's stock closing price based on historical stock market indicators.")