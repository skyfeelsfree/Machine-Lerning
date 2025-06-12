
---

````markdown
# 💇‍♀️ Hair Fall Level Prediction using Machine Learning & Deep Learning

이 프로젝트는 다양한 기계학습과 딥러닝 모델을 사용하여 **머리카락 빠짐(hair fall)** 정도를 예측하는 예제입니다. Google Drive에서 데이터를 불러오고, 전처리, 모델 학습, 시각화를 모두 포함한 **교육용 종합 실습** 코드입니다.

## ✅ 사용한 기술
- 데이터 전처리: pandas, sklearn
- 시각화: matplotlib
- 모델: 다중 선형 회귀, 랜덤 포레스트, 딥러닝(Keras)
- 평가 지표: MSE, MAE, R² Score

---

## 1. 📁 구글 마운트 및 데이터 수집

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 데이터 불러오기
url = "https://drive.google.com/uc?id=1UriMxGMrB960jzfMDRpSxGA20cL0Vz7E"
df = pd.read_csv(url)

# 데이터 확인
print(df.info())
print(df.head())
````

---

## 2. 🧼 전처리

```python
# 결측값 평균으로 채우기
df.fillna(df.mean(), inplace=True)

# 범주형 변수 처리 (One-Hot Encoding)
encoder = OneHotEncoder(sparse_output=False)
hair_texture_encoded = encoder.fit_transform(df[['hair_texture']])
df = df.drop(columns=['hair_texture'])
df = pd.concat([df, pd.DataFrame(hair_texture_encoded)], axis=1)

# 컬럼 이름 문자열로 변환
df.columns = df.columns.astype(str)

# 데이터 정규화
scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df)

# 입력/출력 분리
X = df.drop(columns=['hair_fall'])
Y = df['hair_fall']
```

---

## 3. 🤖 기계학습 모델 적용하기

```python
# 입력/출력 분리
X = df.drop(columns=['hair_fall'])
Y = df['hair_fall']

# 학습/테스트 분할
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 다중 선형 회귀
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)
Y_pred = lin_reg.predict(X_test)
print(f"Linear Regression MSE: {mean_squared_error(Y_test, Y_pred):.4f}")
print(f"Linear Regression MAE: {mean_absolute_error(Y_test, Y_pred):.4f}")
print(f"Linear Regression R² Score: {r2_score(Y_test, Y_pred):.4f}")

# 랜덤 포레스트
rf_reg = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
rf_reg.fit(X_train, Y_train)
Y_pred_rf = rf_reg.predict(X_test)
print(f"Random Forest MSE: {mean_squared_error(Y_test, Y_pred_rf):.4f}")
print(f"Random Forest MAE: {mean_absolute_error(Y_test, Y_pred_rf):.4f}")
print(f"Random Forest R² Score: {r2_score(Y_test, Y_pred_rf):.4f}")
```

---

## 4. 🧠 딥러닝 모델 추가

```python
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

# 시드 고정
tf.random.set_seed(3)

# 모델 구성
model = Sequential()
model.add(Dense(28, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

# 컴파일
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

# 학습
history = model.fit(X_train, Y_train, epochs=15, batch_size=10, validation_split=0.25)

# 시각화
plt.plot(history.history['loss'], 'b-', label='Train Loss')
plt.plot(history.history['val_loss'], 'r--', label='Validation Loss')
plt.xlabel('Epoch')
plt.legend()
plt.title('Loss Progress')
plt.show()

# 평가
loss, mse = model.evaluate(X_test, Y_test)
print(f"Deep Learning Model - MSE: {mse:.4f}")
```

---

## 5. 📊 시각화 및 모델 비교

```python
# 예측 결과 비교
plt.figure(figsize=(10, 5))
plt.scatter(Y_test, Y_pred, label='Linear Regression', alpha=0.5, color='blue')
plt.scatter(Y_test, Y_pred_rf, label='Random Forest', alpha=0.5, color='green')
plt.scatter(Y_test, model.predict(X_test), label='Deep Learning', alpha=0.5, color='red')
plt.xlabel("Actual Hair Fall Level")
plt.ylabel("Predicted Hair Fall Level")
plt.legend()
plt.title("Comparison of Predictions from Different Models")
plt.show()

# 오차 분포
residuals_lin = Y_test - Y_pred
residuals_rf = Y_test - Y_pred_rf
residuals_dl = Y_test - model.predict(X_test).flatten()

plt.figure(figsize=(10, 5))
plt.hist(residuals_lin, bins=30, alpha=0.5, label='Linear Regression', color='blue')
plt.hist(residuals_rf, bins=30, alpha=0.5, label='Random Forest', color='green')
plt.hist(residuals_dl, bins=30, alpha=0.5, label='Deep Learning', color='red')
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Residual Distribution of Different Models")
plt.legend()
plt.show()

# 성능 비교 지표 시각화
metrics = {
    "MSE": [mean_squared_error(Y_test, Y_pred), mean_squared_error(Y_test, Y_pred_rf), mse],
    "MAE": [mean_absolute_error(Y_test, Y_pred), mean_absolute_error(Y_test, Y_pred_rf), mean_absolute_error(Y_test, model.predict(X_test).flatten())],
    "R² Score": [r2_score(Y_test, Y_pred), r2_score(Y_test, Y_pred_rf), None]
}

df_metrics = pd.DataFrame(metrics, index=['Linear Regression', 'Random Forest', 'Deep Learning'])
df_metrics.plot(kind='bar', figsize=(10, 5))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.show()
```

---

## 📌 참고 사항

* 이 코드는 교육용 예제로, 실제 데이터 분석 프로젝트에서는 과적합 방지, 더 정교한 전처리, 하이퍼파라미터 튜닝 등의 과정이 추가로 필요합니다.
* 데이터 출처는 Google Drive 링크로 제공됩니다.
* 딥러닝의 R² Score는 기본 제공되지 않으므로, 필요하면 사용자 정의 함수로 계산할 수 있습니다.

---

## ✨ 목적

이 프로젝트는 **기초적인 머신러닝 및 딥러닝 실습을 통해 전처리부터 예측, 시각화까지 전 과정을 경험**할 수 있도록 설계되었습니다. 다양한 모델을 비교하고 학습 성능을 이해하는 데 도움이 됩니다.

```

--- 
