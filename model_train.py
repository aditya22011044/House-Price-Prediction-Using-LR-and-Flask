import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle


df = pd.read_csv('house_data.csv')

features = ['bedrooms', 'bathrooms', 'floors', 'yr_built', 'sqft_living']
target = 'price'

X = df[features]
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

pickle.dump(lr, open('model.pkl', 'wb'))

print("✅ Model trained and saved successfully!")
