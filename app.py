import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
data = pd.read_csv(url)

# Define features and target
X = data.drop('charges', axis=1)
y = data['charges']

# Define preprocessing for numerical and categorical features
numerical_features = ['age', 'children']
categorical_features = ['sex', 'smoker', 'region']

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor()
}

# Streamlit UI
st.title("Insurance Charges Prediction")
st.write("Choose a model and predict the insurance charges based on the provided features")

# Model selection
selected_model = st.selectbox("Select Model", list(models.keys()))

# Selected model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', models[selected_model])
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the selected model
pipeline.fit(X_train, y_train)

age = st.slider("Age", 18, 100, 30)
sex = st.radio("Sex", ['male', 'female'])
children = st.slider("Children", 0, 10, 1)
smoker = st.radio("Smoker", ['yes', 'no'])
region_options = ['southwest', 'northeast']
region = st.selectbox("Region", region_options)

input_data = pd.DataFrame([[age, sex, children, smoker, region]],
                          columns=['age', 'sex', 'children', 'smoker', 'region'])

if st.button("Predict"):
    prediction = pipeline.predict(input_data)[0]
    st.write(f"Predicted Charges: ${round(prediction, 2)}")

# Display evaluation metrics
if st.button("Evaluate Model"):
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    st.write(f"Root Mean Squared Error: {rmse}")
    st.write(f"R^2 Score: {r2}")
