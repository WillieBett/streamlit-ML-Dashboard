import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Streamlit Dashboard title
st.title("Train a Machine Learning Model")

# Upload CSV File
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Check if file is uploaded
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.write("Dataset preview:", df.head())

        # Display dataset information
        st.subheader("Dataset Information")
        st.write(f"**Rows:** {df.shape[0]}, **Columns:** {df.shape[1]}")
        st.write(df.describe())

        # Select features and target column
        target_column = st.selectbox("Select target column", df.columns)

        # Select columns for model
        feature_columns = st.multiselect("Select feature columns", df.columns.tolist(), default=df.columns.tolist())

        # Preprocessing (handle missing values)
        st.subheader("Preprocessing")

        # Option to fill missing values with mean/median for numeric columns
        if st.checkbox("Fill missing values"):
            fill_value = st.radio("Select value to fill missing data", ("mean", "median"))
            for col in df.select_dtypes(include=["float64", "int64"]).columns:
                if fill_value == "mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)
            st.write("Missing values filled.")

        # Prepare data for training
        X = df[feature_columns]
        y = df[target_column]

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Feature Scaling (optional, for models like SVM, KNN)
        if st.checkbox("Standardize features"):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Model selection
        model_option = st.selectbox("Select Model", ["Linear Regression", "Random Forest"])

        # Train Model
        if st.button("Train Model"):
            if model_option == "Linear Regression":
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Metrics
                mse = mean_squared_error(y_test, y_pred)
                st.write(f"Mean Squared Error: {mse}")

                # Display predictions vs actual
                st.subheader("Predictions vs Actual")
                comparison = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
                st.write(comparison)

            elif model_option == "Random Forest":
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Accuracy and confusion matrix
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {accuracy}")
                
                cm = confusion_matrix(y_test, y_pred)
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)

        # Option for making predictions
        st.subheader("Make Predictions")
        if st.button("Make Prediction"):
            input_data = []
            for col in feature_columns:
                input_value = st.number_input(f"Enter value for {col}", value=0.0)
                input_data.append(input_value)

            input_data = np.array(input_data).reshape(1, -1)
            if st.checkbox("Standardize input features"):
                input_data = scaler.transform(input_data)

            prediction = model.predict(input_data)
            st.write(f"Predicted value: {prediction[0]}")

    except Exception as e:
        st.error(f"Error: {e}")
