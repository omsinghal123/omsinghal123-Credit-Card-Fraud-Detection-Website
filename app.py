import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load data
data = pd.read_csv('creditcard.csv')

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Select features and target
X = data.drop(columns="Class", axis=1)
y = data["Class"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# Streamlit app
st.title("Credit Card Fraud Detection")

# Display model accuracy
st.write(f"**Accuracy on Training data:** {train_acc:.2f}")
st.write(f"**Accuracy on Test data:** {test_acc:.2f}")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file with transaction features", type="csv")

if uploaded_file is not None:
    # Read the uploaded file, assuming it may have a header
    input_data = pd.read_csv(uploaded_file)
    
    # Ensure the input data does not have the 'Class' column and matches the feature columns
    input_data = input_data.drop(columns=['Class'], errors='ignore')
    
    # Check if the input data has the correct columns
    expected_columns = X.columns.tolist()
    if all(col in input_data.columns for col in expected_columns):
        input_data = input_data[expected_columns]  # Reorder columns to match training data
        predictions = model.predict(input_data)
        
        # Add predictions to the DataFrame
        input_data['Prediction'] = predictions
        input_data['Result'] = input_data['Prediction'].apply(lambda x: 'Legitimate' if x == 0 else 'Fraudulent')
        
        # Display the DataFrame with predictions
        st.write("### Predictions")
        
        # Limit the number of rows displayed
        max_rows = 1000
        if len(input_data) > max_rows:
            st.warning(f"Displaying first {max_rows} rows only. Please download the full results for more details.")
            input_data = input_data.head(max_rows)
        
        # Highlight fraudulent transactions
        def highlight_fraud(s):
            return ['background-color: #ffcccc' if v == 'Fraudulent' else '' for v in s]

        st.dataframe(input_data.style.apply(highlight_fraud, subset=['Result']))
    else:
        st.write(f"Expected columns: {expected_columns}")
        st.write(f"Uploaded columns: {input_data.columns.tolist()}")
        st.write("Please ensure the uploaded file has the correct columns.")
else:
    st.info("Please upload a CSV file to see predictions.")
