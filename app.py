import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title("House Price Prediction App")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    st.subheader("Data Preview")
    st.write(data.head())

    x = data[['SquareFootage']]
    y = data['Price']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(x_train, y_train)

    st.subheader("Training Data Sample")
    st.write(pd.concat([x_train, y_train], axis=1).head())

    st.subheader("Testing Data Sample")
    st.write(pd.concat([x_test, y_test], axis=1).head())

    y_pred = model.predict(x_test)

    st.subheader("Regression Plot")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x_test, y_test, color='green', label='Actual Prices')
    ax.plot(x_test, y_pred, color='red', label='Regression Line')
    ax.set_xlabel('Square Footage')
    ax.set_ylabel('Price')
    ax.set_title('Simple Linear Regression: Actual Vs Predicted Price')
    ax.legend()
    st.pyplot(fig)

    sqft_input = st.number_input("Enter Square Footage for Price Prediction", min_value=100, max_value=10000, value=2000)
    new_house_sqft = pd.DataFrame({'SquareFootage': [sqft_input]})
    predict_price = model.predict(new_house_sqft)

    st.subheader("Predicted Price")
    st.write(f"For a house with {sqft_input} sqft, the predicted price is: ${predict_price[0]:,.2f}")
else:
    st.info("Please upload an Excel file to get started.")