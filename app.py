import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="centered"
)
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, 'model_results')
 
@st.cache_resource
 
def load_model():
 
    linear_model = joblib.load(os.path.join(model_dir, 'linear_regression_model.pkl'))
    decision_model = joblib.load(os.path.join(model_dir, 'decision_tree_model.pkl'))
    random_forest_model = joblib.load(os.path.join(model_dir,'random_forest_model.pkl'))
    standard_scaler_model = joblib.load(os.path.join(model_dir,'standard_scaler.pkl'))
    min_max_scaler_model = joblib.load(os.path.join(model_dir,'min_max_scaler.pkl'))
    encoder = joblib.load(os.path.join(model_dir,'label_encoder.pkl'))
 
    return linear_model,decision_model,random_forest_model,standard_scaler_model,min_max_scaler_model,encoder
 
linear_model,decision_model,random_forest_model,standard_scaler_model,min_max_scaler_model,encoder = load_model()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #666666;
    font-size: 17px;
    margin-bottom: 30px;
}

.salary-label {
    text-align: center;
    font-size: 20px;
    font-weight: 500;
    margin-top: 20px;
}

.salary-value {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-top: 5px;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">💼 Employee Salary Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Multi-Model Salary Prediction & Comparison</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR - ABOUT MODELS
# ============================================================

st.sidebar.header("ℹ️ About Models")

st.sidebar.markdown("""
This application uses three different machine learning models
to predict annual employee salary.
""")

st.sidebar.markdown("### 📈 Linear Regression")

st.sidebar.write(
    "Linear Regression predicts salary by finding a linear relationship "
    "between the employee's characteristics and annual salary. "
    "It works best when the relationship between the input features "
    "and salary is approximately linear."
)

st.sidebar.markdown("### 🌳 Decision Tree")

st.sidebar.write(
    "Decision Tree Regression makes predictions by splitting employees "
    "into different groups based on their feature values. "
    "It can capture non-linear relationships and is easy to understand."
)

st.sidebar.markdown("### 🌲 Random Forest")

st.sidebar.write(
    "Random Forest combines predictions from multiple decision trees "
    "to produce a more robust prediction. It can capture complex "
    "relationships and generally provides more stable predictions "
    "than a single decision tree."
)

st.sidebar.markdown("---")

st.sidebar.info(
    "The three models use the same input features, allowing their "
    "salary predictions to be compared."
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("employee_salary_regression.csv")

    return df


df = load_data()


# ============================================================
# DATA PREPROCESSING
# ============================================================

education_encoder = LabelEncoder()
job_encoder = LabelEncoder()

df["education_level"] = education_encoder.fit_transform(
    df["education_level"]
)

df["job_role"] = job_encoder.fit_transform(
    df["job_role"]
)


# ============================================================
# SCALE ANNUAL SALARY
# ============================================================
# This follows the notebook:
#
# scaler = MinMaxScaler()
# df['annual_salary_usd'] = scaler.fit_transform(
#     df[['annual_salary_usd']]
# )
# ============================================================

salary_scaler = MinMaxScaler()

df["annual_salary_usd"] = salary_scaler.fit_transform(
    df[["annual_salary_usd"]]
)


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df[
    [
        "age",
        "years_experience",
        "education_level",
        "city_tier",
        "performance_score",
        "num_skills",
        "remote_work"
    ]
]

y = df["annual_salary_usd"]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# TRAIN MODELS
# ============================================================

@st.cache_resource
def train_models(X_train, y_train):

    # Linear Regression
    linear_reg_model = LinearRegression()

    model_one = linear_reg_model.fit(
        X_train,
        y_train
    )


    # Decision Tree
    decision_tree_model = DecisionTreeRegressor()

    model_two = decision_tree_model.fit(
        X_train,
        y_train
    )


    # Random Forest
    random_forest_model = RandomForestRegressor()

    model_three = random_forest_model.fit(
        X_train,
        y_train
    )


    return (
        model_one,
        model_two,
        model_three
    )


model_one, model_two, model_three = train_models(
    X_train,
    y_train
)


# ============================================================
# TEST DATA PREDICTIONS
# ============================================================

y_pred_one = model_one.predict(X_test)

y_pred_two = model_two.predict(X_test)

y_pred_three = model_three.predict(X_test)


# ============================================================
# MODEL PERFORMANCE METRICS
# ============================================================

# -------------------------
# MAE
# -------------------------

mae_model_one = mean_absolute_error(
    y_test,
    y_pred_one
)

mae_model_two = mean_absolute_error(
    y_test,
    y_pred_two
)

mae_model_three = mean_absolute_error(
    y_test,
    y_pred_three
)


# -------------------------
# MSE
# -------------------------

mse_model_one = mean_squared_error(
    y_test,
    y_pred_one
)

mse_model_two = mean_squared_error(
    y_test,
    y_pred_two
)

mse_model_three = mean_squared_error(
    y_test,
    y_pred_three
)


# -------------------------
# RMSE
# -------------------------

rmse_model_one = np.sqrt(
    mse_model_one
)

rmse_model_two = np.sqrt(
    mse_model_two
)

rmse_model_three = np.sqrt(
    mse_model_three
)


# -------------------------
# R² SCORE
# -------------------------

r2_score_1 = r2_score(
    y_test,
    y_pred_one
)

r2_score_2 = r2_score(
    y_test,
    y_pred_two
)

r2_score_3 = r2_score(
    y_test,
    y_pred_three
)


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown("### 👤 Enter Employee Details")

col1, col2 = st.columns(2)


with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )

    years_experience = st.number_input(
        "Years of Experience",
        min_value=0,
        max_value=50,
        value=5,
        step=1
    )

    number_of_skills = st.number_input(
        "Number of Skills",
        min_value=0,
        max_value=50,
        value=5,
        step=1
    )


with col2:

    current_salary = st.number_input(
        "Current Salary (USD)",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    education = st.selectbox(
        "Education Level",
        [
            "High school",
            "Bachelors",
            "Masters",
            "PhD"
        ]
    )


# ============================================================
# MODEL SELECTION
# ============================================================

st.markdown("### 🤖 Select Prediction Model")

selected_model = st.radio(
    "Choose one model:",
    [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ],
    horizontal=True
)


# ============================================================
# EDUCATION ENCODING
# ============================================================

education_name_mapping = {
    "High school": "High School",
    "Bachelors": "Bachelor",
    "Masters": "Master",
    "PhD": "PhD"
}

education_dataset_name = education_name_mapping[
    education
]

education_encoded = education_encoder.transform(
    [education_dataset_name]
)[0]


# ============================================================
# DEFAULT VALUES FOR OTHER FEATURES
# ============================================================

default_city_tier = int(
    df["city_tier"].median()
)

default_performance_score = float(
    df["performance_score"].median()
)

default_remote_work = int(
    df["remote_work"].mode()[0]
)


# ============================================================
# PREDICT SALARY BUTTON
# ============================================================

predict_button = st.button(
    "🔮 Predict Salary",
    use_container_width=True
)


# ============================================================
# RESULTS
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # INPUT DATA
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        {
            "age": [age],
            "years_experience": [years_experience],
            "education_level": [education_encoded],
            "city_tier": [default_city_tier],
            "performance_score": [default_performance_score],
            "num_skills": [number_of_skills],
            "remote_work": [default_remote_work]
        }
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if selected_model == "Linear Regression":

        scaled_prediction = model_one.predict(
            input_data
        )[0]

    elif selected_model == "Decision Tree":

        scaled_prediction = model_two.predict(
            input_data
        )[0]

    else:

        scaled_prediction = model_three.predict(
            input_data
        )[0]


    # --------------------------------------------------------
    # CONVERT SCALED PREDICTION BACK TO USD
    # --------------------------------------------------------

    prediction = salary_scaler.inverse_transform(
        [[scaled_prediction]]
    )[0][0]


    # ========================================================
    # PREDICTED SALARY
    # ========================================================

    st.markdown("---")

    st.markdown(
        "<p class='salary-label'>💰 Predicted Annual Salary</p>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<p class='salary-value'>${prediction:,.2f}</p>",
        unsafe_allow_html=True
    )

    st.success(
        f"Prediction generated using the **{selected_model}** model."
    )


    # ========================================================
    # SALARY COMPARISON
    # ========================================================

    difference = prediction - current_salary

    st.markdown("### 📊 Salary Comparison")

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Current Salary",
            f"${current_salary:,.2f}"
        )


    with col2:

        st.metric(
            "Predicted Salary",
            f"${prediction:,.2f}"
        )


    with col3:

        st.metric(
            "Difference",
            f"${difference:,.2f}"
        )


    # ========================================================
    # EMPLOYEE SUMMARY
    # ========================================================

    st.markdown("### 📋 Employee Summary")

    summary = pd.DataFrame(
        {
            "Attribute": [
                "Age",
                "Years of Experience",
                "Education Level",
                "Number of Skills",
                "Current Salary",
                "Selected Model"
            ],

            "Value": [
                age,
                years_experience,
                education,
                number_of_skills,
                f"${current_salary:,.2f}",
                selected_model
            ]
        }
    )

    st.table(summary)


    # ========================================================
    # MODEL PERFORMANCE COMPARISON
    # ========================================================

    st.markdown("---")

    st.markdown(
        "## 📈 Model Performance Comparison"
    )

    st.write(
        "The models are evaluated using the scaled annual salary "
        "values from the test dataset."
    )


    performance_data = pd.DataFrame(
        {
            "Model": [
                "Linear Regression",
                "Decision Tree",
                "Random Forest"
            ],

            "R² Score": [
                r2_score_1,
                r2_score_2,
                r2_score_3
            ],

            "MSE": [
                mse_model_one,
                mse_model_two,
                mse_model_three
            ],

            "RMSE": [
                rmse_model_one,
                rmse_model_two,
                rmse_model_three
            ],

            "MAE": [
                mae_model_one,
                mae_model_two,
                mae_model_three
            ]
        }
    )


    st.dataframe(
        performance_data.style.format(
            {
                "R² Score": "{:.4f}",
                "MSE": "{:.6f}",
                "RMSE": "{:.6f}",
                "MAE": "{:.6f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # VISUALIZATION
    # ========================================================

    st.markdown("---")

    st.markdown(
        "## 📊 Visualization"
    )

    st.write(
        "The following visualizations show the relationship between "
        "employee characteristics and annual salary."
    )


    # ========================================================
    # TWO SCATTER PLOTS AS SUBPLOTS
    # ========================================================

    x_plot = df["age"]

    y_plot = df["annual_salary_usd"]


    fig = plt.figure(
        figsize=(10, 6)
    )


    # --------------------------------------------------------
    # SUBPLOT 1
    # --------------------------------------------------------

    plt.subplot(
        1,
        2,
        1
    )

    plt.title(
        "Age vs Annual Salary"
    )

    plt.xlabel(
        "Age"
    )

    plt.ylabel(
        "Annual Salary (USD)"
    )

    plt.scatter(
        x_plot,
        y_plot,
        marker="*",
        color="green",
        alpha=0.5
    )


    # --------------------------------------------------------
    # SUBPLOT 2
    # --------------------------------------------------------

    plt.subplot(
        1,
        2,
        2
    )

    plt.title(
        "years_experience vs annual_salary_usd"
    )

    plt.xlabel(
        "years_experience"
    )

    plt.ylabel(
        "annual_salary_usd"
    )

    plt.scatter(
        df["years_experience"],
        df["annual_salary_usd"],
        marker="*"
    )


    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)