import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)


def explain_prediction(probability, input_dict, surname):
    prompt = f"""
  You are an expert data scientist at a bank, where you specialize in interpreting and explaining predictions of machine learning models.
  Your machine learning model has predicted that a customer named {surname} will churn with a probability of {round(probability *100, 1)}% based on the information provided below.
  Here is the information about the customer:
  {input_dict}
  
  Here are the machine learning model's top 10 features that are most important and should be used to make the churn prediction:
  +---------------------+-------------+
  |      Feature        | Importance  |
  +---------------------+-------------+
  | NumOfProducts       |   0.323888  |
  | IsActiveMember      |   0.164146  |
  | Age                 |   0.109550  |
  | Geography_Germany   |   0.091373  |
  | Balance             |   0.052786  |
  | Geography_France    |   0.046463  |
  | Gender_Female       |   0.045283  |
  | Geography_Spain     |   0.036855  |
  | CreditScore         |   0.035005  |
  | EstimatedSalary     |   0.032655  |
  | HasCrCard           |   0.031940  |
  | Tenure              |   0.030054  |
  | Gender_Male         |   0.000000  |
  +---------------------+-------------+

  {pd.set_option('display.max_colwidth', None)}
  
  Here are the summary statistics for the churned customers:
  {df[df["Exited"] == 1].describe()}
  
  Here are the summary statistics for the non-churned customers:
  {df[df["Exited"] == 0].describe()}
  !IMPORTANT!
  
  If a customer has over a 40% risk of churning, generate a 3 sentence explanation of why they are at risk to churn.
  If a customer has less than a 40% risk of churning, generate a 3 sentence explanation of why they might not be at risk to churn.
  Your explanation should be based on the customer's information and the summary statistics for the churned and non-churned customers, and the feature importances provided.
  Don't mention the probability of churning, or the machine learning model, or say anything like 'Based on the machine learning model's prediction and top 10 most important features...', just explain the prediction.
  Don't mention or repeat the process you used to make the prediction, just explain the prediction.
  Don't use let's or other interactive language, just explain the prediction.
  Make 3 sentences have a 1 line break between them. Don't use any other line breaks or no line breaks at all.
  Convert any features names to their actual names, not the encoded ones, in plain text English.
  Example 1:
  Based on the provided customer information and summary statistics, customer Brownless has a credit score of 581, age of 34, tenure of 1, balance of 101633, number of products of 1, has a credit card of 1, is an inactive member of 0, estimated salary of 110431.51, and belongs to the geography of Germany.

Given that Brownless has a high importance score for Geography_Germany, a significant feature associated with 91.37% of churned customers compared to 45.06% of non-churned customers, this may indicate that geographical location plays a critical role in churning decisions. However, given that their other characteristics are relatively comparable to those of the non-churned customers, it can be noted that while geography may be an influential factor, other aspects of their profile may support the decision to retain them as a customer.

Assuming that geographical location plays a significant role, but considering the customer's profile is relatively in line with others, it can be argued that the decision to churn or retain could be influenced by a variety of factors such as communication with the customer or financial history. However, at this point, there is not enough information available to make a definitive judgment, suggesting the customer may not be at a high level of risk to churn.
  """

    print("EXPLANATION PROMPT: ", prompt)

    raw_response = client.chat.completions.create(
        model="llama-3.2-3b-preview", messages=[{"role": "user", "content": prompt}]
    )

    return raw_response.choices[0].message.content


def generate_email(probability, input_dict, explanation, surname):
    prompt = f"""
    You are an expert email writer at a bank, where you specialize in writing emails to customers for ensuring that the customers stay with the bank and are incetivized to stay with various offers.
    You noticed that customer {surname} has a {round(probability * 100, 1)}% chance of churning.
    Here is the information about the customer:
    {input_dict}
    Here is the explanation for the prediction:
    {explanation}
    Generate an email to the customer based on the above information and explanation about them to ask them to stay with the bank if they are at risk of churning, or offer them more incentives so they that remain loyal customers to the bank.
    Make sure to list out a set of incentives to stay based on their information, in bullet points format. Don't ever mention the probability of churning, or the machine learning model, or say anything like 'Based on the machine learning model's prediction and top 10 most important features...'.

    """

    raw_response = client.chat.completions.create(
        model="llama-3.2-3b-preview", messages=[{"role": "user", "content": prompt}]
    )
    print("\n\nEMAIL PROMPT: ", prompt)

    return raw_response.choices[0].message.content


def load_model(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


dt_model = load_model("dt_model.pkl")
knn_model = load_model("knn_model.pkl")
nb_model = load_model("nb_model.pkl")
rf_model = load_model("rf_model.pkl")
svm_model = load_model("svm_model.pkl")
voting_clf = load_model("voting_clf.pkl")
xgb_model = load_model("xgb_model.pkl")
xgb_fe_model = load_model("xgboost_feature_engineered.pkl")
xgb_SMOTE = load_model("xgboost_SMOTE.pkl")


def prepare_input(
    credit_score,
    location,
    gender,
    age,
    tenure,
    balance,
    num_products,
    has_credit_card,
    is_active_member,
    estimated_salary,
):
    input_dict = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": int(has_credit_card),
        "IsActiveMember": int(is_active_member),
        "EstimatedSalary": estimated_salary,
        "Geography_France": 1 if location == "France" else 0,
        "Geography_Germany": 1 if location == "Germany" else 0,
        "Geography_Spain": 1 if location == "Spain" else 0,
        "Gender_Female": 1 if gender == "Female" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
    }

    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict


def prepare_input_2(
    credit_score,
    location,
    gender,
    age,
    tenure,
    balance,
    num_products,
    has_credit_card,
    is_active_member,
    estimated_salary,
):
    input_dict_2 = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": int(has_credit_card),
        "IsActiveMember": int(is_active_member),
        "EstimatedSalary": estimated_salary,
        "Geography_France": 1 if location == "France" else 0,
        "Geography_Germany": 1 if location == "Germany" else 0,
        "Geography_Spain": 1 if location == "Spain" else 0,
        "Gender_Female": 1 if gender == "Female" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
        "CLV": balance * estimated_salary / 100000,
        "TenureAgeRatio": tenure / age,
        "AgeGroup_MiddleAged": 1 if 30 <= age < 50 else 0,
        "AgeGroup_Senior": 1 if 50 <= age < 70 else 0,
        "AgeGroup_Elderly": 1 if age >= 70 else 0,
    }

    input_df_2 = pd.DataFrame([input_dict_2])
    return input_df_2, input_dict_2


def make_predictions(input_df, input_dict):
    probabilities = {
        "XGBoost": xgb_model.predict_proba(input_df)[0][1],
        "Random Forest": rf_model.predict_proba(input_df)[0][1],
        "K-Nearest Neighbor": knn_model.predict_proba(input_df)[0][1],
        "Support Vector Machine": svm_model.predict_proba(input_df)[0][1],
        "Decision Tree": dt_model.predict_proba(input_df)[0][1],
        "Naive Bayes": nb_model.predict_proba(input_df)[0][1],
    }

    avg_prob = np.mean(list(probabilities.values()))

    st.markdown("### Model Probabilities")
    for model, prob in probabilities.items():
        st.write(f"{model} {prob:.3f}")
    st.write(f"Average Probability: {avg_prob:.3f}")

    return avg_prob


def make_predictions_2(input_df, input_dict):

    voting_prediction = int(voting_clf.predict(input_df)[0])

    probabilities = {
        "XGBoost_Feature_Engineered": xgb_fe_model.predict_proba(input_df)[0][1],
        "XGBoost_SMOTE": xgb_SMOTE.predict_proba(input_df)[0][1],
    }

    avg_prob = np.mean(list(probabilities.values()))

    st.write(f"_Voting Classifier Prediction (0 or 1): {voting_prediction}_")

    st.markdown("### Advanced Model Probabilities")
    for model, prob in probabilities.items():
        st.write(f"{model} Probability: {prob:.3f}")

    st.write(f"Average Probability (XGBoost models): {avg_prob:.3f}")
    return avg_prob


st.title("Customer Churn Prediction")

df = pd.read_csv("churn.csv")

customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:

    selected_customer_id = int(selected_customer_option.split(" - ")[0])
    print("Selected Customer ID:", selected_customer_id)

    selected_surname = selected_customer_option.split(" - ")[1]
    print("Surname", selected_surname)

    selected_customer = df.loc[df["CustomerId"] == selected_customer_id].iloc[0]
    print("Selected Customer", selected_customer)

    col1, col2 = st.columns(2)

    with col1:

        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=int(selected_customer["CreditScore"]),
        )

        location = st.selectbox(
            "Location",
            ["Spain", "France", "Germany"],
            index=["Spain", "France", "Germany"].index(selected_customer["Geography"]),
        )

        gender = st.radio(
            "Gender",
            ["Male", "Female"],
            index=0 if selected_customer["Gender"] == "Male" else 1,
        )

        age = st.number_input(
            "Age", min_value=18, max_value=100, value=int(selected_customer["Age"])
        )

        tenure = st.number_input(
            "Tenure (years)",
            min_value=0,
            max_value=50,
            value=int(selected_customer["Tenure"]),
        )

    with col2:

        balance = st.number_input(
            "Balance", min_value=0, value=int(selected_customer["Balance"])
        )

        num_products = st.number_input(
            "Number of Products",
            min_value=1,
            max_value=10,
            value=int(selected_customer["NumOfProducts"]),
        )

        has_credit_card = st.checkbox(
            "Has Credit Card", value=bool(selected_customer["HasCrCard"])
        )

        is_active_member = st.checkbox(
            "Is Active Member", value=bool(selected_customer["IsActiveMember"])
        )

        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=float(selected_customer["EstimatedSalary"]),
        )

    input_df, input_dict = prepare_input(
        credit_score,
        location,
        gender,
        age,
        tenure,
        balance,
        num_products,
        has_credit_card,
        is_active_member,
        estimated_salary,
    )

    input_df_2, input_dict_2 = prepare_input_2(
        credit_score,
        location,
        gender,
        age,
        tenure,
        balance,
        num_products,
        has_credit_card,
        is_active_member,
        estimated_salary,
    )

    avg_prob = make_predictions(input_df, input_dict)
    avg_prob_2 = make_predictions_2(input_df_2, input_dict_2)

    explanation = explain_prediction(avg_prob, input_dict, selected_surname)

    st.write("### Explanation")
    st.write(explanation)
