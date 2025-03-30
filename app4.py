#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify, render_template
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load synthetic customer data
try:
    customers = pd.read_csv("synthetic_customers.csv")
except FileNotFoundError:
    print("Error: synthetic_customers.csv not found! Please check the file path.")
    exit()

# Load Generative AI model
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Banking products
BANKING_PRODUCTS = {
    "savings account": "Best for customers who want to save money securely and earn interest.",
    "credit card": "Ideal for customers with high monthly spending and a good credit score.",
    "personal loan": "Useful for customers with moderate income and unexpected financial needs.",
    "home loan": "Recommended for high-income customers looking to invest in property.",
    "investment plan": "Great for customers with high spending in the 'Investment' category.",
    "travel card": "Perfect for frequent travelers with high spending in the 'Travel' category."
}

app = Flask(__name__)

def get_customer_info(user_input):
    """Fetch customer details based on ID or name"""
    match = customers[(customers["customer_id"].str.lower() == user_input) | 
                      (customers["name"].str.lower() == user_input)]
    return match.iloc[0].to_dict() if not match.empty else None

def recommend_product(customer):
    """Recommend a banking product based on customer data"""
    if customer["spending_category"] == "Luxury" and customer["monthly_spending"] > 10000:
        return "Credit Card"
    elif customer["income"] > 50000:
        return "Investment Plan"
    elif customer["spending_category"] == "Travel":
        return "Travel Card"
    elif customer["income"] < 30000:
        return "Savings Account"
    else:
        return "Personal Loan"

@app.route("/", methods=["GET"])
def home():
    return "Welcome to U Bank Chatbot! Use the chatbot endpoint to interact."

@app.route("/chatbot", methods=["POST"])
def chatbot():
    user_input = request.form.get("user_input", "").strip().lower()
    if not user_input:
        return jsonify({"response": "Please enter a valid Name or Customer ID."})

    customer = get_customer_info(user_input)

    if not customer:
        return jsonify({"response": "Invalid Name/ID. Please try again or visit a branch for assistance."})

    response = f"**Welcome, {customer['name']}!**\n"
    response += f"- Age: {customer['age']} years\n"
    response += f"- Income: ${customer['income']}\n"
    response += f"- Spending Category: {customer['spending_category']}\n"
    response += f"- Monthly Spending: ${customer['monthly_spending']}\n\n"

    response += "**Our Banking Products:**\n"
    for product, description in BANKING_PRODUCTS.items():
        response += f"- **{product.capitalize()}**: {description}\n"

    recommended_product = recommend_product(customer)
    response += f"\n**Recommended Product:** {recommended_product}\n"

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(port=5050)  # No debug mode to avoid issues

