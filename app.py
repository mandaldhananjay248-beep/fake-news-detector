import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import time

# Set page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Title and description
st.title("📰 Fake News Detection System")
st.markdown("""
This application uses machine learning to classify news articles as **Real** or **Fake**.
Simply paste a news article in the text box below and click the 'Analyze' button.
""")

# Preprocessing function
def wordopt(text):
    # Convert into lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove newline characters
    text = re.sub(r'\n', '', text)
    return text

# Load and preprocess data (cached to improve performance)
@st.cache_data
def load_data():
    true = pd.read_csv('True_small.csv')
    fake = pd.read_csv('Fake_small.csv')
    
    true['label'] = 1
    fake['label'] = 0
    
    news = pd.concat([fake, true], axis=0)
    news = news.drop(['title', 'subject', 'date'], axis=1)
    news = news.sample(frac=1)
    news.reset_index(inplace=True, drop=True)
    news['text'] = news['text'].apply(wordopt)
    
    return news

# Train models (cached to avoid retraining)
@st.cache_resource
def train_models():
    news = load_data()
    
    x = news['text']
    y = news['label']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    
    vectorization = TfidfVectorizer()
    xv_train = vectorization.fit_transform(x_train)
    xv_test = vectorization.transform(x_test)
    
    # Initialize and train models
    LR = LogisticRegression()
    LR.fit(xv_train, y_train)
    
    DTC = DecisionTreeClassifier()
    DTC.fit(xv_train, y_train)
    
    RFC = RandomForestClassifier()
    RFC.fit(xv_train, y_train)
    
    GBC = GradientBoostingClassifier()
    GBC.fit(xv_train, y_train)
    
    return vectorization, LR, DTC, RFC, GBC, xv_test, y_test, x_test, y_test

# Load models
vectorization, LR, DTC, RFC, GBC, xv_test, y_test, x_test_orig, y_test_orig = train_models()

# Function to output label
def output_label(n):
    if n == 0:
        return "Fake News ❌"
    elif n == 1:
        return "Real News ✅"

# Function for manual testing
def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    
    pred_LR = LR.predict(new_xv_test)
    pred_DTC = DTC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    
    return pred_LR[0], pred_DTC[0], pred_RFC[0], pred_GBC[0]

# Create tabs
tab1, tab2, tab3 = st.tabs(["Detect Fake News", "Model Performance", "About"])

with tab1:
    st.header("News Text Analysis")
    
    # Input text area
    news_text = st.text_area(
        "Paste the news article text here:",
        height=300,
        placeholder="Enter news content to analyze..."
    )
    
    # Analyze button
    if st.button("Analyze Article", type="primary"):
        if news_text.strip() == "":
            st.error("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                # Add a small delay to show the spinner
                time.sleep(1)
                
                # Get predictions
                pred_LR, pred_DTC, pred_RFC, pred_GBC = manual_testing(news_text)
                
                # Calculate consensus
                predictions = [pred_LR, pred_DTC, pred_RFC, pred_GBC]
                consensus = "Real News ✅" if sum(predictions) >= 3 else "Fake News ❌"
                
                # Display results
                st.subheader("Analysis Results")
                
                # Overall consensus with color
                if consensus == "Real News ✅":
                    st.success(f"Overall Consensus: {consensus}")
                else:
                    st.error(f"Overall Consensus: {consensus}")
                
                # Individual model results
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Logistic Regression", output_label(pred_LR))
                with col2:
                    st.metric("Decision Tree", output_label(pred_DTC))
                with col3:
                    st.metric("Random Forest", output_label(pred_RFC))
                with col4:
                    st.metric("Gradient Boosting", output_label(pred_GBC))

with tab2:
    st.header("Model Performance Metrics")
    
    # Calculate accuracy scores
    lr_score = LR.score(xv_test, y_test)
    dtc_score = DTC.score(xv_test, y_test)
    rfc_score = RFC.score(xv_test, y_test)
    gbc_score = GBC.score(xv_test, y_test)
    
    # Display scores
    st.subheader("Accuracy Scores on Test Data")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Logistic Regression", f"{lr_score:.2%}")
    with col2:
        st.metric("Decision Tree", f"{dtc_score:.2%}")
    with col3:
        st.metric("Random Forest", f"{rfc_score:.2%}")
    with col4:
        st.metric("Gradient Boosting", f"{gbc_score:.2%}")
    
    # Add a bar chart for comparison
    chart_data = pd.DataFrame({
        'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting'],
        'Accuracy': [lr_score, dtc_score, rfc_score, gbc_score]
    })
    
    st.bar_chart(chart_data.set_index('Model'))

with tab3:
    st.header("About This Project")
    
    st.markdown("""
    This Fake News Detection System uses machine learning to classify news articles as real or fake.
    
    ### How it works:
    1. The system was trained on a dataset of both real and fake news articles
    2. Text preprocessing includes:
       - Converting to lowercase
       - Removing URLs, HTML tags, and punctuation
       - Eliminating newline characters
    3. Four different machine learning models were trained:
       - Logistic Regression
       - Decision Tree Classifier
       - Random Forest Classifier
       - Gradient Boosting Classifier
    
    ### Dataset:
    The model was trained on a combination of:
    - **True_small.csv**: Legitimate news articles
    - **Fake_small.csv**: Fake news articles
    
    ### Limitations:
    - Accuracy depends on the training data
    - May not perform well on very short texts
    - Context and recent events might affect performance
    """)

# Add a sidebar with additional information
with st.sidebar:
    st.header("Fake News Detector")
    st.markdown("""
    This tool analyzes news content using multiple machine learning models to detect potentially fake news.
    
    ### Instructions:
    1. Paste news text in the input box
    2. Click 'Analyze Article'
    3. View results from all models
    4. Consider the overall consensus
    
    **Note:** This is a tool to assist in identification, not a definitive verdict.
    """)
    
    st.divider()
    
    st.markdown("""
    ### Model Information:
    - **Logistic Regression**: Linear model for classification
    - **Decision Tree**: Non-linear model with interpretable rules
    - **Random Forest**: Ensemble of decision trees
    - **Gradient Boosting**: Sequential building of weak predictors
    """)

# Footer
st.divider()
st.caption("Fake News Detection System | Built with Streamlit and Scikit-learn")