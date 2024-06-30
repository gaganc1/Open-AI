import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from transformers import pipeline

# Load pre-trained sentiment analysis pipeline
sentiment_analyzer = pipeline('sentiment-analysis')

# Define function to extract messages from conversations
def extract_messages(conversations):
    messages = []
    for conversation in conversations:
        for turn in conversation:
            messages.append(turn['value'])
    return messages

# Define function to cluster conversations into topics
def cluster_conversations(messages, n_clusters=10):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(messages)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    return clusters

# Define function to analyze sentiment
def analyze_sentiment(messages):
    sentiments = []
    for message in messages:
        result = sentiment_analyzer(message[:512])  # Truncate to 512 characters for analysis
        sentiments.append(result[0]['label'])
    return sentiments

# Load data from JSON file
uploaded_file = st.file_uploader("Choose a JSON file", type="json")
if uploaded_file is not None:
    conversations = pd.read_json(uploaded_file)

    # Extract messages and cluster them into topics
    messages = extract_messages(conversations)
    clusters = cluster_conversations(messages)
    
    # Analyze sentiment for each message
    sentiments = analyze_sentiment(messages)

    # Create DataFrame to display results
    data = {'Conversation No': range(1, len(messages) + 1), 'Message': messages, 'Topic': clusters, 'Sentiment': sentiments}
    df = pd.DataFrame(data)

    # Display Counts
    st.title("Counts")
    st.subheader("Topic Counts")
    topic_counts = df['Topic'].value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Count']
    st.table(topic_counts)

    st.subheader("Sentiment Counts")
    sentiment_counts = df['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    st.table(sentiment_counts)

    # Display Sessions
    st.title("Sessions")
    st.dataframe(df[['Conversation No', 'Topic', 'Sentiment']])
