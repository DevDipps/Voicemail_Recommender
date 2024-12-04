import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Function to preprocess and compute similarity
# Function to preprocess and compute similarity
def compute_recommendations(data, similarity_threshold=0.95):
    # Updated columns of interest
    columns_of_interest = [
        'total day minutes', 'total day calls', 'total day charge',
        'total eve minutes', 'total eve calls', 'total eve charge',
        'total night minutes', 'total night calls', 'total night charge',
        'total intl minutes', 'total intl calls', 'total intl charge',
        'customer service calls'
    ]
    
    # Split data into customers with and without a voice mail plan
    with_voicemail = data[data['voice mail plan'] == 'yes'][columns_of_interest]
    without_voicemail = data[data['voice mail plan'] == 'no'][columns_of_interest]
    
    # Normalize data for fair comparison
    scaler = MinMaxScaler()
    with_voicemail_normalized = scaler.fit_transform(with_voicemail)
    without_voicemail_normalized = scaler.transform(without_voicemail)
    
    # Compute cosine similarity between customers without and with a voice mail plan
    similarity_matrix = cosine_similarity(without_voicemail_normalized, with_voicemail_normalized)
    
    # Find the maximum similarity score for each customer without a voicemail plan
    max_similarity_scores = similarity_matrix.max(axis=1)
    most_similar_indices = similarity_matrix.argmax(axis=1)
    
    # Add results back to the customers without voicemail
    without_voicemail_results = data[data['voice mail plan'] == 'no'].copy()
    without_voicemail_results['max_similarity'] = max_similarity_scores
    without_voicemail_results['most_similar_customer'] = most_similar_indices
    
    # Filter results based on the similarity threshold
    filtered_results = without_voicemail_results[without_voicemail_results['max_similarity'] >= similarity_threshold]
    
    # Sort by max similarity in descending order
    filtered_results = filtered_results.sort_values(by='max_similarity', ascending=False)
    
    return filtered_results[['state', 'phone number', 'voice mail plan', 'max_similarity', 'most_similar_customer']]

# Streamlit UI
def main():
    st.title("Personalized Voicemail Plan Recommendation System")
    st.write("Upload the telecom churn data to generate recommendations for customers without a voicemail plan.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    if uploaded_file is not None:
        # Load the data
        data = pd.ExcelFile(uploaded_file).parse('in')
        
        # Display the raw data
        st.subheader("Dataset Preview")
        st.dataframe(data.head())
        
        # Generate recommendations
        st.subheader("Recommendations (Filtered by Similarity Threshold ≥ 0.95)")
        recommendations = compute_recommendations(data, similarity_threshold=0.95)
        st.dataframe(recommendations)
        
        # Option to download results
        st.download_button(
            label="Download Recommendations",
            data=recommendations.to_csv(index=False),
            file_name="voicemail_recommendations.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()