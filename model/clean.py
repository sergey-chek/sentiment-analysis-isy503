import pandas as pd
import numpy as np
import re


# Output folder path
output_folder = 'model/export-data'
# Load the CSV file from the local path
data = pd.read_csv(f'{output_folder}/cleaned_reviews.csv', sep=',', encoding='utf8')

# Removing Duplicate Reviews
# Duplicate reviews might be spam or repeated entries that should be removed
data.drop_duplicates(subset = ['review_text_correct'], inplace = True)
print('Number of reviews after removing duplicates: ', str(len(data)))

# ------------------------------------------------------------------------------------------------
# Outlier removal to eliminate really short or wrong reviews using Median Absolute Deviation (MAD)
# The Median Absolute Deviation (MAD) is a robust method for detecting outliers in data
# 1. Calculate the Length of Each Reviews
# 2. Calculate the Median and MAD
# 3. Identify Outliers (a threshold (usually 2.5 or 3 times the MAD) to identify outliers)
# 4. Filter Out Outliers

data['review_length'] = data['review_text_correct'].str.split().apply(len)
# Calculate the median of review lengths
median_review_length = data['review_length'].median() 
# Calculate the Median Absolute Deviation (MAD)
mad = np.abs(data['review_length'] - median_review_length).median()
# Define a threshold for identifying outliers
threshold = 2.5  # can adjust this threshold based on needs
# Define the upper and lower fences for outlier detection
upper_fence_mad = median_review_length + threshold * mad 
lower_fence_mad = median_review_length - threshold * mad 
# Identify outliers based on the MAD method
outliers_mad = data[(data['review_length'] > upper_fence_mad) | (data['review_length'] < lower_fence_mad)]
# Remove outliers
data = data[~data.index.isin(outliers_mad.index)]
print("-----------------------------------------------------")
print("Outlier removal using Median Absolute Deviation (MAD)")
print("Number of rows after MAD removal:", len(data))
print("Number of rows removed:", len(outliers_mad))
print("-----------------------------------------------------")


# ------------------------------------------------------------------------------------------------
# Additional outlier removal using IQR (Interquartile Range)
'''
1. Calculate the First Quartile (q1)
2. Calculate the Third Quartile (q3)
3. Calculate the Interquartile Range (IQR)
4. Determine the Lower and Upper Boundaries
5. Remove Outliers
'''
# Calculate the first quartile (Q1) and third quartile (Q3)
q1 = data['review_length'].quantile(0.25) 
q3 = data['review_length'].quantile(0.75) 
# Calculate the IQR (Interquartile Range)
iqr = q3 - q1
# Define the upper and lower fences for outlier detection
upper_bound = q3 + 1.5 * iqr 
lower_bound = q1 - 1.5 * iqr 
# Identify outliers based on the IQR method
outliers_iqr = data[(data['review_length'] > upper_bound) | (data['review_length'] < lower_bound)]
# Remove outliers
data = data[~data.index.isin(outliers_iqr.index)]
print("-----------------------------------------------------")
print("Outlier removal using IQR (Interquartile Range)")
print("Number of rows after IQR removal:", len(data))
print("Number of rows removed:", len(outliers_iqr))
print("-----------------------------------------------------")


# ------------------------------------------------------------------------------------------------
#Cleaning the data from punctuation (Need to test it, because it can increase or reduce the overall accuracy of the future model)

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

# Clean 'review_text_correct' column
data['review_text_correct'] = data['review_text_correct'].apply(remove_punctuation)


# ------------------------------------------------------------------------------------------------

#Uses the NLTK library to perform sentiment analysis on text data, identifying and filtering out reviews with
#nonsensical or spam-like reviews

#If the threshold (compound) is too low (e.g., -0.9): Only a very small number of extremely negative reviews will be detected, 
# potentially missing less negative but still significant reviews.
#If the threshold is too high (e.g., 0): More reviews will be detected, including those that may be only slightly negative, 
# increasing the number of false positives.

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

# Instantiate the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Define a function to detect anomalies based on sentiment analysis
def detect_anomalies(text):
    # Return True if the sentiment is significantly negative, else False
    return sia.polarity_scores(text)['compound'] < -0.5

# Filter out reviews identified as anomalies
filtered_data_nlp = data[~data['review_text_correct'].apply(detect_anomalies)]

# Evaluate the effectiveness of NLP-based anomaly detection
num_removed_reviews = len(data) - len(filtered_data_nlp)
print("NLTK NLP-based Anomaly Detection")
print(f"Number of reviews before filtering: {len(data)}")
print(f"Number of reviews after filtering: {len(filtered_data_nlp)}")



# ------------------------------------------------------------------------------------------------
# Padding/Truncating the remaining data
max_review_length = 100  # Define the maximum review length
data['padded_review_text'] = data['review_text_correct'].apply(
    lambda x: ' '.join(x.split()[:max_review_length]).ljust(max_review_length)
)
print(f"Data padded/truncated to a max review length of {max_review_length} words.")

# Saving the cleaned and padded data
cleaned_file_path = f'{output_folder}/cleaned_and_padded_reviews.csv'
data.to_csv(cleaned_file_path, index=False, encoding='utf-8')
print(f"Cleaned and padded data saved to {cleaned_file_path}")

# Final logging of dataset statistics
final_max_review_length = data['padded_review_text'].apply(lambda x: len(x.split())).max()
final_avg_review_length = data['padded_review_text'].apply(lambda x: len(x.split())).mean()
print(f"Final maximum review length: {final_max_review_length} words")
print(f"Final average review length: {final_avg_review_length:.2f} words")