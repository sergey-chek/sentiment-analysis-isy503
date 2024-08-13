import pandas as pd
import numpy as np

# Output folder path
output_folder = 'model/export-labeled-data'

# Load the CSV file from the local path
data = pd.read_csv(f'{output_folder}/cleaned_reviews.csv', sep=',', encoding='utf8')

# Working with the data....

#Removing Duplicate Reviews:
'''Duplicate reviews might be spam or repeated entries that should be removed.'''

data.drop_duplicates(subset = ['review_text_correct'], inplace = True)
print('Number of reviews: ', str(len(data)))





# Outlier removal to eliminate really short or wrong reviews
'''
The Median Absolute Deviation (MAD) is a robust method for detecting outliers in data
1. Calculate the Length of Each Reviews
2. Calculate the Median and MAD
3. Identify Outliers (a threshold (usually 2.5 or 3 times the MAD) to identify outliers)
4. Filter Out Outliers


'''
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
data_after_mad_removal = data[~data.index.isin(outliers_mad.index)]
print(" Outlier removal using Median Absolute Deviation (MAD)")
print("Number of rows after MAD removal:", len(data_after_mad_removal))
print("Number of rows removed:", len(outliers_mad))



# Outlier removal using IQR (Interquartile Range)
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
data_after_iqr_removal = data[~data.index.isin(outliers_iqr.index)]
print(" Outlier removal using IQR (Interquartile Range)")
print("Number of rows after IQR removal:", len(data_after_iqr_removal))
print("Number of rows removed:", len(outliers_iqr))
