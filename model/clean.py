import pandas as pd

# Output folder path
output_folder = 'model/export-labeled-data'

# Load the CSV file from the local path
data = pd.read_csv(f'{output_folder}/cleaned_reviews.csv', sep=',', encoding='utf8')

# Working with the data....