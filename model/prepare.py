import os
import csv
import re
import pandas as pd
from autocorrect import Speller

# Input and output folder paths
input_folder = 'model/import-labeled-data'
output_folder = 'model/export-labeled-data'

def process_review_file(input_file, output_file): 
    # Open the file and read the data
    with open(input_file, 'r', encoding='utf-8') as file:
        data = file.read()

    # Split the data into individual reviews
    reviews = re.findall(r'<review>(.*?)</review>', data, re.DOTALL)

    # Headers for the CSV file
    headers = [
        'unique_id', 'asin', 'product_name', 'product_type', 'helpful',
        'rating', 'title', 'date', 'reviewer', 'reviewer_location', 'review_text', 'type'
    ]

    # Open a CSV file for writing
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        # Iterate through each review and extract the fields
        for review in reviews:
            review_dict = {}
            for header in headers:
                pattern = f'<{header}>(.*?)</{header}>'
                match = re.search(pattern, review, re.DOTALL)
                review_dict[header] = match.group(1).strip() if match else ''

            # Determine the type based on the rating
            if float(review_dict['rating']) >= 4:
                review_dict['type'] = 1
            elif float(review_dict['rating']) <= 2:
                review_dict['type'] = 0

            # Check if the rating is within the specified intervals
            if float(review_dict['rating']) >= 4 or float(review_dict['rating']) <= 2:
                # Write the row to the CSV
                writer.writerow(review_dict)

    print(f"CSV file '{output_file}' created successfully.")

def process_all_review_files(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".review"):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.csv")
            process_review_file(input_file, output_file)

def merge_csv_files(folder_path, output_file):
    # List to hold all the dataframes
    all_dataframes = []

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            csv_file = os.path.join(folder_path, filename)
            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_file)
            all_dataframes.append(df)

    # Concatenate all DataFrames
    merged_df = pd.concat(all_dataframes, ignore_index=True)

    # Write the merged DataFrame to a new CSV file
    merged_df.to_csv(output_file, index=False)

merged_output_file = os.path.join(output_folder, 'merged_output.csv')  # Output file for merged CSV

# Process all .review files and save them as .csv
process_all_review_files(input_folder, output_folder)

# Merge all .csv files in the output folder into a single file
merge_csv_files(output_folder, merged_output_file)

# Load the CSV file from the local path
data = pd.read_csv(f'{output_folder}/merged_output.csv', sep=',', encoding='utf8')

# Print the number of reviews (rows) in the CSV file
print('\nNumber of reviews:', len(data))

# Delete the unnecessary columns
data.drop(['unique_id', 'asin', 'product_name', 'product_type', 'helpful',
           'date', 'reviewer', 'reviewer_location'], axis=1, inplace=True)

# Show the number of positive and negative reviews
type_review = data['type'].value_counts()
print("\nNumber of positive reviews:", type_review[1])
print("Number of negative reviews:", type_review[0])

print('\nUnique values in the rating column:')
print(data['rating'].unique())

print('\nFirst few rows of the data:')
print(data.head())

# Set language to Eng
spell = Speller(lang='en')

# Apply autocorrect with progress display
total_rows = len(data)
for i in range(total_rows):
    data.at[i, 'review_text_correct'] = spell(data.at[i, 'review_text'])
    if (i + 1) % 100 == 0 or i == total_rows - 1:  # Display progress every 100 rows
        print(f'Progress: {i + 1}/{total_rows}')

# Save the cleaned data to a CSV file
output_file = 'cleaned_reviews.csv'
data.to_csv(f'{output_folder}/{output_file}', index=False, encoding='utf-8')
print(f"\nCleaned data saved to {output_folder}/{output_file}")
