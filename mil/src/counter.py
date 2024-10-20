import pandas as pd
from collections import Counter

# Load the CSV data into a DataFrame
df = pd.read_csv('/home/alvi/KG_Defence/datasets/coco/csv_val.csv', sep='$')
# df = pd.read_csv('/home/alvi/KG_Defence/datasets/coco/csv_train.csv', sep='$')

# Assuming the file names are in the 'FileName' column, replace it with the actual column name
file_names = df['image_file']

# Count the occurrences of each file name
file_name_counts = Counter(file_names)

# Find file names that occur more than five times
more_than_five_occurrences = [name for name, count in file_name_counts.items() if count > 5]

# Print the file names with more than five occurrences
print("File names with more than five occurrences:", more_than_five_occurrences)

less_than_five_occurrences = [name for name, count in file_name_counts.items() if count < 5]

# Print the file names with more than five occurrences
print("File names with less than five occurrences:", less_than_five_occurrences)
