import pandas as pd
from datetime import datetime

# Load the CSV file
file_path = "/home/mtl/Downloads/users.csv"
data = pd.read_csv(file_path)

import pandas as pd
from datetime import datetime

# Convert 'created_time' to datetime format (handling timezone)
data['created_time'] = pd.to_datetime(data['created_time'], errors='coerce')

# Define the start and end date range and make them timezone-aware
start_date = pd.Timestamp('2020-10-01 11:40:30.851437+00:00', tz='UTC')
end_date   = pd.Timestamp('2025-01-14 11:40:30.851437+00:00', tz='UTC')

# Filter the data where 'created_time' is within the specified date range
filtered_data = data[(data['created_time'] >= start_date) & (data['created_time'] <= end_date)]


# Print the filtered data
# print("Filtered Data:")
# print(filtered_data)

# Optionally save the filtered data to a new CSV file
filtered_file_path = "filtered_file.csv"  # Replace with the desired output file path
filtered_data.to_csv(filtered_file_path, index=False)
# print(f"Filtered data saved to {filtered_file_path}")


# Load the CSV file
data = pd.read_csv('/home/mtl/Music/Data_Analysis/filtered_file.csv')

# Filter rows where latitude is not null
filtered_data = data.dropna(subset=['latitude'])

# Select only latitude, longitude, and phone_number columns
result = filtered_data[['latitude', 'longitude', 'phone_number']]

# Save the result to a new CSV or display it
result.to_csv('filtered_data.csv', index=False)

# If you want to print the result
print(result)