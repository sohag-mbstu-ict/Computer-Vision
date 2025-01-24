import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class CSVDateFilter:
    def __init__(self, input_file, output_file):
        """
        Initialize the class with input and output file paths.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.data = None

    def load_data(self):
        """
        Load the CSV file into a pandas DataFrame.
        """
        self.data = pd.read_csv(self.input_file)
        self.data['created_time'] = pd.to_datetime(self.data['created_time'], errors='coerce')

    def filter_by_date_range(self, start_date, end_date):
        """
        Filter the data by the specified date range.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Ensure the start and end dates are timezone-aware
        start_date = pd.Timestamp(start_date, tz='UTC')
        end_date = pd.Timestamp(end_date, tz='UTC')

        # Filter the data
        filtered_data = self.data[
            (self.data['created_time'] >= start_date) & 
            (self.data['created_time'] <= end_date)
        ]
        return filtered_data

    def get_filtered_data(self, start_date,end_date):
        """
        Save the filtered data to the output file.
        """
        # Load the data
        self.load_data()
        # Filter the data by the specified date range
        filtered_data = self.filter_by_date_range(start_date, end_date)
        filtered_data.to_csv(self.output_file, index=False)

    def get_unique_counts(self,data):
        # Count the occurrences of each unique value in the gender column
        gender_counts = data['gender'].value_counts()
        # Convert the gender counts into the desired dictionary format
        gender_counts = {'Gender': gender_counts.index.tolist(), 'Count': gender_counts.values.tolist()}

        country_counts = data['country'].value_counts()
        country_counts = {'Gender': country_counts.index.tolist(), 'Count': country_counts.values.tolist()}   
        # Display the counts
        print("Counts of different values in the 'gender' column:")
        print(gender_counts)  
        print("Counts of different values in the 'country' column:")
        print(country_counts)
        return gender_counts,country_counts

    def get_plot(self,plot_data,img_name):
        # Plot with seaborn
        plt.figure(figsize=(8, 6))
        bar_plot = sns.barplot(
            x='Gender', 
            y='Count', 
            data=plot_data, 
            palette='pastel', 
            hue=None, 
            legend=False  # Explicitly disable legend
        )
        plt.title('Gender Distribution', fontsize=16)
        plt.xlabel('Gender', fontsize=14)
        plt.ylabel('Count', fontsize=14)

        # Add numbers on top of the bars
        for bar in bar_plot.patches:
            bar_height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,  # X-coordinate (center of bar)
                bar_height,                         # Y-coordinate (top of bar)
                f'{int(bar_height)}',               # Text to display
                ha='center', va='bottom', fontsize=12  # Centered alignment
            )

        # Save the plot to a file
        plt.savefig(f'{img_name}.png')  # Save the plot as a PNG file

# Usage
if __name__ == "__main__":
    input_file = "/home/mtl/Downloads/users.csv"
    output_file = "/home/mtl/Music/Data_Analysis/output_csv/weekly.csv"
    start_date = '2024-04-03 11:40:30.851437+00:00'
    end_date = '2024-04-25 11:40:30.851437+00:00'
    # Create an instance of the class
    csv_filter = CSVDateFilter(input_file, output_file)
    
    # Save the filtered data to a CSV file
    # csv_filter.get_filtered_data(start_date,end_date)
    # print(f"Filtered data saved to {output_file}")

    # # Load the CSV file into a DataFrame
    data = pd.read_csv('/home/mtl/Music/Data_Analysis/output_csv/weekly.csv')
    # Count the total number of entries (rows)
    total_entries = data.shape[0]
    # Print the total number of entries
    print(f"Total number of entries: {total_entries}")
    
    gender_counts,country_counts = csv_filter.get_unique_counts(data)
    gender_counts
    csv_filter.get_plot(gender_counts,"gender")
    csv_filter.get_plot(country_counts,"country")





