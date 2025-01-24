import pandas as pd

def get_location():
    # Data for all districts with coordinates
    data = {
            'Name': ['Dhaka', 'Chittagong', 'Khulna', 'Rajshahi', 'Barisal', 'Sylhet', 'Rangpur', 'Mymensingh'],
            'lat': [23.8103, 22.3569, 22.8456, 24.3745, 22.7010, 24.8949, 25.7466, 24.7471],
            'lon': [90.4125, 91.7832, 89.5403, 88.6042, 90.3535, 91.8687, 89.2508, 90.4203]
        }

        # Create a DataFrame
    # district_data = pd.DataFrame(data)
    # print("district_data : ",district_data)
    district_data = pd.read_csv("/home/mtl/Music/Data_Analysis/filtered_data.csv")
    # district_data = pd.DataFrame(data)
    return district_data
