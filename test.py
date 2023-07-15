import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Step 1: Read the data file and the recommender file
data = pd.read_csv(r"C:\Users\KanoN\Desktop\grad-cloud\testing_the_algorithm\myenv\filtered_data.csv")
recommender = pd.read_csv(r"C:\Users\KanoN\Desktop\grad-cloud\testing_the_algorithm\myenv\Recos_for_patients.csv")

# Preprocessing the data file
# Example: Convert categorical variables to numerical representation
data['Sex'] = data['Sex'].map({'Female': 0, 'Male': 1})
data['Family History'] = data['Family History'].map({'Yes': 1, 'No': 0})
data['Smoking'] = data['Smoking'].map({'Yes': 1, 'No': 0})
data['Drinking'] = data['Drinking'].map({'Yes': 1, 'No': 0})
data['Lifestyle'] = data['Lifestyle'].map({'Sedentary': 0, 'Moderate': 1, 'Active': 2})
data['Eating Habits'] = data['Eating Habits'].map({'Junk Food': 0, 'Healthy Food': 1})
data['Gestational Diabetes'] = data['Gestational Diabetes'].map({'Yes': 1, 'No': 0})
data['Polycystic Ovaries'] = data['Polycystic Ovaries'].map({'Yes': 1, 'No': 0})
data['Waist Size'] = data['Waist Size']  # No mapping required for 'Waist Size' column (assuming it contains numerical values)
#data['Fasting Plasma Glucose'] = data['Fasting Plasma Glucose']  # No mapping required for 'Fasting Plasma Glucose' column (assuming it contains numerical values)
#data['Casual Glucose Tolerance'] = data['Casual Glucose Tolerance']  # No mapping required for 'Casual Glucose Tolerance' column (assuming it contains numerical values)


# Step 3: Implement hybrid filtering
def hybrid_recommender(input_data):
    # Content-based filtering
    content_based_filtered_data = data.copy()

    if input_data['Lifestyle'] == 'Moderate':
        content_based_filtered_data = content_based_filtered_data[content_based_filtered_data['Lifestyle'] == 1]

    if input_data['Lifestyle'] == 'Active':
        content_based_filtered_data = content_based_filtered_data[content_based_filtered_data['Lifestyle'] == 2]

    if input_data['Lifestyle'] == 'Sedentary':
        content_based_filtered_data = content_based_filtered_data[content_based_filtered_data['Lifestyle'] == 0]

    if input_data['Eating Habits'] == 'Junk Food':
        content_based_filtered_data = content_based_filtered_data[content_based_filtered_data['Eating Habits'] == 0]
    else:
        content_based_filtered_data = content_based_filtered_data[content_based_filtered_data['Eating Habits'] != 0]

    if input_data['Waist Size'] > 0:
        content_based_filtered_data = content_based_filtered_data[
            content_based_filtered_data['Waist Size'] <= input_data['Waist Size']]

    if input_data['Gestational Diabetes'] == 'Yes':
        content_based_filtered_data = content_based_filtered_data[
            content_based_filtered_data['Gestational Diabetes'] == 1]
    elif input_data['Gestational Diabetes'] == 'No':
        content_based_filtered_data = content_based_filtered_data[
            content_based_filtered_data['Gestational Diabetes'] == 0]

    if input_data['Drinking'] == 'Yes':
        content_based_filtered_data = content_based_filtered_data[content_based_filtered_data['Drinking'] == 1]
    elif input_data['Drinking'] == 'No':
        content_based_filtered_data = content_based_filtered_data[
            content_based_filtered_data['Drinking'] == 0]
    if input_data['Smoking'] == 'Yes':
        content_based_filtered_data = content_based_filtered_data[content_based_filtered_data['Smoking'] == 1]
    elif input_data['Smoking'] == 'No':
        content_based_filtered_data = content_based_filtered_data[
            content_based_filtered_data['Smoking'] == 0]



    if input_data['Polycystic Ovaries'] == 'No':
        content_based_filtered_data = content_based_filtered_data[
            content_based_filtered_data['Polycystic Ovaries'] == 1]
    '''
    if input_data['Fasting Plasma Glucose'] > 0:
        content_based_filtered_data = content_based_filtered_data[
            content_based_filtered_data['Fasting Plasma Glucose'] <= input_data['Fasting Plasma Glucose']]
    '''


    if content_based_filtered_data.empty:
        return []  # Return an empty list if the DataFrame is empty after filtering

    #print(content_based_filtered_data)
    #collaborative

    collaborative_filtered_data = content_based_filtered_data.copy()


    if len(collaborative_filtered_data) == 0:  # Check if there are any rows
        return []

    # Select user profile features
    user_profile = collaborative_filtered_data.iloc[0, [1, 3, 7, 8, 13, 14, 15, 17]].values.reshape(1, -1)

    # Drop rows with missing values in specific columns
    collaborative_filtered_data = collaborative_filtered_data.dropna(subset=[
        'Age', 'BMI', 'Lifestyle', 'Eating Habits', 'Waist Size', 'Gestational Diabetes',
        'Polycystic Ovaries', 'RecoID'])

    # Select relevant columns for collaborative filtering
    collaborative_filtered_data = collaborative_filtered_data[
        ['Age', 'BMI', 'Lifestyle', 'Eating Habits', 'Waist Size', 'Gestational Diabetes',
         'Polycystic Ovaries', 'RecoID']]

    # Compute cosine similarity between user profile and each row in the filtered data
    user_similarities = cosine_similarity(user_profile, collaborative_filtered_data)[0]

    # Add similarity scores to the data
    collaborative_filtered_data['Similarity'] = user_similarities

    # Sort the data by similarity in descending order
    collaborative_filtered_data = collaborative_filtered_data.sort_values(by='Similarity', ascending=False)

    # Exclude the first row (which is the user's own profile)
    collaborative_filtered_data = collaborative_filtered_data[1:]

    # Check if there are any recommendations
    if len(collaborative_filtered_data) == 0:
        return []
    # Check the values in the "ID" column of recommender DataFrame

    # Merge with recommender
    collaborative_filtered_data = collaborative_filtered_data.merge(recommender, how='left',
                                                                    left_on='RecoID', right_on='ID')

    #print(collaborative_filtered_data[['RecoID', 'ID', 'Recommendation']])
    #df =collaborative_filtered_data
    #pd.set_option('display.max_rows', None)
    #pd.set_option('display.max_columns', None)
    #print(df)
    # Exclude rows with missing values (nan) from recommendations
    collaborative_filtered_data = collaborative_filtered_data.dropna(subset=['RecoID'])

    recommended_actions = collaborative_filtered_data['Recommendation'].unique()
    #print("Unique RecoID values in collaborative_filtered_data:")
    #print(collaborative_filtered_data['RecoID'].unique())

    #print(collaborative_filtered_data)
    return recommended_actions[:1] # Return top 3 recommendations


# Step 4: Example cases
input_data1 = {
    'Lifestyle': 'Moderate',
    'Eating Habits': 'Junk Food',
    'Waist Size': 80,
    'Gestational Diabetes': 'Yes',
    'Polycystic Ovaries': 'No',
    'Smoking': 'Yes',
    'Drinking': 'Yes'
}
recommendations1 = hybrid_recommender(input_data1)

input_data2 = {
    'Lifestyle': 'Moderate',
    'Eating Habits': 'Junk Food',
    'Waist Size': 80,
    'Gestational Diabetes': 'Yes',
    'Polycystic Ovaries': 'Yes',
    'Smoking': 'Yes',
    'Drinking': 'Yes'
}

input_data3 = {
    'Lifestyle': 'Frequent',
    'Eating Habits': 'Healthy Food',
    'Waist Size': 90,
    'Gestational Diabetes': 'No',
    'Polycystic Ovaries': 'No',
    'Smoking': 'No',
    'Drinking': 'No'
}

# Step 5: Obtain recommendations for each case

#recommendations2 = hybrid_recommender(input_data2)
#recommendations3 = hybrid_recommender(input_data3)


# Step 6: Print the recommendations
print("Recommendations for Case 1:")
if len(recommendations1) == 0:
    print("No recommendations found.")
else:
    for recommendation in recommendations1:
        print(recommendation)

'''
print("Recommendations for Case 2:")
if len(recommendations1) == 0:
    print("No recommendations found.")
else:
    for recommendation in recommendations2:
        print(recommendation)

print("Recommendations for Case 3:")
if len(recommendations1) == 0:
    print("No recommendations found.")
else:
    for recommendation in recommendations3:
        print(recommendation)
'''