# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/jadhavsainath/tourism_package_prediction/tourism.csv"
tourism_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Define the target variable for the classification task
target = 'ProdTaken'   # Whether the customer purchased the wellness tourism package

# List of numerical features in the dataset
numeric_features = [
    'Age',                        # Age of the customer
    'CityTier',                   # City category based on development and living standards
    'NumberOfPersonVisiting',     # Total number of people accompanying the customer
    'PreferredPropertyStar',      # Preferred hotel star rating
    'NumberOfTrips',              # Average number of trips taken annually
    'Passport',                   # Whether the customer holds a passport (0 or 1)
    'OwnCar',                     # Whether the customer owns a car (0 or 1)
    'NumberOfChildrenVisiting',   # Number of children accompanying the customer
    'MonthlyIncome',              # Gross monthly income of the customer
    'PitchSatisfactionScore',     # Satisfaction score for the sales pitch
    'NumberOfFollowups',          # Number of salesperson follow-ups
    'DurationOfPitch'             # Duration of the sales pitch
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',              # Method by which the customer was contacted
    'Occupation',                 # Customer's occupation
    'Gender',                     # Gender of the customer
    'MaritalStatus',              # Marital status of the customer
    'Designation',                # Customer's designation in their organization
    'ProductPitched'              # Type of product pitched to the customer
]

# Define predictor matrix (X) using selected numeric and categorical features
X = tourism_dataset[numeric_features + categorical_features]

# Define target variable
y = tourism_dataset[target]


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42,    # Ensures reproducibility by setting a fixed random seed,
    stratify=y
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="jadhavsainath/tourism_package_prediction",
        repo_type="dataset",
    )
