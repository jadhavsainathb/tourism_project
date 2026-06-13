
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download model from Hugging Face Hub

model_path = hf_hub_download(
repo_id="jadhavsainath/tourism-package-prediction",
filename="best_churn_model_v1.joblib"
)

# Load model

model = joblib.load(model_path)

st.set_page_config(
page_title="Wellness Tourism Package Predictor..",
page_icon="✈️",
layout="wide"
)

st.title("🏖️ Wellness Tourism Package Prediction")
st.markdown("""
This application helps **Visit with Us** identify customers who are likely
to purchase the **Wellness Tourism Package**.

Enter customer information and interaction details below to generate a prediction.
""")

st.header("Customer Details")

col1, col2 = st.columns(2)

with col1:
Age = st.number_input("Age", min_value=18, max_value=100, value=35)
CityTier = st.selectbox("City Tier", [1, 2, 3])
NumberOfPersonVisiting = st.number_input(
"Number of Persons Visiting",
min_value=1,
max_value=20,
value=2
)
PreferredPropertyStar = st.selectbox(
"Preferred Property Star",
[1, 2, 3, 4, 5]
)
NumberOfTrips = st.number_input(
"Number of Trips per Year",
min_value=0,
max_value=20,
value=2
)
Passport = st.selectbox("Has Passport", ["No", "Yes"])

with col2:
OwnCar = st.selectbox("Owns Car", ["No", "Yes"])
NumberOfChildrenVisiting = st.number_input(
"Number of Children Visiting",
min_value=0,
max_value=10,
value=0
)
MonthlyIncome = st.number_input(
"Monthly Income",
min_value=0,
value=50000
)

```
TypeofContact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Inquiry"]
)

Occupation = st.selectbox(
    "Occupation",
    [
        "Salaried",
        "Small Business",
        "Large Business",
        "Freelancer"
    ]
)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)
```

st.header("Travel & Customer Information")

col3, col4 = st.columns(2)

with col3:
MaritalStatus = st.selectbox(
"Marital Status",
["Single", "Married", "Divorced"]
)

```
Designation = st.selectbox(
    "Designation",
    [
        "Executive",
        "Manager",
        "Senior Manager",
        "AVP",
        "VP"
    ]
)
```

with col4:
ProductPitched = st.selectbox(
"Product Pitched",
[
"Basic",
"Standard",
"Deluxe",
"Super Deluxe",
"King"
]
)

```
PitchSatisfactionScore = st.slider(
    "Pitch Satisfaction Score",
    1,
    5,
    3
)

NumberOfFollowups = st.number_input(
    "Number of Follow-ups",
    min_value=0,
    max_value=20,
    value=3
)

DurationOfPitch = st.number_input(
    "Duration of Pitch (minutes)",
    min_value=0,
    max_value=120,
    value=15
)
```

input_data = pd.DataFrame([{
"Age": Age,
"CityTier": CityTier,
"NumberOfPersonVisiting": NumberOfPersonVisiting,
"PreferredPropertyStar": PreferredPropertyStar,
"NumberOfTrips": NumberOfTrips,
"Passport": 1 if Passport == "Yes" else 0,
"OwnCar": 1 if OwnCar == "Yes" else 0,
"NumberOfChildrenVisiting": NumberOfChildrenVisiting,
"MonthlyIncome": MonthlyIncome,
"PitchSatisfactionScore": PitchSatisfactionScore,
"NumberOfFollowups": NumberOfFollowups,
"DurationOfPitch": DurationOfPitch,
"TypeofContact": TypeofContact,
"Occupation": Occupation,
"Gender": Gender,
"MaritalStatus": MaritalStatus,
"Designation": Designation,
"ProductPitched": ProductPitched
}])

classification_threshold = 0.45

if st.button("Predict Purchase Probability"):

```
probability = model.predict_proba(input_data)[0, 1]
prediction = (probability >= classification_threshold).astype(int)

st.subheader("Prediction Result")

if prediction == 1:
    st.success(
        f"Customer is likely to purchase the Wellness Tourism Package. "
        f"(Probability: {probability:.2%})"
    )
else:
    st.warning(
        f"Customer is unlikely to purchase the Wellness Tourism Package. "
        f"(Probability: {probability:.2%})"
    )

st.progress(float(probability))
```
