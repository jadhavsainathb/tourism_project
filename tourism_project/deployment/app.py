
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Wellness Tourism Package Predictor",
    page_icon="🏖️",
    layout="wide"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="jadhavsainath/tourism-model",
        filename="best_tourism_model_v1.joblib"
    )
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {str(e)}")
    st.stop()

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🏖️ Wellness Tourism Package Prediction")

st.markdown("""
This application helps **Visit with Us** identify customers who are likely to purchase the **Wellness Tourism Package**.

Fill in the customer profile below and click **Predict Purchase Probability**.
""")

# --------------------------------------------------
# Input Form
# --------------------------------------------------

with st.expander("📝 Enter Customer Information", expanded=True):

    with st.form("prediction_form"):

        col1, col2, col3 = st.columns(3)

        # -----------------------------------
        # Column 1 - Customer Information
        # -----------------------------------

        with col1:

            st.subheader("👤 Customer")

            Age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=35
            )

            CityTier = st.selectbox(
                "City Tier",
                [1, 2, 3]
            )

            MonthlyIncome = st.number_input(
                "Monthly Income",
                min_value=0,
                value=50000
            )

            Gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            MaritalStatus = st.selectbox(
                "Marital Status",
                ["Single", "Married", "Divorced"]
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

        # -----------------------------------
        # Column 2 - Travel Information
        # -----------------------------------

        with col2:

            st.subheader("✈️ Travel")

            NumberOfTrips = st.number_input(
                "Trips Per Year",
                min_value=0,
                max_value=20,
                value=2
            )

            NumberOfPersonVisiting = st.number_input(
                "Persons Visiting",
                min_value=1,
                max_value=20,
                value=2
            )

            NumberOfChildrenVisiting = st.number_input(
                "Children Visiting",
                min_value=0,
                max_value=10,
                value=0
            )

            PreferredPropertyStar = st.selectbox(
                "Preferred Property Star",
                [1, 2, 3, 4, 5]
            )

            Passport = st.selectbox(
                "Passport",
                ["No", "Yes"]
            )

            OwnCar = st.selectbox(
                "Own Car",
                ["No", "Yes"]
            )

        # -----------------------------------
        # Column 3 - Sales Interaction
        # -----------------------------------

        with col3:

            st.subheader("📞 Sales Interaction")

            TypeofContact = st.selectbox(
                "Type of Contact",
                ["Company Invited", "Self Inquiry"]
            )

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

            PitchSatisfactionScore = st.slider(
                "Pitch Satisfaction Score",
                min_value=1,
                max_value=5,
                value=3
            )

            NumberOfFollowups = st.number_input(
                "Number Of Followups",
                min_value=0,
                max_value=20,
                value=3
            )

            DurationOfPitch = st.number_input(
                "Duration Of Pitch (Minutes)",
                min_value=0,
                max_value=120,
                value=15
            )

        submitted = st.form_submit_button(
            "🔮 Predict Purchase Probability",
            use_container_width=True
        )

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if submitted:

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

    with st.spinner("🔍 Analyzing customer profile..."):

        try:

            prediction_proba = model.predict_proba(input_data)[0, 1]

            prediction = int(
                prediction_proba >= classification_threshold
            )

            st.divider()

            st.subheader("📊 Prediction Result")

            c1, c2 = st.columns(2)

            with c1:
                st.metric(
                    "Purchase Probability",
                    f"{prediction_proba:.2%}"
                )

            with c2:
                st.metric(
                    "Prediction",
                    "Likely Buyer" if prediction else "Unlikely Buyer"
                )

            st.progress(float(prediction_proba))

            if prediction_proba >= 0.75:

                st.success(
                    "🟢 Very High Purchase Potential"
                )

                st.balloons()

                st.info(
                    "🎯 Recommended Action: Contact immediately and offer premium package discounts."
                )

            elif prediction_proba >= 0.50:

                st.info(
                    "🔵 Moderate Purchase Potential"
                )

                st.info(
                    "🎯 Recommended Action: Schedule follow-up calls and targeted promotions."
                )

            else:

                st.warning(
                    "🟠 Low Purchase Potential"
                )

                st.warning(
                    "🎯 Recommended Action: Add customer to a nurturing campaign."
                )

            with st.expander("📋 Customer Summary"):

                st.dataframe(
                    input_data,
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                f"Prediction failed: {str(e)}"
            )

