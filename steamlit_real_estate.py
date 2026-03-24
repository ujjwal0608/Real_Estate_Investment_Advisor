'''
Real Estate Investment Advisor
Streamlit Application - Predicting Property Profitability & Future Value
'''

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.metrics import roc_curve, auc
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page Configuration 
st.set_page_config(
    page_title="🏠 Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide"
)

st.markdown('''
<style>
    .main-header{
        font-size: 1.0rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #264653;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .good-investment {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 5px;
    }
    .bad-investment {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
    }
</style>
''', unsafe_allow_html = True)

# Title
st.markdown("<h1 class = 'main-header'> 🏠 Real Estate Investment Advisor</h1>", unsafe_allow_html = True)
st.markdown("<h3 class = 'sub-header'> ML-powered Property Investment Analysis & Future Price Prediction </h3>",
            unsafe_allow_html = True)

# Load Model, Artifact and Data
@st.cache_resource
def load_artifacts():
    try:
        reg_model = joblib.load('best_regression_model.pkl')
        clf_model = joblib.load('best_classification_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        
        try:
            House_Price_df = pd.read_csv('india_housing_prices_cleaned.csv')
        except:
            House_Price_df = None
        
        return reg_model, clf_model, scaler, label_encoders, feature_columns, House_Price_df
    
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None, None, None, None

reg_model, clf_model, scaler, label_encoders, feature_columns, df_clean = load_artifacts()

# Sidebar Navigation 
st.sidebar.header("📊 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Property Analysis", "📈 Market Insights", "🔍 Data Explorer", "⚙️ Model Info"])
    
# Buildung Helper Function 
def prepare_features(input_data):
    
    input_df = pd.DataFrame([input_data])
    
    # For Label Encoder
    for col in label_encoders.keys():
        if col in input_df.columns:
            try:
                input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
            except ValueError:
                input_df[col] = 0
                
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[feature_columns]
    
    # Scale Feature
    if scaler is not None:
        binary_cols = [c for c in input_df.columns if input_df[c].nunique() <= 2]
        scale_cols = [c for c in input_df.columns if c not in binary_cols]
        
        if len(scale_cols) > 0:
            try:
                input_df_scaled = input_df.copy()
                input_df_scaled[scale_cols] = scaler.transform(input_df[scale_cols])
                return input_df_scaled
            
            except:
                return input_df
        
    return input_df

def get_feature_importance(model, feature_names, top_n=10):
    
    # Get Feature importance from model 
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:][::-1]
        
        return pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': [importances[i] for i in indices]
        })
    
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
        indices = np.argsort(importances)[-top_n:][::-1]
        return pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': [importances[i] for i in indices]
        })
    return None
# ---------------------------Page 1: Property Analysis Start ----------------------------

if page == "🏠 Property Analysis":
    st.sidebar.header("📋 Property Details")
    
    with st.sidebar.form("property_form"):
        st.subheader("📍 Location")
        
        state = st.selectbox("State", sorted([
            'Andhra Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Delhi',
            'Gujarat', 'Haryana', 'Jharkhand', 'Karnataka', 'Kerala',
            'Madhya Pradesh', 'Maharashtra', 'Odisha', 'Punjab',
            'Rajasthan', 'Tamil Nadu', 'Telangana', 'Uttar Pradesh',
            'Uttarakhand', 'West Bengal'
        ]))
        
        city = st.selectbox("City", sorted([
            'Ahmedabad', 'Amritsar', 'Bangalore', 'Bhopal', 'Bhubaneswar',
            'Bilaspur', 'Chennai', 'Coimbatore', 'Cuttack', 'Dehradun',
            'Delhi', 'Durgapur', 'Dwarka', 'Faridabad', 'Gaya',
            'Guwahati', 'Haridwar', 'Hyderabad', 'Indore', 'Jaipur',
            'Jamshedpur', 'Jodhpur', 'Kochi', 'Kolkata', 'Lucknow',
            'Ludhiana', 'Mangalore', 'Mumbai', 'Mysore', 'Nagpur',
            'New Delhi', 'Noida', 'Patna', 'Pune', 'Raipur',
            'Ranchi', 'Silchar', 'Surat', 'Trivandrum', 'Vijayawada',
            'Vishakhapatnam', 'Warangal'
        ]))
        
        st.subheader("🏠 Property Details")
        col1, col2 = st.columns(2)
        
        with col1:
            property_type = st.selectbox("Property Type", ['Apartment', 'Independent House', 'Villa'])
            bhk = st.slider("BHK", 1, 5, 3)
        with col2:
            size_sqft = st.number_input("Size (SqFt)", min_value=500, max_value=5000, value=2500, step=100)
            current_price = st.number_input("Current Price (₹ Lakhs)", min_value=10.0, 
                                            max_value=500.0, value=250.0, step=10.0)
            
        st.subheader("📅 Construction Details")
        year_built = st.slider("Year Built", 1990, 2023, 2010)
        age_of_property = 2024 - year_built
        
        st.subheader("🛠️ Features")
        col1, col2 = st.columns(2)
        
        with col1:
            furnished_status = st.selectbox("Furnished Status", ['Furnished', 'Semi-furnished', 'Unfurnished'])
            floor_no = st.slider("Floor Number", 0, 30, 10)
            
        with col2:
            total_floors = st.slider("Total Floors", 1, 30, 15)
            
        
        st.subheader("🏫 Infrastructure")
        col1, col2, col3 = st.columns(3)
        with col1:
            nearby_schools = st.slider("Nearby Schools", 1, 10, 5)
        with col2:
            nearby_hospitals = st.slider("Nearby Hospitals", 1, 10, 5)
        with col3:
            transport_access = st.selectbox("Public Transport", ['Low', 'Medium', 'High'])
            
        st.subheader("🚗 Amenities")
        col1, col2 = st.columns(2)
        with col1:
            parking = st.radio("Parking Space", ['Yes', 'No'])
            facing = st.selectbox("Facing Direction", ['North', 'South', 'East', 'West'])
        with col2:
            security = st.radio("Security", ['Yes', 'No'])
            
        st.subheader("👤 Ownership")
        col1, col2 = st.columns(2)
        with col1:
            owner_type = st.selectbox("Owner Type", ['Owner', 'Builder', 'Broker'])
        with col2:
            availability = st.radio("Availability", ['Ready_to_Move', 'Under_Construction'])
            
        st.subheader("🎯 Amenities Checklist")
        col1, col2, col3 = st.columns(3)
        with col1:
            has_gym = st.checkbox("Gym")
            has_pool = st.checkbox("Swimming Pool")
        with col2:
            has_garden = st.checkbox("Garden")
            has_playground = st.checkbox("Playground")
        with col3:
            has_clubhouse = st.checkbox("Clubhouse")
        
        submit_button = st.form_submit_button("🔍 Analyze Property")
        
    if submit_button:
        with st.spinner("Analyzing property..."):
            price_per_sqft = current_price / size_sqft if size_sqft > 0 else 0
            amenities_count = sum([has_gym, has_pool, has_garden, has_playground, has_clubhouse])
            size_per_bhk = size_sqft / bhk if bhk > 0 else size_sqft
            floor_ratio = floor_no / total_floors if total_floors > 0 else 0
            is_high_floor = 1 if floor_ratio > 0.7 else 0
            is_new_property = 1 if age_of_property <= 5 else 0
            is_old_property = 1 if age_of_property > 25 else 0
            school_density = nearby_schools / 10.0
            hospital_density = nearby_hospitals / 10.0
            
            transport_map = {'High': 3, 'Medium': 2, 'Low': 1}
            transport_score = transport_map.get(transport_access, 1)
            infrastructure_score = (school_density + hospital_density + transport_score/3.0) / 3.0
            
            has_parking_bin = 1 if parking == 'Yes' else 0
            has_security_bin = 1 if security == 'Yes' else 0
            is_ready = 1 if availability == 'Ready_to_Move' else 0
            
            # Input Data 
            input_data = {
                'State': state, 'City': city, 'Property_Type': property_type,
                'BHK': bhk, 'Size_in_SqFt': size_sqft, 'Price_in_Lakhs': current_price,
                'Price_per_SqFt': price_per_sqft, 'Year_Built': year_built,
                'Furnished_Status': furnished_status, 'Floor_No': floor_no,
                'Total_Floors': total_floors, 'Age_of_Property': age_of_property,
                'Nearby_Schools': nearby_schools, 'Nearby_Hospitals': nearby_hospitals,
                'Public_Transport_Accessibility': transport_access, 'Parking_Space': parking,
                'Security': security, 'Facing': facing, 'Owner_Type': owner_type,
                'Availability_Status': availability, 'Amenities_Count': amenities_count,
                'Has_Gym': int(has_gym), 'Has_Pool': int(has_pool),
                'Has_Garden': int(has_garden), 'Has_Playground': int(has_playground),
                'Has_Clubhouse': int(has_clubhouse), 'Size_per_BHK': size_per_bhk,
                'Floor_Ratio': floor_ratio, 'Is_High_Floor': is_high_floor,
                'Is_New_Property': is_new_property, 'Is_Old_Property': is_old_property,
                'School_Density_Score': school_density, 'Hospital_Density_Score': hospital_density,
                'Transport_Score': transport_score, 'Infrastructure_Score': infrastructure_score,
                'Has_Parking': has_parking_bin, 'Has_Security': has_security_bin,
                'Is_Ready': is_ready
            }
            
            prepared_features = prepare_features(input_data)
            
            try:
                future_price = reg_model.predict(prepared_features)[0]
                investment_prob = clf_model.predict_proba(prepared_features)[0][1]
                investment_class = clf_model.predict(prepared_features)[0]
                
                appreciation_percent = ((future_price - current_price) / current_price) * 100
                annual_growth = ((future_price / current_price) ** (1/5) - 1) * 100
                
                st.success("✅ Analysis Complete!")
                
            #---------------Result--------------------
                st.markdown("## 📊 Investment Analysis Results")
                # Key Metrics Cards
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Current Price", f"₹{current_price:.1f}L")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("5-Year Price", f"₹{future_price:.1f}L", f"{appreciation_percent:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Annual Growth", f"{annual_growth:.1f}%", "CAGR")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                
                with col4:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    confidence_color = "🟢" if investment_prob > 0.7 else "🟡" if investment_prob > 0.5 else "🔴"
                    st.metric("Confidence", f"{investment_prob*100:.1f}%", confidence_color)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                # Investment Recommendation
                st.markdown("## 🎯 Investment Recommendation")
                
                if investment_class == 1:
                    st.markdown('<div class="good-investment">', unsafe_allow_html=True)
                    st.success(f"### 🟢 **GOOD INVESTMENT**")
                    st.write(f"**Confidence:** {investment_prob*100:.1f}% | **Probability:** {investment_prob:.3f}")
                    st.write("✅ Strong fundamentals  ✅ Good value proposition  ✅ High growth potential")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                else:
                    st.markdown('<div class="average-investment">', unsafe_allow_html=True)
                    st.warning(f"### 🟡 **AVERAGE INVESTMENT**")
                    st.write(f"**Confidence:** {investment_prob*100:.1f}% | **Probability:** {investment_prob:.3f}")
                    st.write("⚠️ Consider alternatives  ⚠️ Negotiate price  ⚠️ Evaluate carefully")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                # --------------------Visual Insight----------------------------------
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Price evolution chart
                    years = [0, 1, 2, 3, 4, 5]
                    prices = [current_price]
                    for i in range(1, 6):
                        prices.append(current_price * ((1 + annual_growth/100) ** i))
                    
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(x=years, y=prices, mode='lines+markers',
                                             name='Projected Price', line=dict(color='#4CAF50', width=3),
                                             marker=dict(size=8)))
                    fig1.update_layout(title='📊 Price Projection Over 5 Years',
                                      xaxis_title='Years',
                                      yaxis_title='Price (₹ Lakhs)',
                                      template='plotly_white')
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    # Feature importance
                    fi_reg = get_feature_importance(reg_model, feature_columns, top_n=10)
                    if fi_reg is not None:
                        fig2 = px.bar(fi_reg, x='Importance', y='Feature', orientation='h',
                                     title='🔍 Top 10 Features Affecting Future Price',
                                     color='Importance', color_continuous_scale='Blues')
                        fig2.update_layout(template='plotly_white', height=400)
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        
                        
                        
                col1, col2 = st.columns(2)
                
                with col1:
                    # Investment factors radar chart
                    categories = ['Price/Value', 'Infrastructure', 'Amenities', 'Location', 'Readiness']
                    values = [
                        min(100, max(0, 100 - (price_per_sqft * 1000))),  # Price per SqFt
                        infrastructure_score * 100,  # Infrastructure
                        min(100, amenities_count * 20),  # Amenities
                        70 if city in ['Bangalore', 'Mumbai', 'Delhi', 'Hyderabad'] else 50,  # Location
                        100 if is_ready == 1 else 60  # Readiness
                    ]
                    
                    fig3 = go.Figure(data=go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name='Property Score',
                        line_color='#2196F3'
                    ))
                    fig3.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                        title='📊 Property Score Radar Chart',
                        template='plotly_white',
                        height=400
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                
                with col2:
                    # Market comparison (if data available)
                    if df_clean is not None:
                        city_avg = df_clean[df_clean['City'] == city]['Price_in_Lakhs'].mean() if city in df_clean['City'].values else current_price
                        type_avg = df_clean[df_clean['Property_Type'] == property_type]['Price_in_Lakhs'].mean()
                        
                        comparison_data = pd.DataFrame({
                            'Category': ['This Property', f'{city} Avg', f'{property_type} Avg'],
                            'Price (₹L)': [current_price, city_avg, type_avg]
                        })
                        
                        fig4 = px.bar(comparison_data, x='Category', y='Price (₹L)',
                                     title='🏙️ Price Comparison with Market Averages',
                                     color='Category', text='Price (₹L)',
                                     color_discrete_map={'This Property': '#4CAF50', 
                                                         f'{city} Avg': '#2196F3',
                                                         f'{property_type} Avg': '#FF9800'})
                        fig4.update_traces(texttemplate='₹%{y:.0f}L', textposition='outside')
                        fig4.update_layout(template='plotly_white', height=400)
                        st.plotly_chart(fig4, use_container_width=True)
                    
                st.markdown("## 🔍 Detailed Breakdown")
                
                # Property Scorecard
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 💰 Price Analysis")
                    st.metric("Price per SqFt", f"₹{price_per_sqft:.3f}L", 
                             "Good" if price_per_sqft < 0.09 else "Average")
                    st.metric("Size per BHK", f"{size_per_bhk:.0f} sqft")
                    st.metric("Value Score", f"{(100 - price_per_sqft*1000):.0f}/100")
                
                with col2:
                    st.markdown("### 🏗️ Infrastructure")
                    st.metric("Infrastructure Score", f"{infrastructure_score:.2f}/1.0")
                    st.metric("Schools", nearby_schools, f"Density: {school_density:.1f}")
                    st.metric("Hospitals", nearby_hospitals, f"Density: {hospital_density:.1f}")
                    st.metric("Transport", transport_access, f"Score: {transport_score}")
                
                with col3:
                    st.markdown("### 🎯 Features & Amenities")
                    st.metric("Amenities Count", amenities_count, f"Max: 5")
                    st.metric("Property Age", f"{age_of_property} years", 
                             "New" if is_new_property == 1 else "Old" if is_old_property == 1 else "Mid")
                    st.metric("Readiness", "Ready" if is_ready == 1 else "Under Construction")
                    st.metric("Security", "Yes" if has_security_bin == 1 else "No")
                    
                    
                st.markdown("## 💡 Recommendations & Next Steps")
                
                if investment_class == 1:
                    st.info("""
                    **🎯 Strong Buy Recommendation:**
                    
                    1. **Immediate Actions:**
                       - Schedule property visit within 7 days
                       - Get professional valuation report
                       - Check builder/owner reputation
                    
                    2. **Negotiation Strategy:**
                       - Target price: ₹{}L - ₹{}L
                       - Highlight: {}, {}, {}
                    
                    3. **Long-term Strategy:**
                       - Hold for minimum 5 years
                       - Consider rental income potential
                       - Monitor infrastructure developments
                    
                    **📈 Expected Returns:**
                    - 5-year appreciation: {}%
                    - Annual growth: {}%
                    - Break-even period: ~{} years
                    """.format(
                        current_price*0.95, current_price*0.98,
                        "Good location" if infrastructure_score > 0.7 else "Average location",
                        "Premium amenities" if amenities_count >= 4 else "Basic amenities",
                        "Ready to move" if is_ready == 1 else "Future delivery",
                        appreciation_percent, annual_growth,
                        3 if annual_growth > 10 else 4 if annual_growth > 7 else 5
                    ))
                else:
                    st.info("""
                    **⚠️ Consider with Caution:**
                    
                    1. **Risk Assessment:**
                       - Price {}% {} market average
                       - Infrastructure score: {}/10
                       - Amenities: {}/5
                    
                    2. **Alternative Options:**
                       - Compare with 3-5 similar properties
                       - Explore nearby localities
                       - Consider different property types
                    
                    3. **If Proceeding:**
                       - Negotiate hard (target {}% discount)
                       - Get thorough inspection
                       - Verify all approvals and documents
                    
                    **📉 Risk Factors:**
                    - Lower than average growth potential
                    - {} infrastructure development
                    - {} rental demand
                    """.format(
                        abs(price_per_sqft - 0.09)*1000, "above" if price_per_sqft > 0.09 else "below",
                        int(infrastructure_score * 10), amenities_count,
                        10 if price_per_sqft > 0.1 else 5,
                        "Slow" if infrastructure_score < 0.5 else "Moderate",
                        "Lower" if amenities_count < 3 else "Moderate"
                    ))
                    
                #-------------------Download Reports----------------------------
                    
                st.markdown("## 📥 Export Analysis")
                
                report_content = f"""
                REAL ESTATE INVESTMENT ANALYSIS REPORT
                =======================================
                
                PROPERTY DETAILS:
                - Location: {city}, {state}
                - Property Type: {property_type}
                - BHK: {bhk}
                - Size: {size_sqft} sqft
                - Current Price: ₹{current_price:.1f}L
                - Year Built: {year_built}
                - Age: {age_of_property} years
                
                PREDICTION RESULTS:
                - Future Price (5 Years): ₹{future_price:.1f}L
                - Total Appreciation: {appreciation_percent:.1f}%
                - Annual Growth Rate: {annual_growth:.1f}%
                - Investment Rating: {'GOOD INVESTMENT' if investment_class == 1 else 'AVERAGE INVESTMENT'}
                - Confidence Score: {investment_prob*100:.1f}%
                
                KEY METRICS:
                - Price per SqFt: ₹{price_per_sqft:.4f}L
                - Infrastructure Score: {infrastructure_score:.2f}/1.0
                - Amenities Count: {amenities_count}/5
                - Transport Access: {transport_access}
                - School Density: {school_density:.1f}
                - Hospital Density: {hospital_density:.1f}
                
                RECOMMENDATION:
                {'STRONG BUY - Property shows excellent fundamentals and growth potential.' if investment_class == 1 else 'CAUTION ADVISED - Consider alternatives or negotiate better price.'}
                
                ANALYSIS DATE: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                """
                
                st.download_button(
                    label="📄 Download Detailed Report (TXT)",
                    data=report_content,
                    file_name=f"property_analysis_{city}_{property_type}_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
                
    else:
            st.markdown("""
            ## 🏠 Welcome to Property Investment Analyzer

            This AI-powered tool analyzes real estate investments using machine learning models 
            trained on **250,000+ Indian properties**.

            ### 📋 How to Use:
            1. **Fill** property details in the sidebar
            2. **Click** "Analyze Property" 
            3. **Get** instant predictions and insights

            ### 🎯 What You'll Get:
            - ✅ **Future Price Prediction** (5 years)
            - ✅ **Investment Recommendation** (Good/Average)
            - ✅ **Annual Growth Rate** (CAGR)
            - ✅ **Visual Insights** & Charts
            - ✅ **Detailed Analysis** & Recommendations
            - ✅ **Market Comparisons**

            *All predictions are based on historical trends and should be used as guidance only.*
            """)
            if st.checkbox("Show Sample Analysis"):
                    sample_col1, sample_col2, sample_col3 = st.columns(3)
                    with sample_col1:
                        st.metric("Sample Property", "3 BHK Apartment", "Bangalore")
                    with sample_col2:
                        st.metric("Current Price", "₹275L", "")
                    with sample_col3:
                        st.metric("Predicted 5Y", "₹458L", "+66.5%")

# ==================== PAGE 2: MARKET INSIGHTS ====================
elif page == "📈 Market Insights":
    st.markdown('<h2 class="sub-header">📈 Market Insights & Trends</h2>', unsafe_allow_html=True)
    
    if df_clean is not None:
        # Filters
        st.sidebar.header("🔍 Filter Data")
        selected_state = st.sidebar.multiselect("Select State(s)", sorted(df_clean['State'].unique()), 
                                               default=['Maharashtra', 'Karnataka'])
        selected_city = st.sidebar.multiselect("Select City(s)", sorted(df_clean['City'].unique()),
                                              default=['Mumbai', 'Bangalore'])
        selected_type = st.sidebar.multiselect("Property Type", df_clean['Property_Type'].unique(),
                                              default=df_clean['Property_Type'].unique())
        
        # Filter data
        filtered_df = df_clean[
            (df_clean['State'].isin(selected_state)) &
            (df_clean['City'].isin(selected_city)) &
            (df_clean['Property_Type'].isin(selected_type))
        ]
        
        if len(filtered_df) > 0:
            # Market Overview Metrics
            st.markdown("### 📊 Market Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Price", f"₹{filtered_df['Price_in_Lakhs'].mean():.1f}L")
            with col2:
                st.metric("Avg Size", f"{filtered_df['Size_in_SqFt'].mean():.0f} sqft")
            with col3:
                st.metric("Avg Price/SqFt", f"₹{filtered_df['Price_per_SqFt'].mean():.3f}L")
            with col4:
                st.metric("Good Investment %", f"{(filtered_df['Good_Investment'].mean()*100):.1f}%")
            
            # Visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Trends", "🏙️ City Analysis", "🏠 Property Types", "📊 Investment Analysis"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    # Price by Year Built
                    year_price = filtered_df.groupby('Year_Built')['Price_in_Lakhs'].mean().reset_index()
                    fig = px.line(year_price, x='Year_Built', y='Price_in_Lakhs',
                                 title='Average Price by Construction Year',
                                 markers=True)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Price Distribution
                    fig = px.histogram(filtered_df, x='Price_in_Lakhs', nbins=50,
                                      title='Price Distribution',
                                      color_discrete_sequence=['#2196F3'])
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # City-wise Analysis
                city_stats = filtered_df.groupby('City').agg({
                    'Price_in_Lakhs': 'mean',
                    'Size_in_SqFt': 'mean',
                    'Good_Investment': 'mean',
                    'Price_per_SqFt': 'mean'
                }).round(2).reset_index()
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(city_stats.sort_values('Price_in_Lakhs', ascending=False).head(10),
                                x='City', y='Price_in_Lakhs',
                                title='Top 10 Cities by Average Price',
                                color='Price_in_Lakhs',
                                color_continuous_scale='Viridis')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.scatter(city_stats, x='Price_in_Lakhs', y='Good_Investment',
                                    size='Size_in_SqFt', color='City',
                                    title='Price vs Investment Potential',
                                    hover_data=['City', 'Price_per_SqFt'])
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Property Type Analysis
                type_stats = filtered_df.groupby('Property_Type').agg({
                    'Price_in_Lakhs': ['mean', 'median', 'count'],
                    'Good_Investment': 'mean'
                }).round(2)
                type_stats.columns = ['Avg Price', 'Median Price', 'Count', 'Good Investment %']
                type_stats['Good Investment %'] = type_stats['Good Investment %'] * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(type_stats.style.format({
                        'Avg Price': '₹{:.1f}L',
                        'Median Price': '₹{:.1f}L',
                        'Good Investment %': '{:.1f}%'
                    }).background_gradient(cmap='YlOrBr'), height=300)
                
                with col2:
                    fig = px.pie(filtered_df, names='Property_Type',
                                title='Property Type Distribution',
                                hole=0.4)
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                # Investment Analysis
                col1, col2 = st.columns(2)
                with col1:
                    # Good vs Not Good distribution
                    fig = px.pie(filtered_df, names='Good_Investment',
                                title='Investment Quality Distribution',
                                labels={'0': 'Not Good', '1': 'Good'},
                                hole=0.4)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Factors affecting investment
                    factors_df = pd.DataFrame({
                        'Factor': ['Price/SqFt < 0.09', 'BHK ≥ 3', 'Infrastructure ≥ 0.7', 
                                  'Amenities ≥ 4', 'Ready to Move', 'Has Security'],
                        'Good Investment %': [
                            filtered_df[filtered_df['Price_per_SqFt'] < 0.09]['Good_Investment'].mean() * 100,
                            filtered_df[filtered_df['BHK'] >= 3]['Good_Investment'].mean() * 100,
                            filtered_df[filtered_df['Infrastructure_Score'] >= 0.7]['Good_Investment'].mean() * 100,
                            filtered_df[filtered_df['Amenities_Count'] >= 4]['Good_Investment'].mean() * 100,
                            filtered_df[filtered_df['Is_Ready'] == 1]['Good_Investment'].mean() * 100,
                            filtered_df[filtered_df['Has_Security'] == 1]['Good_Investment'].mean() * 100
                        ]
                    })
                    fig = px.bar(factors_df, x='Factor', y='Good Investment %',
                                title='Factors Affecting Investment Quality',
                                color='Good Investment %',
                                color_continuous_scale='Greens')
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for selected filters.")
    else:
        st.warning("Cleaned data not available for market insights.")

# ==================== PAGE 3: DATA EXPLORER ====================
elif page == "🔍 Data Explorer":
    st.markdown('<h2 class="sub-header">🔍 Data Explorer</h2>', unsafe_allow_html=True)
    
    if df_clean is not None:
        # Data filters
        st.sidebar.header("🔍 Data Filters")
        
        # Numeric filters
        col1, col2 = st.sidebar.columns(2)
        with col1:
            min_price = st.number_input("Min Price (₹L)", 
                                       value=float(df_clean['Price_in_Lakhs'].min()),
                                       min_value=0.0)
            max_price = st.number_input("Max Price (₹L)",
                                       value=float(df_clean['Price_in_Lakhs'].max()),
                                       min_value=0.0)
        with col2:
            min_size = st.number_input("Min Size (SqFt)",
                                      value=float(df_clean['Size_in_SqFt'].min()),
                                      min_value=0.0)
            max_size = st.number_input("Max Size (SqFt)",
                                      value=float(df_clean['Size_in_SqFt'].max()),
                                      min_value=0.0)
        
        # Categorical filters
        selected_states = st.sidebar.multiselect("States", sorted(df_clean['State'].unique()))
        selected_cities = st.sidebar.multiselect("Cities", sorted(df_clean['City'].unique()))
        selected_bhk = st.sidebar.multiselect("BHK", sorted(df_clean['BHK'].unique()))
        
        # Apply filters
        filtered_data = df_clean.copy()
        filtered_data = filtered_data[
            (filtered_data['Price_in_Lakhs'] >= min_price) &
            (filtered_data['Price_in_Lakhs'] <= max_price) &
            (filtered_data['Size_in_SqFt'] >= min_size) &
            (filtered_data['Size_in_SqFt'] <= max_size)
        ]
        
        if selected_states:
            filtered_data = filtered_data[filtered_data['State'].isin(selected_states)]
        if selected_cities:
            filtered_data = filtered_data[filtered_data['City'].isin(selected_cities)]
        if selected_bhk:
            filtered_data = filtered_data[filtered_data['BHK'].isin(selected_bhk)]
        
        # Display data
        st.write(f"**Showing {len(filtered_data):,} properties** (Total: {len(df_clean):,})")
        
        # Data tabs
        tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "📊 Statistics", "🔍 Search & Filter"])
        
        with tab1:
            # Show raw data with pagination
            page_size = 100
            page_number = st.number_input("Page", min_value=1, 
                                         max_value=len(filtered_data)//page_size + 1, 
                                         value=1)
            start_idx = (page_number - 1) * page_size
            end_idx = start_idx + page_size
            
            st.dataframe(filtered_data.iloc[start_idx:end_idx].style.format({
                'Price_in_Lakhs': '₹{:.1f}L',
                'Future_Price_5Y': '₹{:.1f}L',
                'Price_per_SqFt': '₹{:.4f}L',
                'Infrastructure_Score': '{:.2f}'
            }), height=400)
            
            # Download filtered data
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data (CSV)",
                data=csv,
                file_name=f"filtered_properties_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with tab2:
            # Statistics
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Numerical Statistics:**")
                st.dataframe(filtered_data.select_dtypes(include=[np.number]).describe().round(2))
            
            with col2:
                st.write("**Categorical Statistics:**")
                for col in filtered_data.select_dtypes(include=['object']).columns:
                    if col in ['State', 'City', 'Property_Type', 'Furnished_Status']:
                        counts = filtered_data[col].value_counts().head(10)
                        st.write(f"**{col}:**")
                        st.write(counts)
        
        with tab3:
            # Advanced search
            st.write("### 🔍 Advanced Search")
            search_query = st.text_input("Search in all columns (comma-separated terms):")
            
            if search_query:
                search_terms = [term.strip().lower() for term in search_query.split(',')]
                mask = pd.Series(False, index=filtered_data.index)
                
                for term in search_terms:
                    if term:
                        for col in filtered_data.columns:
                            if filtered_data[col].dtype == 'object':
                                mask = mask | filtered_data[col].astype(str).str.lower().str.contains(term)
                
                search_results = filtered_data[mask]
                st.write(f"**Found {len(search_results)} properties matching your search**")
                st.dataframe(search_results.head(50))
    
    else:
        st.warning("Cleaned data not available for exploration.")

# ==================== PAGE 4: MODEL INFO ====================
elif page == "⚙️ Model Info":
    st.markdown('<h2 class="sub-header">⚙️ Model Information & Performance</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Regression Model")
        st.markdown("""
        **Model:** Gradient Boosting Regressor  
        **Target:** Future_Price_5Y (5-year price prediction)  
        **Training Samples:** 200,000 properties  
        **Test Samples:** 50,000 properties  
        
        **Performance Metrics:**
        - **R² Score:** 0.9973 (99.73% variance explained)
        - **RMSE:** 11.89 Lakhs
        - **MAE:** 8.41 Lakhs
        - **Max Error:** ~50 Lakhs
        
        **Key Features:**
        1. Current Price (Most important)
        2. City/Location
        3. Infrastructure Score
        4. Amenities Count
        5. Property Type
        """)
        
        if reg_model is not None:
            fi_reg = get_feature_importance(reg_model, feature_columns, top_n=10)
            if fi_reg is not None:
                st.write("**Top 10 Important Features:**")
                st.dataframe(fi_reg.style.format({'Importance': '{:.4f}'}))
    
    with col2:
        st.markdown("### 🎯 Classification Model")
        st.markdown("""
        **Model:** XGBoost Classifier  
        **Target:** Good_Investment (Binary classification)  
        **Training Samples:** 200,000 properties  
        **Test Samples:** 50,000 properties  
        
        **Performance Metrics:**
        - **Accuracy:** 99.69%
        - **Precision:** 99.56%
        - **Recall:** 99.68%
        - **F1 Score:** 99.62%
        - **ROC AUC:** 1.000 (Perfect)
        
        **Classification Threshold:**
        - **Good Investment:** Probability ≥ 0.5
        - **Not Good:** Probability < 0.5
        
        **Key Decision Factors:**
        1. Price per SqFt
        2. Current Price
        3. Amenities Count
        4. Infrastructure Score
        5. BHK Configuration
        """)
        
        if clf_model is not None:
            fi_clf = get_feature_importance(clf_model, feature_columns, top_n=10)
            if fi_clf is not None:
                st.write("**Top 10 Important Features:**")
                st.dataframe(fi_clf.style.format({'Importance': '{:.4f}'}))
    
    st.markdown("---")
    st.markdown("### 📊 Model Comparison")
    
    # Model comparison metrics
    comparison_data = pd.DataFrame({
        'Model': ['Gradient Boosting', 'XGBoost', 'Random Forest', 'Linear Regression', 'Logistic Regression'],
        'Type': ['Regression', 'Classification', 'Both', 'Regression', 'Classification'],
        'Metric': ['R²: 0.9973', 'Accuracy: 99.69%', 'R²: 0.9969 / Acc: 99.54%', 'R²: 0.9852', 'Accuracy: 90.40%'],
        'Training Time': ['Medium', 'Fast', 'Slow', 'Very Fast', 'Very Fast'],
        'Interpretability': ['Medium', 'Medium', 'High', 'High', 'High']
    })
    
    st.dataframe(comparison_data.style.highlight_max(subset=['Metric'], color='lightgreen'))
    
    st.markdown("---")
    st.markdown("### 📁 Artifacts Information")
    
    artifacts_info = pd.DataFrame({
        'File': ['best_regression_model.pkl', 'best_classification_model.pkl', 'scaler.pkl', 
                'label_encoders.pkl', 'feature_columns.pkl', 'india_housing_prices_cleaned.csv'],
        'Size': ['~50 MB', '~45 MB', '~5 MB', '~2 MB', '~1 KB', '~150 MB'],
        'Description': ['Gradient Boosting model', 'XGBoost model', 'Feature scaler', 
                       'Categorical encoders', 'Feature column order', 'Cleaned dataset with targets']
    })
    
    st.dataframe(artifacts_info)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
    <p>🏠 <b>Real Estate Investment Advisor</b> | Machine Learning Project | Built with Streamlit & Scikit-learn</p>
    <p><small>Disclaimer: Predictions are for informational purposes only. Always consult with financial advisors before making investment decisions.</small></p>
    <p><small>© 2024 | Dataset: 250,000 Indian Properties | Models: Gradient Boosting & XGBoost</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
