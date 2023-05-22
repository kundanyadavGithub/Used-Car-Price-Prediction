import pandas as pd   # data preprocessing
import numpy as np    # mathematical computation
from sklearn import *
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model and dataset
with open('pipeline_rf_1.pkl', 'rb') as f:
    loaded_pipeline = pickle.load(f)
    
df = pickle.load(open('car_details_data3.pkl','rb')) 


# Set page config
def main():
        
    st.set_page_config(
        page_title="🚗Car पूर्वदृष्टि🚗",
        page_icon=":car:",
        layout="centered",
        #background_color="#FFA500" # Orange background color
        
        )
    # Page title
    st.title("कुंदन द्वारा इस्तेमाल की गई कार(🚗) डेटा का EDA")
    
    # Add a scatter plot of the age and selling price
    
    st.subheader('Year vs Selling Price :money_with_wings:')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x="year", y="selling_price", ax=ax) # corrected y-axis label
    st.pyplot(fig)
    
    # Add a pair plot of the numerical features
    st.subheader('Pair Plot of Numerical Features :bar_chart:')
    num_cols = ['Brand_name','year','km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'selling_price']
    fig = sns.pairplot(df[num_cols], diag_kind='kde', plot_kws={'alpha': 0.4})
    fig.fig.set_size_inches(12, 10)
    st.pyplot(fig)
    

    # Add a correlation heatmap
    
    st.subheader('Correlation Heatmap :fire:')
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr,mask=mask,annot=True, cmap='RdBu', square=True, ax=ax)
    st.pyplot(fig)
    
    # Add the input form for the prediction using sidebar
    st.sidebar.header('📝 Fill the details to predict the Car Price')
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Enter Car Details")
    
    Brand_name = st.sidebar.selectbox('Chose the Brand',df['Brand_name'].unique())
    #Age = st.sidebar.number_input('Age of vehicle in years (1-32)', value=10)
    year = st.sidebar.selectbox('Chose the manufactured year of the car',df['year'].unique())
    Km_driven = st.sidebar.number_input('Enter the kilometers reading of the vehicle', value=300000)
    fuel = st.sidebar.selectbox('Fuel Type',['Petrol','Diesel','CNG','LPG','Electric'])
    seller_type = st.sidebar.selectbox('Seller type',df['seller_type'].unique())
    transmission = st.sidebar.selectbox('Select the type of Transmission',df['transmission'].unique())
    owner = st.sidebar.selectbox('Select the Type of Owner',df['owner'].unique())


    d={ "Brand_name":  Brand_name,"year": year,"km_driven":Km_driven,"fuel":fuel,"seller_type"
    : seller_type,"transmission":transmission,"owner":owner}

    test=pd.DataFrame(data=d,index=[0])
    if st.sidebar.button('Predict Car Selling Price \u20B9'):
        st.sidebar.success(loaded_pipeline.predict(test)) 
    
    
        #footer section
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Follow me on")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.sidebar.button('GitHub'):
            st.sidebar.markdown('[Visit my GitHub profile and check the codes](https://github.com/kundanyadavGithub/)')
    with col2:
        if st.sidebar.button('LinkedIn'):
            st.sidebar.markdown('[Visit my LinkedIn profile](https://www.linkedin.com/in/kundanyadav94/)')

if __name__ == "__main__":
    main()
