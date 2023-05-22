import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn import *
import pickle 
import streamlit as st

 # load the model and dataset
# Load the saved pipeline object from the file
with open('pipeline_rf_1.pkl', 'rb') as f:
    loaded_pipeline = pickle.load(f)
    
df = pickle.load(open('car_details_data3.pkl','rb')) 

def main():
        # Set page config
    st.set_page_config(
        page_title="🚗CAR DEKHO🚗",
        page_icon=":car:",
        layout="centered",
        #background_color="#FFA500" # Orange background color
        )
main()


st.title('🚗 Used Car Selling Price Prediction App')
st.header('Fill the datails to predict used Car selling Price')

Brand_name = st.selectbox('Chose the Brand',df['Brand_name'].unique())
#Age = st.sidebar.number_input('Age of vehicle in years (1-32)', value=10)
year = st.selectbox('Chose the manufactured year of the car',df['year'].unique())
Km_driven = st.number_input('Enter the kilometers reading of the vehicle', value=300000)
fuel = st.selectbox('Fuel Type',['Petrol','Diesel','CNG','LPG','Electric'])
seller_type = st.selectbox('Seller type',df['seller_type'].unique())
transmission = st.selectbox('Select the type of Transmission',df['transmission'].unique())
owner = st.selectbox('Select the Type of Owner',df['owner'].unique())


d={ "Brand_name":  Brand_name,"year": year,"km_driven":Km_driven,"fuel":fuel,"seller_type"
 : seller_type,"transmission":transmission,"owner":owner}

test=pd.DataFrame(data=d,index=[0])
if st.button('Predict Car Selling Price \u20B9'):
   st.success(loaded_pipeline.predict(test)) 

    
#test = np.array([Brand_name,year,Km_driven,fuel,seller_type,transmission,owner])
#test = test.reshape([1,7])

    
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

        

