# import all libraries 
import numpy as np 
import pickle 
import streamlit as st 
  

  
# Creating a function for prediction 
   
  
def main(): 
  
    # Giving a title 
    st.title('Crop Recommendation System') 
    
    options = ['Decision Tree', 'Random Forest', 'Support Vector Classifier','K-Nearest Neighbour','Ensemble']
    selected_value = st.selectbox("Select the model:", options)
    if selected_value=='Decision Tree':
        loaded_model = pickle.load(open('model_DT.pkl', 'rb')) 
    elif selected_value=='Random Forest':
        loaded_model = pickle.load(open('model_RF.pkl', 'rb'))
    elif selected_value=='Support Vector Classifier':
        loaded_model = pickle.load(open('model_SVM.pkl', 'rb')) 
    elif selected_value=='K-Nearest Neighbour':
        loaded_model = pickle.load(open('model_KNN.pkl', 'rb')) 
    elif selected_value=='Ensemble':
        loaded_model = pickle.load(open('model_EN.pkl', 'rb')) 

    # Getting input from the user 
    Nitrogen = st.text_input('Nitrogen Content:') 
    Phosphorus = st.text_input('Phosphorus level:') 
    potassium = st.text_input('potassium value:') 
    Temperature = st.text_input('Temperature value:') 
    Humidity = st.text_input('Humidity value:') 
    Ph = st.text_input('PH value:') 
    Rainfall = st.text_input('Rainfall: ') 
  
    # Code for prediction 
    diagnosis = '' 
    input_data=[Nitrogen, Phosphorus, potassium, Temperature , Humidity, Ph , Rainfall]
    input_data_as_nparray = np.asarray(input_data) 
    input_data_reshaped = input_data_as_nparray.reshape(1, -1) 
    prediction = loaded_model.predict(input_data_reshaped)
    
    # Making a button for prediction 
    if st.button('Predict'): 
        diagnosis = prediction
    st.success(diagnosis) 
  
if __name__ == '__main__': 
    main()