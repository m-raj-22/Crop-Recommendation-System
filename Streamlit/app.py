# import all libraries 
import numpy as np 
import pickle 
import streamlit as st 
  
# Loading the saved model 
loaded_model = pickle.load(open('model_SVM.pkl', 'rb')) 
  
# Creating a function for prediction 
  
def crop_prediction(input_data): 
  
    # Changing the data into a NumPy array 
    input_data_as_nparray = np.asarray(input_data) 
  
    # Reshaping the data since there is only one instance 
    input_data_reshaped = input_data_as_nparray.reshape(1, -1) 
  
    prediction = loaded_model.predict(input_data_reshaped) 
  
    return prediction
  
def main(): 
  
    # Giving a title 
    st.title('Crop Recommendation System') 
    
    

    # Getting input from the user 
    Nitrogen = st.text_input('Nitrogen Content:') 
    Phosphorus = st.text_input('Phosphorus level:') 
    potassium = st.text_input('potassium value:') 
    Temperature = st.text_input('Temperature value:') 
    Humidity = st.text_input('Humidity value:') 
    Ph = st.text_input('PH value:') 
    Rainfall = st.text_input('Rainfall: ') 
  
    # Code for prediction 
    label = '' 
    diagnosis=''
    # Making a button for prediction 
    if st.button('Predict'): 
        diagnosis = crop_prediction( 
            [Nitrogen, Phosphorus, potassium, Temperature , Humidity, Ph , Rainfall]) 
        
    st.success(diagnosis) 
  
if __name__ == '__main__': 
    main()
