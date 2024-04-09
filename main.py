
import streamlit as st
import os
import pickle


# Load the model
model = pickle.load(open(os.path.abspath('Data/finalized_model.sav'), 'rb'))
st.set_page_config(layout="wide", page_title='Iris Prediction', page_icon='🌺')
def predict(sepal_length, sepal_width, petal_length, petal_width):
    result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    return result

def main():
    
    container=st.container(height=100,border=False)
    with container:
        st.title("Iris Flower Prediction",)   
    container2=st.container()
    with container2:
        col1, col2 = st.columns(2)
        with col1:
            sepal_length = st.number_input("Sepal Length",value=None, placeholder="Type the length of the Sepal")
            sepal_width = st.number_input("Sepal Width",value=None, placeholder="Type the width of the Sepal")
            petal_length = st.number_input("Petal Length",value=None, placeholder="Type the length of the Petal")
            petal_width = st.number_input("Petal Width",value=None, placeholder="Type the width of the Petal")
            if st.button("Submit"):
                    result = predict(sepal_length, sepal_width, petal_length, petal_width)
                    with col2:
                        if result[0]=='Iris-virginica':
                            st.image(os.path.abspath('Data/imgs/Iris-virginica.png'),width=400,caption='Iris-virginica')
                        elif result[0]=='Iris-setosa':
                            st.image(os.path.abspath('Data/imgs/Iris-setosa.png'),width=400,caption='Iris-setosa')
                        else:
                            st.image(os.path.abspath('Data/imgs/Iris-versicolor.png'),width=400,caption='Iris-versicolor')
                            
                        
                
                
                
            
        
        

if __name__ == '__main__':
    main()