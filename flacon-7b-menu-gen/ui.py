import streamlit as st 
import app

st.title("Restaurant Name & Menu Generator")

prompt = st.text_input('enter here')

if prompt:
    response = app.get_response(prompt)
    st.header(response['restaurant_name'].strip())
    menu_items = response['menu items'].split(",")
    st.write("*Menu Items*")
    for item in menu_items:
        st.write(item)