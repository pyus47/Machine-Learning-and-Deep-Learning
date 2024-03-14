import streamlit as  st 
# adding title of your app 
st.title('My First testing for codanics course (6 months long )')
# adding simple text 
st.write('Here is the sample text') 
# user imput 
number = st.slider('pick a number ',0,100)
# printing the user selected number 
st.write(f'you selected the {number}')
# adding the hello button 
if st.button('Greetings'):
    st.write('Hi,Hello There')
else:
    st.write('Goodbye')
# adding a radio button 
    gener = st.radio('what is your favourite movie genre'
                     , ['Marvel','DC','Action'])
    # print the genre you selected 
st.write(f'you selected the follwing genre {gener}')
# adding a drop down list 
option = st.selectbox('How would you like to be contacted?',
                      ['Email','phone','Home phone'])
# add the drop down list on the left of the side bar 
option = st.sidebar.selectbox('how would you like to be contacted',
                              ['Email','Home Phone','Mobile Phone'])
# add your whatapp number taking text input 
st.sidebar.text_input('Enter Your Wathapps Number Here ')
# add the file upload option 
uploaded_file = st.sidebar.file_uploader('Choose a csv file', type='csv')

