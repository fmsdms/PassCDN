import streamlit as st



hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stTextInput label{font: normal 1rem courier !important; }
            footer:after {
                content:'Powered By Deadline'; 
                visibility: visible;
                display: block;
                position: relative;
                padding: 5px;
                top: 2px;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
def app():
    st.title("About Me")
    st.write('-------------------------')
    st.write('Software Engineering Bachelors Computer Science of SiChuan Univ.')
    st.write('Cyber Security Master College of Cybersecurity of SiChuan Univ.')
    st.write('Will be on boarding Intel')
    st.markdown('Github: [@goshorting](https://github.com/goshorting?tab=repositories)')
    st.write('Email: gs@goshorting.com')
