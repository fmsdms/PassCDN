import streamlit as st
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus']=False
from matplotlib.font_manager import FontProperties


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stTextInput label{font: normal 1rem courier !important; }
            footer:after {
                content:'Powered by Deadline'; 
                visibility: visible;
                display: block;
                position: relative;
                padding: 5px;
                top: 2px;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

#app
import psm
import about

PAGES = {
    "PassCDN-PSM": psm,
    "About":about,
}
st.sidebar.title('Goshorting')
selection = st.sidebar.radio("Navigation", list(PAGES.keys()))
page = PAGES[selection]
page.app()
