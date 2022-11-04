
import streamlit as st
import requests
import json
def setup():
    st.set_page_config(
        page_title="titles", 
        page_icon=":shark:",
        layout="wide",
        initial_sidebar_state="expanded", 
        menu_items={       
            'About': ""}
    )

def sidebar():
    config_dict = {}
    sbform = st.sidebar.form("Configs")
    sbform.write("Configs")

    config_dict["top_p"]       = sbform.slider('Top-P:',min_value=0.0,max_value=1.0,value=0.9,step=0.1)
    config_dict["model_path"] = sbform.selectbox('Model_id',['Model1','Model2'])
    config_dict["num_beams"] = sbform.slider('Num_Beams:',min_value=1,max_value=5,value=5,step=1)
    clicked =sbform.form_submit_button("Submit")    
    return config_dict, example_id, clicked

def load_dataset(data_file:str):
    with open(data_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        samples = [json.loads(line) for line in lines]
    return samples

def load_example(example_id:int): 
    sample = samples[example_id]
    text = dict_text(example)
    st.markdown(text)
    # input_context = form.text_input('label',value=text,placeholder=text)
    clicked=st.button("submit")  
    return example,clicked


setup()

st.header("Demo")

config, submited = sidebar()
if submited:
    backend_url="http://127.0.0.1:8000/config"
    response = requests.post(backend_url, json=config, verify=False)

data_file = "train.json"
samples = load_dataset(data_file)
example, clicked = load_example(example_id)
if clicked:
    with st.spinner('loading...'):
        backend_url="http://127.0.0.1:8000/predict"
        response = requests.post(backend_url, json=example, verify=False) 
        pred = response.json().get("context")
        st.markdown(f"{pred}")

