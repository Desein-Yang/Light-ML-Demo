
import streamlit as st
import requests
import json
def setup():
    st.set_page_config(
        page_title="知识对话-QAG", 
        page_icon=":shark:",
        layout="wide",
        initial_sidebar_state="expanded", 
        menu_items={       
            'About': "https://github.com/IDEA-CCNL/Fengshenbang-LM/"}
    )

def sidebar():
    config_dict = {}
    sbform = st.sidebar.form("参数设置")
    sbform.write("参数设置")

    config_dict["top_p"]       = sbform.slider('Top-P:',min_value=0.0,max_value=1.0,value=0.9,step=0.1)
    config_dict["model_path"] = sbform.selectbox('Model_id',['Randeng-BART-139M-QG-Chinese','FNLP-BART-139M-QG-Chinese'])
    config_dict["num_beams"] = sbform.slider('Num_Beams:',min_value=1,max_value=5,value=5,step=1)
    example_id = sbform.number_input('Example_id',min_value=0, max_value=10836,step=1)
    clicked =sbform.form_submit_button("Submit")    
    return config_dict, example_id, clicked

def load_dataset(data_file:str):
    with open(data_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        samples = [json.loads(line) for line in lines]
    return samples

def load_example(example_id:int): 
    sample = samples[example_id]
    example = {
        "context": sample["context"],
        "answer": sample["answer"][0],
        "bos_span":sample["ans_span"][0][0],
        "eos_span":sample["ans_span"][0][1]
    }
    text = dict_text(example)
    st.markdown("**输入样例**")
    st.markdown(text)
    # input_context = form.text_input('知识',value=text,placeholder=text)
    clicked=st.button("输入")  
    return example,clicked

def inputblank():
    form = st.form("文本输入")
    if config["example_id"] is not "None": # 加载本文
        # item = prepared_examples[example_id]
        example = {
            "context":"这里是知识",
            "answer": "知识",
            "ans_span":[3,5]
        }
        text = dict_text(example)
        input_context = form.text_input('知识',value=text,placeholder=text)
    else: # 输入文本
        input_context = form.text_input('知识',value="")
        example = text_dict(input_context)

    form.form_submit_button("提问")   
    return example

def text_dict(text: str) -> dict: # parse to triple dict
    if len(text) == 0:
        raise RuntimeError("请输入知识")
    if ("【" not in text) or ("】" not in text):
        raise RuntimeError("请在输入知识中用【】标出答案")
    
    bos_span = text.find("【")
    eos_span = text.find("】")
    answer = text[bos_span+1:eos_span]
    context = text[:bos_span]+answer+text[eos_span+1:]
    return {
        "context" :context,
        "answer"  :answer,
        "bos_span":bos_span,
        "eos_span":eos_span
    }
def dict_text(example:dict)-> str:
    bos, eos = example["bos_span"],example["eos_span"]
    text = example["context"][:bos] + "【" + example["answer"] + "】"+example["context"][eos:]
    return text

setup()

st.header("Demo for knowledge-based QAG")

data_file = "/cognitive_comp/yangqi/project/data/ChineseSQuAD/zen2/dev_v2_pairs.json"
samples = load_dataset(data_file)

config, example_id, submited = sidebar()
if submited:
    backend_url="http://127.0.0.1:8000/config"
    response = requests.post(backend_url, json=config, verify=False)

example, clicked = load_example(example_id)
if clicked:
    with st.spinner('正在提问...'):
        backend_url="http://127.0.0.1:8000/qg_gen"
        response = requests.post(backend_url, json=example, verify=False)
        json_response = response.json()
        pred = json_response.get("question")
        st.markdown("**问题**")
        st.markdown(f"{pred}")

    st.markdown("**回答**")
