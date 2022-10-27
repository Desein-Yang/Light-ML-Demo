import uvicorn
from fastapi import FastAPI, Depends
from pydantic import BaseModel    #fastapi的一个依赖,需要从pydantic中引入
from transformers import AutoTokenizer, BartForConditionalGeneration
from typing import List, Tuple

class Config(BaseModel): # generate 参数/device
    model_path: str
    num_beams: int
    top_p: float

class QOutput(BaseModel): 
    question: str

class QGInput(BaseModel): 
    context: str
    bos_span: int
    eos_span: int
    answer: str

class EXOutput(BaseModel):
    ans: List[str]
    ans_span: List[List[int]]

# pipelines
class QGModel:
    def __init__(self, sptoken: str, tokenizer_type: str):
        self.sptoken = sptoken
        self.tokenizer_type=tokenizer_type
        self.device="cuda:2"
        self.top_p = 0.9
        self.model_path = "/cognitive_comp/yangqi/model/Randeng-BART-139M-QG-Chinese"
        self.num_beams=5

    def load_model(self):
        self.model = BartForConditionalGeneration.from_pretrained(self.model_path).to(self.device)

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,use_fast=False,additional_special_tokens=[self.sptoken])

    def setup_config(self, config:Config)-> None:
        self.top_p = config.top_p
        self.model_path = "/cognitive_comp/yangqi/model/" + config.model_path
        self.num_beams = config.num_beams
        print("config")

    def data_process(self, inputs: QGInput) -> dict:
        # masked_ctx = self.replace_all(inputs.context, inputs.answer, self.sptoken)
        masked_ctx = self.replace_span(
            inputs.context, 
            inputs.bos_span,inputs.eos_span, 
            self.sptoken
        )
        
        pre_prompt, mid_prompt = "知识:", "回答:"
        context = self.truncate_sequence(masked_ctx, 416-len(pre_prompt)-1)
        answer = self.truncate_sequence(inputs.answer, 32 - len(mid_prompt)-1)

        x_trunc = f'{pre_prompt}{context}{mid_prompt}{answer}'
        if self.tokenizer_type == "bart":
            x_trunc = self.tokenizer.bos_token + x_trunc + self.tokenizer.eos_token

        encoder_input = self.tokenizer.encode_plus(
            x_trunc,
            max_length=448,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )

        return encoder_input

    @staticmethod
    def replace_all(source:str, target:str, sptoken:str) -> str:
        return source.replace(target, sptoken)

    @staticmethod
    def replace_span(source:str, bos_span:int, eos_span:int, sptoken:str) -> str:
        return source[:bos_span] + sptoken + source[eos_span:]

    @staticmethod
    def truncate_sequence(doc: str, max_len: int):
        return doc if len(doc) < max_len else doc[:max_len]

    def predict(self, inputs: QGInput) -> dict:
        # if not self.model or not self.tokenizer:
        #     raise RuntimeError("模型加载失败")
        # input_dicts = process(input)
        # prediction = self.model.predict()
        encode_inputs = self.data_process(inputs)
        out = self.model.generate(                
            input_ids=encode_inputs['input_ids'].to(self.device),
            attention_mask=encode_inputs['attention_mask'].to(self.device),
            do_sample=True,
            num_beams=self.num_beams,
            max_length=64,
            top_p =self.top_p,
        )
        pred = self.tokenizer.batch_decode(out,clean_up_tokenization_spaces=True, skip_special_tokens=True)[0]
        pred = pred.split(":")[1]
        
        return QOutput(question=pred)

class EXModel:
    def load_model(self):
        pass

    def predict(self,inputs:QGInput) -> EXOutput:
        return output

app = FastAPI()
qg_model = QGModel(sptoken="[ANS]",tokenizer_type="bert")
# ex_model = EXModel()

@app.on_event("startup") # 开始时加载模型
async def startup():
    qg_model.load_model()
    qg_model.load_tokenizer()


@app.post("/qg_gen") # 注入预测路由，调用预测函数
def gen_question(output:QOutput = Depends(qg_model.predict)) -> QOutput:
    return output

@app.post("/config")
def read_config(inputs:Config) -> None:
    qg_model.setup_config(inputs)


# @app.post("/uniex")
# def extract_ans(output:EXOutput = Depends(ex_model.predict)) -> EXOutput:
#     return output



# if __name__ == "__main__":
#     uvicorn.run("backend:app",host="0.0.0.0", port=8000)