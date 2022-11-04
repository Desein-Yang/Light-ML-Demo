import uvicorn
from fastapi import FastAPI, Depends
from pydantic import BaseModel    # this is to verify params
from transformers import AutoTokenizer, GPT2LMHeadModel
from typing import List, Tuple

# data class
class Config(BaseModel): # config from frontend
    model_path: str
    num_beams: int
    top_p: float

class Output(BaseModel): 
    output: str

class Input(BaseModel): 
    input:str


# inference pipelines
class Model:
    def __init__(self, tokenizer_type: str):
        # config hyperparameter
        self.tokenizer_type=tokenizer_type
        self.device="cuda:2"
        self.top_p = 0.9
        self.model_path = "/model/gpt2"
        self.num_beams=5

    def load_model(self):
        self.model = GPT2LMHeadModel.from_pretrained(self.model_path).to(self.device)

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,use_fast=False,additional_special_tokens=[self.sptoken])

    def setup_config(self, config:Config)-> None:
        self.top_p = config.top_p
        self.model_path = config.model_path
        self.num_beams = config.num_beams
        print("config")

    def data_process(self, inputs: Input) -> dict:
        # edit your process sh 
        pre_prompt, mid_prompt = "AAA", "BBB"
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

    def predict(self, inputs: Input) -> Output:
        """predict pipeline of your model"""
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
        
        return Output(question=pred)

# initialize
app = FastAPI()
model = Model(tokenizer_type="gpt")

# load model when page loading
@app.on_event("startup") 
async def startup():
    model.load_model()
    model.load_tokenizer()


# inject the function of api into endpoint
@app.post("/predict") 
def run(output:Output = Depends(model.predict)) -> Output:
    return output

@app.post("/config") 
def read_config(inputs:Config) -> None:
    model.setup_config(inputs)





# if __name__ == "__main__":
#     uvicorn.run("backend:app",host="0.0.0.0", port=8000)
