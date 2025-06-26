# curl -X POST https://anuarsh--rerank1-web-inference.modal.run -H "Content-Type: application/json" -d '{"question": "как получить справку безработного?"}'

import modal
import json, sys
import pickle
from typing import Dict
import torch
from transformers import AutoTokenizer, AutoModel
from modeling import MyModel

def pickle_load(path):
    with open(path, "rb") as file: 
        obj = pickle.load(file)
        return obj

#====================
app = modal.App("rerank1")

image = modal.Image.debian_slim().pip_install(
	"torch", "transformers", "accelerate", "fastapi[standard]"
)

#image = image.add_local_file("./modeling.py", remote_path="/modeling.py")
image = image.add_local_python_source("modeling")
image = image.add_local_dir("./model_temp/checkpoint-134000", remote_path="/model_checkpoint")
image = image.add_local_file("./temp/data.pkl", remote_path="/data.pkl")


@app.cls(gpu="any", image=image, timeout=600) #gpu="any" | A100
class ModelRunner:
	@modal.enter()
	def setup(self):
		#sys.path.append("/")
		from modeling import MyModel
		model_id = "google-bert/bert-base-uncased"		
		self.tokenizer = AutoTokenizer.from_pretrained(model_id)
		self.model = MyModel(model_id)
		self.model._load_from_checkpoint("/model_checkpoint")
		self.model.eval()		
		self.embedding_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3") #, trust_remote_code=True
		self.embedding_model.eval()
		#chunks_list, input_values, position_ids = pickle_load("/data.pkl")

	@modal.method()
	def run(self, question: str):
		chunks_list, input_values, position_ids = pickle_load("/data.pkl")
		question_emb = self.embedding_model.encode([question], task="retrieval.query")[0]
		input_values[0] = question_emb
		input_values, position_ids = torch.tensor([input_values]), torch.tensor([position_ids], dtype=torch.long) #B=1,S,1024, B,S
		batch_preds = self.model.generate(input_values, position_ids)
		probs = batch_preds[0]
		top_indices = torch.topk(probs, 15).indices.detach().tolist()
		chunks_all = ["NONE"]
		for chunks in chunks_list: chunks_all.extend(chunks)
		a = []
		for idx in top_indices: a.append( (chunks_all[idx].lstrip('\n'), probs[idx].item()) )
		return a


@app.function(image=image)
@modal.web_endpoint(method="POST")
def web_inference(req: Dict):
	if "question" in req:
		question = req["question"]
		return ModelRunner().run.remote(question)
   