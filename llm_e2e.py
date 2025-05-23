# venv asr3.8 - US1
# t5 end to end solution for hotpotqa. token embeddings replaced with jinai embedding -- in progress 
import numpy as np
import json
import os
import random
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
#from torch.utils.data import Dataset, DataLoader
from datasets import Dataset, load_from_disk, concatenate_datasets
import evaluate
from typing import Any, Dict, List, Optional, Union
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration
from utils import file_get_contents, file_put_contents, pickle_load, pickle_save, cosine_similarity
from data_loader import multihop_qa_prepare_data, msmarco_prepare_data, financebench_prepare_data, hotpotqa_prepare_data
from rerank_bert import make_embeddings, hashf

def dataset_to_dict(dataset): #4 columns (+answer)
	d = {}
	for (question, chunks_list, labels_list, answer) in dataset:
		for o in [ ("question", question), ("answer", answer), ("chunks_list", chunks_list), ("labels_list", labels_list), ]:
			k, v = o[0], o[1]
			if k not in d: d[k] = []
			d[k].append(v)
	return d


class DataCollator:
	def __call__(self, features) -> Dict[str, torch.Tensor]:
		batch = {"input_values": [], "position_ids":[]}
		answers = [x["answer"] for x in features]
		labels = llm_tokenizer(answers, padding=True, return_tensors="pt").input_ids
		labels[labels == 0] = -100 # Mask out padded tokens

		for x in features:
			question, chunks_list = x["question"], x["chunks_list"]
			question_emb, chunks_emb = emb_cache[hashf(question)], []
			for chunks in chunks_list:
				chunks_emb.append( [emb_cache[hashf(chunk)] for chunk in chunks] )
			random.shuffle(chunks_emb)

			input_values, position_ids = [question_emb], [0] #input value: list of embeddings(list), labels: list of 0/1			
			for i, embs in enumerate(chunks_emb):
				for idx, emb in enumerate(embs):
					input_values.append(emb)
					position_ids.append(idx+1)

			input_values, position_ids = torch.tensor(input_values), torch.tensor(position_ids, dtype=torch.long)
			batch["input_values"].append(input_values)
			batch["position_ids"].append(position_ids)
		
		batch["input_values"] = pad_sequence(batch["input_values"], batch_first=True, padding_value=0) #B,S,C		
		batch["position_ids"] = pad_sequence(batch["position_ids"], batch_first=True, padding_value=0)
		batch["labels"] = labels
		#print("batch shapes:", batch["input_values"].shape, "\n\n", batch["labels"].shape, batch["labels"])
		return batch



class MyModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.embedding_dim = 1024 #jina emb dim
		self.hidden_dim = 512 #512-small, 768-base, 1024-large
		self.llm_model = llm_model		
		self.ln1 = nn.LayerNorm(self.embedding_dim)
		self.fc1 = nn.Linear(self.embedding_dim, self.hidden_dim)
		self.fc2 = nn.Linear(self.hidden_dim, 1)
		self.demb = nn.Embedding(512, self.hidden_dim) #doc_embeddings

	def get_doc_ids(self, position_ids):
		B, T = position_ids.shape
		doc_ids = torch.zeros_like(position_ids, device=device)
		for b in range(B):
			doc_id = 0
			for t in range(T):
				pos = position_ids[b, t].item()
				if pos == 0:
					doc_ids[b, t] = 0  # question token
				elif pos == 1 and t > 0:
					doc_id += 1
					doc_ids[b, t] = doc_id
				else:
					doc_ids[b, t] = doc_id
		#print("pos, doc ids:", position_ids[0], doc_ids[0])
		return doc_ids

	def trans(self, a, position_ids):
		B,T,C = a.size()
		doc_emb = self.demb( self.get_doc_ids(position_ids) )
		x = self.fc1(self.ln1(a))
		x = x + doc_emb
		return x


	def forward(self,
		input_values: Optional[torch.Tensor],
		attention_mask: Optional[torch.Tensor] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		labels: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.Tensor] = None
	):
		out = self.llm_model(inputs_embeds=self.trans(input_values, position_ids), labels=labels)
		return out

	def generate(self, x, position_ids):
		pred = self.llm_model.generate(inputs_embeds=self.trans(x, position_ids), max_new_tokens=20, do_sample=True, num_beams=5, temperature=0.3)
		return pred

	def _load_from_checkpoint(self, load_directory):
		load_path = os.path.join(load_directory, 'state_dict.pt')
		checkpoint = torch.load(load_path)
		self.ln1.load_state_dict(checkpoint['ln1_state_dict'])
		self.fc1.load_state_dict(checkpoint['fc1_state_dict'])
		self.fc2.load_state_dict(checkpoint['fc2_state_dict'])
		self.llm_model.load_state_dict(checkpoint['llm_state_dict'])
		self.demb.load_state_dict(checkpoint['demb_state_dict'])


class OwnTrainer(Trainer):
	def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
		preds, lables = None, None
		eval_dataloader = self.get_eval_dataloader(eval_dataset)
		for step, inputs in enumerate(eval_dataloader):
			with torch.no_grad():
				pred = self.model.generate(inputs['input_values'], inputs['position_ids']) #B,S
			return compute_metrics({"predictions":pred, "labels":inputs['labels']})	
			

	def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False): #called from Trainer._save_checkpoint	
		save_directory, model = output_dir, self.model
		os.makedirs(save_directory, exist_ok=True)
		save_path = os.path.join(save_directory, 'state_dict.pt')
		torch.save({
			'demb_state_dict': model.demb.state_dict(),
			'ln1_state_dict': model.ln1.state_dict(),
			'fc1_state_dict': model.fc1.state_dict(),
			'fc2_state_dict': model.fc2.state_dict(),
			'llm_state_dict': model.llm_model.state_dict()
		}, save_path)

	def _load_optimizer_and_scheduler(self, checkpoint):
		print("OPTIMIZER loading on train()!\n\n")
		super()._load_optimizer_and_scheduler(checkpoint)
	
	def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
		self.model._load_from_checkpoint(resume_from_checkpoint)		
		return self.model


def compute_metrics(x):
	pred_str = llm_tokenizer.batch_decode(x["predictions"], skip_special_tokens=True)
	label_ids = x["labels"]
	label_ids[label_ids == -100] = llm_tokenizer.pad_token_id
	label_str = llm_tokenizer.batch_decode(label_ids, skip_special_tokens=True)
	wer = wer_metric.compute(predictions=pred_str, references=label_str)
	print(wer, len(pred_str))
	for i in range(3 if mode==1 else len(pred_str)): print(label_str[i], "-- p:", pred_str[i])
	return {"eval_accuracy": wer}


###################### __main__ ###########################
gpu, device = True, torch.device("cuda")
llm_tokenizer = T5Tokenizer.from_pretrained("t5-small")
llm_model = T5ForConditionalGeneration.from_pretrained("t5-small")
wer_metric = evaluate.load("wer")
mymodel = None

mode = 2 #1-train, 2-test, 3-inference
emb_cache = {}

if mode==1 and 1==1:
	datasets = [load_from_disk(f"./temp/dataset_{dname}") for dname in ["hotpotqa_40k"]]
	mydataset = concatenate_datasets(datasets)
	emb_cache = make_embeddings(mode, None)
else:
	#prepare data
	if mode==1: #train
		dataset = hotpotqa_prepare_data(1) #40k
	else: #test		
		dataset = hotpotqa_prepare_data(2)
	emb_cache = make_embeddings(mode, dataset, append=True)
	d = dataset_to_dict(dataset)
	del dataset
	mydataset = Dataset.from_dict(d)
	del d
	#mydataset.save_to_disk("./temp/dataset_hotpotqa_40k")
	#exit()
	#./endOf prepare data

mydataset = mydataset.train_test_split(test_size=0.005 if mode==1 else 0.5, seed=42) #0.01 | 0.5
train_dataset, val_dataset = mydataset["train"], mydataset["test"]
#endOf prepare data

mymodel = MyModel()
bce_loss = nn.BCELoss()
data_collator = DataCollator()
training_args = TrainingArguments(
	output_dir="./model_temp/",
	#group_by_length=True, length_column_name="len",
	per_device_train_batch_size=16, #16 - bert-base US1,
	gradient_accumulation_steps=1, #update each 2 * batch_size
	fp16=False,
	evaluation_strategy="steps",
	num_train_epochs=200,
	logging_steps=50,
	save_steps=500,
	eval_steps=500,
	per_device_eval_batch_size=(100 if mode==2 else 23),
	learning_rate=1e-5,
	dataloader_num_workers=4,
	weight_decay=0.005,
	warmup_steps=1000,
	save_total_limit=4,
	ignore_data_skip=True,
	remove_unused_columns=False,
	#label_names=["labels"], #attempt to solve eval problem
	metric_for_best_model="eval_accuracy",
	greater_is_better=False,
	#load_best_model_at_end=True,
)
print("\n\nstarting training", len(train_dataset), len(val_dataset))
trainer = OwnTrainer(
	model=mymodel,
	data_collator=data_collator,
	args=training_args,
	compute_metrics=compute_metrics,
	train_dataset=train_dataset,
	eval_dataset=val_dataset,
	#tokenizer=processor.feature_extractor,
)
if mode==1:
	trainer.train("./model_temp/checkpoint-211000")
elif mode==2: #test
	trainer._load_from_checkpoint("./model_temp/checkpoint-343500")
	trainer.evaluate()

