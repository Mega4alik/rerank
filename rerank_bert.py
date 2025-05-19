# RAG Rerank with BERT -- use chunk embeddings(ex, jina) instead of tokens
import numpy as np
import json
import os
import random
#import multiprocessing
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
#from torch.utils.data import Dataset, DataLoader
from datasets import Dataset, load_from_disk, concatenate_datasets
from typing import Any, Dict, List, Optional, Union
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModel
from utils import file_get_contents, file_put_contents, pickle_load, pickle_save, cosine_similarity
from data_loader import multihop_qa_prepare_data, msmarco_prepare_data, financebench_prepare_data, hotpotqa_prepare_data

COHERE_EVAL = False #whether we are evaluationg cohere rerank now
RAW_EMBEDDINGS_EVAL = False

def hashf(st):
	if len(st) < 120: return st
	else: return st[:50] + "..." + st[-50:]


@torch.inference_mode()
def make_embeddings(mode, dataset, append=False):
	global emb_cache
	path = "./temp/test_rerank_cache.pkl" if mode==2 else "./temp/rerank_cache.pkl"
	if os.path.exists(path):
		emb_cache = pickle_load(path)
		print("loaded emb_cache len:", len(emb_cache))
		if not append: return emb_cache
	
	embedding_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
	embedding_model.eval()
	embedding_model.cuda()
	for step, (question, chunks_list, labels_list) in enumerate(dataset):
		print(f"\rEmb: {step}/2555", end="", flush=True)
		h = hashf(question)
		if h not in emb_cache: emb_cache[h] = embedding_model.encode([question], task="retrieval.query")[0]
		for chunks in chunks_list:			
			for chunk in chunks:
				h = hashf(chunk)
				if h not in emb_cache: emb_cache[h] = embedding_model.encode([chunk], task="retrieval.passage")[0]
	
	pickle_save(path, emb_cache)
	return emb_cache


def dataset_to_dict(dataset):
	d = {}
	for (question, chunks_list, labels_list) in dataset:
		for o in [ ("question", question), ("chunks_list", chunks_list), ("labels_list", labels_list), ]:
			k, v = o[0], o[1]
			if k not in d: d[k] = []
			d[k].append(v)
	return d


class DataCollator:
	def shuffle_lists(self, a, b):
		combined = list(zip(a, b))  # Pair corresponding elements
		random.shuffle(combined)  # Shuffle the pairs
		a_shuffled, b_shuffled = zip(*combined)  # Unzip after shuffling
		return list(a_shuffled), list(b_shuffled)

	def __call__(self, features) -> Dict[str, torch.Tensor]:
		batch = {"input_values": [], "labels":[], "position_ids":[]}
		if COHERE_EVAL: batch = batch | {"chunks_list":[], "labels_list":[], "question":[]}

		for x in features:
			question, labels_list, chunks_list = x["question"], x["labels_list"], x["chunks_list"]
			question_emb, chunks_emb = emb_cache[hashf(question)], []
			for chunks in chunks_list:
				chunks_emb.append( [emb_cache[hashf(chunk)] for chunk in chunks] )
			if not COHERE_EVAL: chunks_emb, labels_list = self.shuffle_lists(chunks_emb, labels_list)
			
			input_values, labels, position_ids = [question_emb], [0], [0] #input value: list of embeddings(list), labels: list of 0/1
			for i, embs in enumerate(chunks_emb):
				for idx, emb in enumerate(embs):
					input_values.append(emb)
					position_ids.append(idx+1)
				labels += labels_list[i]

			input_values, labels, position_ids = torch.tensor(input_values), torch.tensor(labels), torch.tensor(position_ids, dtype=torch.long)
			batch["input_values"].append(input_values)
			batch["labels"].append(labels)
			batch["position_ids"].append(position_ids)
			if COHERE_EVAL:
				batch["chunks_list"].append(chunks_list) #temp for cohere
				batch["labels_list"].append(labels_list) #temp for cohere
				batch["question"].append(question) #temp for cohere

		batch["input_values"] = pad_sequence(batch["input_values"], batch_first=True, padding_value=0) #B,S,C
		batch["labels"] = pad_sequence(batch["labels"], batch_first=True, padding_value=0) #B,S -100
		batch["position_ids"] = pad_sequence(batch["position_ids"], batch_first=True, padding_value=0)
		#print("batch shapes:", batch["input_values"].shape, batch["labels"].shape)
		return batch


class MyModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.embedding_dim = 1024 #jina emb dim
		self.bert_hidden_dim = 768 #bert emb dim: 768-base, 1024-large
		self.llm_model = llm_model		
		self.ln1 = nn.LayerNorm(self.embedding_dim)
		self.fc1 = nn.Linear(self.embedding_dim, self.bert_hidden_dim)
		self.fc2 = nn.Linear(self.bert_hidden_dim, 1)
		self.demb = nn.Embedding(512, self.bert_hidden_dim) #doc_embeddings

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
		#position_ids = torch.arange(0, T, dtype=torch.long, device=device) #[0,1,2,..T]
		pos_emb = self.llm_model.embeddings.position_embeddings(position_ids)
		doc_emb = self.demb( self.get_doc_ids(position_ids) )

		#segment_emb 0,1
		#token_type_ids = np.ones((B,T))
		#token_type_ids[:,0] = 0
		#token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=device) #0 or 1
		#segment_emb = self.llm_model.embeddings.token_type_embeddings(token_type_ids)
		#print(a.shape, segment_emb.shape, a[0][0], segment_emb[0], "\na:", a.mean(dim=(1,2)),  a.var(dim=(1,2)), "\npos:",  segment_emb.mean(), segment_emb.var())

		x = self.fc1(self.ln1(a))
		x = x + pos_emb + doc_emb
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
		out = self.llm_model(inputs_embeds=self.trans(input_values, position_ids), output_hidden_states=True)
		pred = self.fc2(out.hidden_states[-1]).squeeze(-1) #B, S
		pred = torch.sigmoid(pred) #0..1

		if labels is None: #inference
			return pred
		else:
			#print("forward pred, labels:", pred.shape, labels.shape)
			loss = bce_loss(pred, labels.float())
			return {"loss":loss}


	def generate(self, x, position_ids):
		pred = self.forward(input_values=x, position_ids=position_ids)
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
			if RAW_EMBEDDINGS_EVAL: return test_raw_embeddings(inputs["input_values"], inputs["labels"])
			if COHERE_EVAL: return cohere_rerank(inputs["question"], inputs["chunks_list"], inputs["labels_list"])
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
	batch_preds, batch_labels = x["predictions"], x["labels"]
	correct, n = 0, 0	
	for i in range(len(batch_preds)):
		probs, labels = batch_preds[i], batch_labels[i]		
		top_indices = torch.topk(probs, 10).indices.detach().tolist()
		labels_ones = torch.nonzero(labels == 1).squeeze(-1).detach().tolist()
		for lidx in labels_ones:
			if lidx in top_indices: correct+=1
			n+=1
		print(labels_ones, top_indices, correct, n)
	return {"eval_accuracy": (correct / n) }


#========== raw embeddings =======================
def test_raw_embeddings(batch_input_values, batch_labels):
	correct, n = 0, 0
	for i in range(len(batch_labels)):
		input_values, labels = batch_input_values[i], batch_labels[i]
		labels_ones = torch.nonzero(labels == 1).squeeze(-1).detach().tolist()
		qe, a = input_values[0].detach().cpu(), []
		for j in range(1, len(input_values)): a.append((j, cosine_similarity(qe, input_values[j].detach().cpu())))
		a = sorted(a, key=lambda x: x[1], reverse=True)
		top_indices = [x[0] for x in a[:10]]
		for lidx in labels_ones:
			if lidx in top_indices: correct+=1
			n+=1
		print(labels_ones, top_indices, correct, n)
#========== ./endOf raw embeddings =================

#==== cohere ========
if COHERE_EVAL:
	from config import COHERE_API_KEY
	import cohere
	co = cohere.ClientV2(COHERE_API_KEY)
	def cohere_rerank(b_question, b_chunks_list, b_labels_list):
		correct, n = 0, 0
		for i, chunks_list in enumerate(b_chunks_list):
			question, labels_list, documents, labels  = b_question[i], b_labels_list[i], [], []
			for chunks in chunks_list:
				for chunk in chunks: documents.append(chunk)
			for a in labels_list: labels+=a
			print("docs, labels lengths:", len(documents), len(labels))
			labels_ones = torch.nonzero(torch.tensor(labels) == 1).squeeze(-1).detach().tolist()
			documents_reranked = co.rerank(model="rerank-v3.5", query=question, documents=documents, top_n=10)
			top_indices = [r.index for r in documents_reranked.results]
			for lidx in labels_ones:
				if lidx in top_indices: correct+=1
				n+=1
			print(labels_ones, top_indices, correct, n)
		return {"eval_accuracy": (correct / n) }
#==== endOf cohere ========	


###################### __main__ ###########################
if __name__=="__main__":
	gpu, device = True, torch.device("cuda")
	llm_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")  #bert-large-uncased
	llm_model = AutoModel.from_pretrained("google-bert/bert-base-uncased")
	mymodel = None

	mode = 2 #1-train, 2-test, 3-inference

	emb_cache = {}
	if mode==1 and 1==1:
		datasets = [load_from_disk(f"./temp/dataset_{dname}") for dname in ["multihop", "msmarco", "hotpotqa_20k", "hotpotqa_20k_2"]] #"financebench",
		mydataset = concatenate_datasets(datasets)
		make_embeddings(mode, None)
	else:
		#prepare data
		if mode==1: #train
			#dataset = multihop_qa_prepare_data() #2.2k
			#dataset += msmarco_prepare_data(1) #2k
			#dataset += financebench_prepare_data("3M_2018_10K")
			#dataset += financebench_prepare_data("ADOBE_2015_10K")
			dataset = hotpotqa_prepare_data(1) #20k
		else: #test
			#dataset = financebench_prepare_data("AMAZON_2015_10K") # | msmarco_prepare_data(2)
			dataset = hotpotqa_prepare_data(2) #msmarco_prepare_data(2)
		make_embeddings(mode, dataset, append=False)
		d = dataset_to_dict(dataset)
		del dataset
		mydataset = Dataset.from_dict(d)
		del d
		#mydataset.save_to_disk("./temp/dataset_hotpotqa_20k_2")
		#exit()
		#./endOf prepare data

	mydataset = mydataset.train_test_split(test_size=0.01 if mode==1 else 0.5, seed=42) #0.01 | 0.5
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
	  num_train_epochs=100,
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
		trainer.train("./model_temp/checkpoint-91000")
	elif mode==2: #test
		trainer._load_from_checkpoint("./model_temp/checkpoint-134500")
		trainer.evaluate()

