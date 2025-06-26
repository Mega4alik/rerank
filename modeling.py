import os
import torch
from torch import nn
from typing import Any, Dict, List, Optional, Union
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModel

class MyModel(nn.Module):
	def __init__(self, model_id):
		super().__init__()
		self.embedding_dim = 1024 #jina emb dim
		self.bert_hidden_dim = 768 #bert emb dim: 768-base, 1024-large
		self.llm_model = AutoModel.from_pretrained(model_id)
		self.ln1 = nn.LayerNorm(self.embedding_dim)
		self.fc1 = nn.Linear(self.embedding_dim, self.bert_hidden_dim)
		self.fc2 = nn.Linear(self.bert_hidden_dim, 1)
		self.demb = nn.Embedding(512, self.bert_hidden_dim) #doc_embeddings

	def get_doc_ids(self, position_ids):
		B, T = position_ids.shape
		doc_ids = torch.zeros_like(position_ids, device=position_ids.device)
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
