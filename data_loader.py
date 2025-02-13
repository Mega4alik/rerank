import random
import json
from datasets import load_dataset
from utils import file_get_contents, file_put_contents, pickle_load, pickle_save

G_CHUNK_SIZE = 200

def split_article_with_fact(article_text, fact_text, chunk_size=200):	
	def _chunk_words(words, chunk_size):	    
		result = []
		start = 0
		while start < len(words):
			result.append(" ".join(words[start:start + chunk_size]))
			start += chunk_size
		return result    

	# Helper function to locate fact in article (naive approach)
	def find_fact_indices(words, fact):
		for start_index in range(len(words) - len(fact) + 1):
			if words[start_index:start_index + len(fact)] == fact:
				return start_index, start_index + len(fact)
		return None, None

	# Split both article and fact into lists of words
	article_words = article_text.split()
	if fact_text is None: return _chunk_words(article_words, chunk_size)
	fact_words = fact_text.split()
	fact_len = len(fact_words)

	fact_start, fact_end = find_fact_indices(article_words, fact_words)

	# If the fact isn’t found, just chunk normally
	if fact_start is None:
		return _chunk_words(article_words, chunk_size)

	# Otherwise, chunk in three parts:
	# 1. Everything before the fact
	# 2. A single chunk that contains the entire fact
	# 3. Everything after the fact

	chunks = []

	# Chunk everything before the fact
	start = 0
	while start + chunk_size <= fact_start:
		chunks.append(" ".join(article_words[start:start + chunk_size]))
		start += chunk_size

	# Place the fact-containing chunk (could be bigger than chunk_size if needed)
	# The chunk starts where we left off above and ends at least at fact_end
	# (You can extend chunk_end further if you prefer balancing chunk sizes)
	chunk_end = max(fact_end, start + chunk_size)
	chunks.append(" ".join(article_words[start:chunk_end]))

	# Chunk everything after the fact
	start = chunk_end
	while start < len(article_words):
		chunks.append(" ".join(article_words[start:start + chunk_size]))
		start += chunk_size

	return chunks


def multihop_qa_prepare_data():
	dataset = []
	qas = json.loads(file_get_contents("./data/MultiHopRAG.json"))
	articles = json.loads(file_get_contents("./data/corpus.json"))	
	#print(len(qas), len(articles), qas[0], articles[0])
	for step, qa in enumerate(qas[:]):
		if len(qa["evidence_list"])==0: continue
		question = qa["query"]
		chunks_list, labels_list, urls, chunks_n = [], [], [], 0
		#add evidence chunks(fact, url)
		for ev in qa["evidence_list"]:
			urls.append(ev["url"])
			fact = ev["fact"]
			article_text = next((x for x in articles if x["url"] == ev["url"]), None)['body']
			chunks, labels = split_article_with_fact(article_text, fact, chunk_size=G_CHUNK_SIZE), []			
			chunks_list.append(chunks)
			chunks_n+=len(chunks)
			# double check
			found = False
			for chunk in chunks:
				if fact in chunk:
					found = True
					labels.append(1)
				else:
					labels.append(0)
			labels_list.append(labels)
			assert found == True
		#./endOf add evidence chunks(fact, url)
		#add more chunks
		articles2 = [x for x in articles if x["url"] not in urls]
		random.shuffle(articles2)
		for article in articles2:
			chunks = split_article_with_fact(article["body"], None, chunk_size=G_CHUNK_SIZE)
			if chunks_n + len(chunks) > 511: break
			chunks_list.append(chunks)
			labels_list.append([0] * len(chunks))
			chunks_n+=len(chunks)			
		#./endOf add more chunks

		dataset.append( (question, chunks_list, labels_list) )
	return dataset


def msmarco_prepare_data():
	dataset = load_dataset("microsoft/ms_marco", 'v1.1', split="test") #features: ['answers', 'passages', 'query', 'query_id', 'query_type', 'wellFormedAnswers'],  num_rows: 9650	
	datasize, dataset2 = 2000, [] #len(dataset["passages"])
	dataset = dataset.to_pandas()
	for i in range(datasize):
		chunks_list, labels_list, chunks_n = [], [], 0
		question, passages = dataset["query"][i], dataset["passages"][i]
		labels, chunks = passages["is_selected"], passages["passage_text"] #passa_text is array of passages with length of ~40-150 words
		chunks_list.append(chunks)
		labels_list.append(labels)
		chunks_n+=len(chunks)

		#add more chunks
		indexes = list(range(datasize))
		indexes.remove(i)
		random.shuffle(indexes)
		for j in indexes:
			chunks = dataset["passages"][j]["passage_text"]			
			if chunks_n + len(chunks) > 511: break
			chunks_list.append(chunks)
			labels_list.append([0] * len(chunks))
			chunks_n+=len(chunks)
		#./endOf add more chunks
		
		dataset2.append((question, chunks_list, labels_list))
		#print("msmarco:", i, len(chunks_list), len(labels_list))

	return dataset2



def financebench_prepare_data():
	arr = json.loads(file_get_contents("./data/financebench.json"))	
	datasize, dataset, chunks_list_all = len(arr), [], []
	for i in range(511): chunks_list_all.append( [ arr[i]['content'] ] )

	for x in arr[:100]: #idx, content, keywords, questions
		if not "questions" in x: continue
		for question in x["questions"]:
			chunks_list = chunks_list_all.copy()
			labels_list =  [[0]] * len(chunks_list)
			labels_list[x["idx"]] = [1]
			dataset.append( (question, chunks_list, labels_list) )
	return dataset


			

def analyze(dataset):
	s,n = 0,0
	for (question, chunks_list, labels_list) in dataset[:10]:
		print(question)
		for chunks in chunks_list:
			for chunk in chunks:
				#print(chunk)				
				s+=len(chunk)
				n+=1

	print(s / n, len(dataset))	


if __name__=="__main__":
	#dataset = msmarco_prepare_data()
	#dataset = multihop_qa_prepare_data() 
	dataset = financebench_prepare_data()
	analyze(dataset)
	

