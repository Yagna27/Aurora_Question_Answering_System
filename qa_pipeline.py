
import re
import json
import requests
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import CrossEncoder

def fetch_messages():
    url = "https://november7-730026606190.europe-west1.run.app/messages"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    #print(data)
    return data

data = fetch_messages()

df = pd.DataFrame(data['items'])
# make sure expected columns exist:
expected = ['id','user_id','user_name','timestamp','message']
for c in expected:
    if c not in df.columns:
        df[c] = None

df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['message']).reset_index(drop=True)
print("Rows:", len(df))
df.head()


def clean_text(text):
    text = str(text)
    text = text.strip()
    # keep punctuation for dates (commas, colons), remove weird control chars
    text = re.sub(r'[\r\n]+', ' ', text)
    return text

df['message_clean'] = df['message'].apply(clean_text)

# create a combined field that includes user name + message which we will embed
df['embed_text'] = df.apply(lambda r: f"{r['user_name']} : {r['message_clean']} (ts: {r['timestamp']})", axis=1)

# Optional: also create per-user aggregated doc (all messages of a user) to help find user-level context
user_docs = df.groupby('user_name')['message_clean'].agg(lambda msgs: " || ".join(msgs)).reset_index()
user_docs.columns = ['user_name', 'user_agg']
user_docs.head()

# Use a retrieval-optimized model
embed_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

# encode message-level entries
embeddings = embed_model.encode(df['embed_text'].tolist(), convert_to_numpy=True, show_progress_bar=True)

# normalize for cosine via inner product
faiss.normalize_L2(embeddings)

embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)
index.add(embeddings)
print("FAISS index size:", index.ntotal)

# also encode aggregated per-user docs (helps when we detect the user in question)
user_embeddings = embed_model.encode(user_docs['user_agg'].tolist(), convert_to_numpy=True)
faiss.normalize_L2(user_embeddings)
user_index = faiss.IndexFlatIP(user_embeddings.shape[1])
user_index.add(user_embeddings)

# Cross-encoder to re-rank top-k candidates (higher accuracy)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # small & effective

# QA model (you chose deepset/roberta-base-squad2)
qa_model_name = "deepset/roberta-base-squad2"
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
qa_pipe = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)

# build a list of user names (normalize)
user_names = df['user_name'].dropna().unique().tolist()
# create a simple lowercase mapping for matching
user_map_lower = {name.lower(): name for name in user_names}

from fuzzywuzzy import fuzz

def detect_user(question, user_df, threshold=50):
    question_lower = question.lower().strip()
    best_user = None
    best_score = 0

    for full_name in user_df['user_name']:
        name_tokens = full_name.lower().split()
        token_matches = [t for t in name_tokens if t in question_lower]
        score = fuzz.token_set_ratio(full_name.lower(), question_lower)
        if token_matches and score > best_score:
            best_user = full_name
            best_score = score
    if best_score >= threshold:
        return best_user, best_score
    return None, best_score

def retrieve_user_messages(user_name, question, k=10):
    # get indices of messages for that user
    user_msg_indices = df.index[df["user_name"] == user_name].tolist()

    if len(user_msg_indices) == 0:
        return []

    q_vec = embed_model.encode([f"{user_name}: {question}"], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)

    subset_embs = embeddings[user_msg_indices]
    scores = (subset_embs @ q_vec.T).squeeze()
    topk_idx_rel = np.argsort(-scores)[:k]
    top_indices = [user_msg_indices[i] for i in topk_idx_rel]

    return [df.iloc[i]["message_clean"] for i in top_indices]

def rerank_contexts(question, candidate_texts, top_k=5):
    pairs = [[question, text] for text in candidate_texts]
    scores = reranker.predict(pairs)
    order = np.argsort(-scores)
    top_contexts = [candidate_texts[i] for i in order[:top_k]]
    return top_contexts


def run_qa(question, contexts):
    if not contexts:
        return "", 0.0
    context_text = " \n---\n ".join(contexts)
    res = qa_pipe({"question": question, "context": context_text})
    return res.get("answer", ""), res.get("score", 0.0)

gen_model_name = "google/flan-t5-base"  # can be switched to larger
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)

def generate_answer_from_context(question, contexts):
    if not contexts:
        return "No related info found"

    context_text = " \n---\n ".join(contexts)
    prompt = f"Answer the question based on the following user messages:\n\n{context_text}\n\nQuestion: {question}\nAnswer:"

    input_ids = gen_tokenizer(prompt, return_tensors="pt").input_ids
    outputs = gen_model.generate(input_ids, max_new_tokens=100)
    answer = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

    if not answer.strip():
        return "No related info found"
    return answer


def answer_question_pipeline(question, k_faiss=10, top_k=5):
    '''
    Complete pipeline using generative model for final answer:
      1. Detect user (fuzzy match)
      2. Retrieve top messages for that user
      3. Rerank messages with cross-encoder
      4. Generate answer using generative model
    '''
    # Step 1: Detect user
    user_name, user_score = detect_user(question, user_docs)

    if not user_name:
        return {
            "detected_user": None,
            "answer": "No user found",
            "score": 0.0,
            "top_contexts": []
        }

    # Step 2: Retrieve candidate messages
    candidate_texts = retrieve_user_messages(user_name, question, k=k_faiss)

    if not candidate_texts:
        return {
            "detected_user": user_name,
            "answer": "No related info found",
            "score": 0.0,
            "top_contexts": []
        }

    # Step 3: Rerank top-k messages
    top_contexts = rerank_contexts(question, candidate_texts, top_k=top_k)

    # Step 4: Generate answer using generative model
    answer = generate_answer_from_context(question, top_contexts)

    return {
        "detected_user": user_name,
        "answer": answer,
        "score": 1.0 if answer != "No related info found" else 0.0,
        "top_contexts": top_contexts
    }


