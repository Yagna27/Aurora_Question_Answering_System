# Aurora QA System

Aurora QA System is a natural-language question-answering application that infers answers from member messages. It leverages a combination of embedding-based retrieval, cross-encoder re-ranking, and a generative model to produce accurate and readable answers.

---

## Approach

The system follows these steps:

### 1. User Detection
- Uses fuzzy string matching to identify the user the question is about.

### 2. Message Retrieval
- Retrieves the most relevant messages from the identified user using:
  - Embedding-based similarity (`multi-qa-mpnet-base-dot-v1`)
  - FAISS for fast similarity search

### 3. Reranking Contexts
- Re-ranks the top messages using a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to improve relevance.

### 4. Answer Generation
- Uses a generative model (`google/flan-t5-base`) to produce a coherent final answer based on the top-ranked messages.

### 5. Output
The system returns a structured response:

```json
{
    "detected_user": "User Name",
    "answer": "Answer to the question",
    "score": 1.0,
    "top_contexts": ["Relevant message 1", "Relevant message 2", "..."]
}
```
## Why Hugging Face Spaces

Initially, the system was tested for deployment on Render:

- Render's free instance memory (512 MB) was insufficient for the models.
- Managing dependencies and deployment caused repeated build errors.
- Public API access required extra setup for ports and HTTPS.

Hugging Face Spaces provides:

- Free, publicly accessible hosting
- Native Python support
- Easy integration with Gradio for web-based UI and API

This made it the ideal choice for deployment.

---

## How to Use

### 1. Through the Space Web Interface
- Go to the Space
- Type your question in the input box
- Get the inferred answer

### 2. Using the Hugging Face API

```python
from gradio_client import Client

# Connect to your Space
client = Client("Yagna27/Aurora_Question_Answering_System")

# Make a query
result = client.predict(
    question="what is vikram's new office address?",
    api_name="/predict"
)

print(result)
```
Input: question (string) — a natural-language question

Output: string — answer inferred from user messages

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
