
import os
import gradio as gr
from qa_pipeline import answer_question_pipeline

def ask_question(question: str):
    '''
    Wrapper function to call your QA pipeline.
    Returns only the answer string.
    '''
    if not question.strip():
        return "Please enter a question."

    res = answer_question_pipeline(question)
    answer = res.get("answer", "No answer found")
    return answer

# Gradio interface
demo = gr.Interface(
    fn=ask_question,
    inputs=gr.Textbox(lines=2, placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Answer"),
    title="Aurora QA System",
    description="Enter a question and get an answer inferred from user messages."
)

# Launch the Gradio app
demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
    share=True  # Optional: gives you a temporary public link
)
