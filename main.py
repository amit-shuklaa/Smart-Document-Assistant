from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import gradio as gr
import fitz  # PyMuPDF
import io

# Load model and tokenizer
model_id = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to("cuda" if torch.cuda.is_available() else "cpu")

# Global state
text = ""
summary = ""

def generate_answer(prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()

def handle_upload(file):
    global text, summary
    try:
        file_bytes = file.read()
        filename = file.name

        if filename.endswith(".pdf"):
            with fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf") as doc:
                text = "".join([page.get_text() for page in doc])
        else:
            text = file_bytes.decode("utf-8")

        if not text.strip():
            return "❌ Unable to extract text. Please upload a proper text-based PDF or TXT file."

        summary_prompt = f"Summarize this in under 150 words:\n{text[:2000]}"
        summary = generate_answer(summary_prompt)
        return summary

    except Exception as e:
        return f"❌ Error processing file: {str(e)}"

def answer_question(question):
    if not text:
        return "Please upload a document first."
    answer_prompt = f"Answer based only on this text:\n{text[:2000]}\nQuestion: {question}\nAlso explain where you found the answer."
    return generate_answer(answer_prompt)

def challenge_me():
    global text
    if not text or len(text.strip()) < 100:
        return ["❌ Not enough content to generate questions.", "", ""]

    try:
        prompt = (
            f"Document:\n{text[:2000]}\n\n"
            "Generate exactly 3 high-quality logic-based questions with these requirements:\n"
            "1. Each must test comprehension, not just facts\n"
            "2. Format as:\n"
            "Question 1: [question]\n"
            "Question 2: [question]\n"
            "Question 3: [question]\n"
            "3. Include no other text in your response"
        )

        response = generate_answer(prompt)

        questions = []
        for i in range(1, 4):
            prefix = f"Question {i}:"
            start = response.find(prefix)
            if start >= 0:
                start += len(prefix)
                end = response.find("\n", start)
                question = response[start:end if end >= 0 else None].strip()
                if question:
                    questions.append(question)

        if len(questions) == 3:
            return questions

        final_questions = []
        for i in range(3):
            if i < len(questions):
                final_questions.append(questions[i])
            else:
                retry_prompt = (
                    f"Document:\n{text[:2000]}\n\n"
                    f"Generate just 1 logic-based question testing comprehension about this content. "
                    "Return only the question with no numbering or extra text."
                )
                question = generate_answer(retry_prompt).strip()
                final_questions.append(question if question else "Could not generate question")

        return final_questions

    except Exception as e:
        return [f"❌ Error: {str(e)}", "Could not generate question", "Could not generate question"]

def evaluate_answer(q, a):
    eval_prompt = f"Document:\n{text[:2000]}\n\nEvaluate this answer: '{a}' for the question: '{q}'.\nWas it correct? Justify with a snippet."
    return generate_answer(eval_prompt)

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📚 Smart Assistant for Research Summarization")

    with gr.Row():
        file = gr.File(label="Upload PDF or TXT", type="binary")
        summary_output = gr.Textbox(label="📌 Summary (≤150 words)", lines=5)
        file.upload(fn=handle_upload, inputs=file, outputs=summary_output)

    gr.Markdown("### 💬 Ask Anything Mode")
    question = gr.Textbox(label="Ask a question about the document")
    answer = gr.Textbox(label="🤖 Answer with Justification", lines=4)
    question.submit(fn=answer_question, inputs=question, outputs=answer)

    gr.Markdown("### 🧠 Challenge Me Mode")
    challenge_btn = gr.Button("Generate 3 Logic-Based Questions")
    q1 = gr.Textbox(label="Q1")
    q2 = gr.Textbox(label="Q2")
    q3 = gr.Textbox(label="Q3")
    challenge_btn.click(fn=challenge_me, outputs=[q1, q2, q3])

    gr.Markdown("### 🧪 Evaluate My Answers")
    selected_q = gr.Textbox(label="Enter the question")
    user_a = gr.Textbox(label="Your Answer")
    feedback = gr.Textbox(label="🧠 Feedback with Explanation")
    evaluate = gr.Button("Evaluate")
    evaluate.click(fn=evaluate_answer, inputs=[selected_q, user_a], outputs=feedback)

    gr.Markdown("---\n⚡ Built with Hugging Face Transformers & Gradio")

demo.launch(share=True)
