import gradio as gr
import requests

RAG_API_URL = "http://rag-service:8081/query"


def chat(message, history):

    payload = {
        "question": message
    }

    response = requests.post(
        RAG_API_URL,
        json=payload
    )

    data = response.json()

    answer = data["answer"]

    context = data.get("context")

    final_answer = f"""
        {answer}

        Used context:
        {context}
        """

    return final_answer


if __name__=="__main__":
    demo = gr.ChatInterface(
        fn=chat,
        title="RAG Assistant",
        description="Lokalny system RAG"
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=8080
    )