import json
from functools import lru_cache

import gradio as gr
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

MODEL_ID = "Intel/distilbert-base-uncased-finetuned-sst-2-english-int8-dynamic-inc"
ONNX_FILE = "model.onnx"


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


@lru_cache(maxsize=1)
def load_assets():
    """
    Lazy-load (and cache) tokenizer + ONNX session.
    This avoids reloading the model on every request.
    """
    # Download files into HF cache (works locally + on App Runner)
    onnx_path = hf_hub_download(repo_id=MODEL_ID, filename=ONNX_FILE)
    config_path = hf_hub_download(repo_id=MODEL_ID, filename="config.json")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # id2label sometimes comes as {"0": "...", "1": "..."}
    raw = cfg.get("id2label") or {}
    id2label = {int(k): v for k, v in raw.items()} if raw else {0: "NEGATIVE", 1: "POSITIVE"}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    # Create ONNX Runtime session (CPU)
    sess_options = ort.SessionOptions()
    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
        sess_options=sess_options,
    )

    input_names = [i.name for i in session.get_inputs()]
    return tokenizer, session, input_names, id2label


def predict(text: str):
    text = (text or "").strip()
    if not text:
        return {"POSITIVE": 0.0, "NEGATIVE": 0.0}

    tokenizer, session, input_names, id2label = load_assets()

    encoded = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        padding=True,
        max_length=256,
    )

    # ONNX model expects specific inputs 
    ort_inputs = {}
    for name in input_names:
        if name in encoded:
            arr = encoded[name]
            # ensure int64 
            if arr.dtype != np.int64:
                arr = arr.astype(np.int64)
            ort_inputs[name] = arr

    outputs = session.run(None, ort_inputs)
    logits = outputs[0]  # shape: (batch, num_labels)
    probs = softmax(logits)[0]

    # Build a label->prob dict for Gradio Label component
    result = {}
    for i, p in enumerate(probs.tolist()):
        label = id2label.get(i, str(i))
        result[label] = float(p)

    # If labels are like LABEL_0/LABEL_1, normalize to POSITIVE/NEGATIVE (optional)
    # But SST-2 models typically already use POSITIVE/NEGATIVE in id2label.
    return result


with gr.Blocks() as demo:
    gr.Markdown("# 🙂/🙁 Sentiment Classifier (Hugging Face ONNX)")
    gr.Markdown(
        "Type a sentence and get **positive vs negative** probabilities. "
        "This uses an **INT8 ONNX** model so it deploys faster than PyTorch Transformers."
    )

    inp = gr.Textbox(
        label="Your text",
        placeholder="e.g., I loved this project!",
        lines=3,
    )
    out = gr.Label(num_top_classes=2, label="Prediction")

    btn = gr.Button("Analyze 🚀", variant="primary")
    btn.click(predict, inputs=inp, outputs=out)

    gr.Examples(
        examples=[
            "This is amazing — I learned a lot!",
            "I hate this. It never works.",
            "Pretty decent overall, could be better.",
        ],
        inputs=inp,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)