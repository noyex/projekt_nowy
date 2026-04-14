import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from pet_models import BaselineCNN, ImprovedCNN, create_transfer_model

st.set_page_config(page_title="Oxford Pets CNN Dashboard", layout="wide")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARTIFACTS_ROOT = Path("artifacts")
SUMMARY_PATH = ARTIFACTS_ROOT / "summary.json"

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
eval_transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)


def load_summary():
    if not SUMMARY_PATH.exists():
        return None
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model(model_key, num_classes):
    if model_key == "baseline":
        return BaselineCNN(num_classes)
    if model_key == "improved":
        return ImprovedCNN(num_classes)
    if model_key == "transfer":
        return create_transfer_model(num_classes, freeze_backbone=False)
    raise ValueError(f"Nieznany model_key: {model_key}")


def load_experiment_model(exp_name, class_count):
    ckpt_path = ARTIFACTS_ROOT / exp_name / "best_model.pth"
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model = build_model(checkpoint["model_key"], class_count).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def predict(model, image, class_names):
    x = eval_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    idx = int(np.argmax(probs))
    return idx, float(probs[idx]), probs


st.title("Oxford-IIIT Pet — Dashboard eksperymentów CNN")
st.caption("Analiza wyników + test wytrenowanych modeli")

payload = load_summary()
if payload is None:
    st.error("Brak artifacts/summary.json. Najpierw uruchom notebook i wytrenuj modele.")
    st.stop()

class_names = payload["class_names"]
experiments = payload["experiments"]

tab1, tab2, tab3 = st.tabs(["Opis eksperymentów", "Metryki i wykresy", "Test modeli"])

with tab1:
    st.subheader("Kolejne ulepszenia")
    for exp in experiments:
        with st.container(border=True):
            st.markdown(f"### {exp['title']}")
            st.write("Zmiany względem poprzedniego:")
            for change in exp["changes"]:
                st.write(f"- {change}")

with tab2:
    rows = []
    for exp in experiments:
        rows.append(
            {
                "experiment": exp["name"],
                "title": exp["title"],
                "train_acc_last": exp["metrics"]["train_acc_last"],
                "val_acc_best": exp["metrics"]["val_acc_best"],
                "test_acc": exp["metrics"]["test_acc"],
                "test_loss": exp["metrics"]["test_loss"],
                "training_time_sec": exp["metrics"]["training_time_sec"],
            }
        )
    df = pd.DataFrame(rows).sort_values("test_acc", ascending=False).reset_index(drop=True)
    st.dataframe(df, use_container_width=True)

    st.subheader("Porównanie test accuracy")
    st.bar_chart(df.set_index("experiment")["test_acc"])

    st.subheader("Krzywe walidacyjne")
    fig, ax = plt.subplots(figsize=(10, 4))
    for exp in experiments:
        ax.plot(exp["history"]["val_acc"], label=f"{exp['name']} val_acc")
    ax.set_xlabel("Epoka")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

with tab3:
    st.subheader("Sprawdź model na własnym obrazie")
    exp_options = [e["name"] for e in experiments]
    selected_exp = st.selectbox("Wybierz eksperyment", exp_options, index=len(exp_options) - 1)
    uploaded = st.file_uploader("Wgraj zdjęcie psa/kota", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Wgrany obraz", width=300)

        model = load_experiment_model(selected_exp, len(class_names))
        pred_idx, pred_prob, probs = predict(model, image, class_names)

        st.success(f"Predykcja: **{class_names[pred_idx]}** (pewność: {pred_prob:.2%})")

        topk = np.argsort(probs)[::-1][:5]
        top_df = pd.DataFrame(
            {
                "klasa": [class_names[i] for i in topk],
                "prawdopodobieństwo": [float(probs[i]) for i in topk],
            }
        )
        st.dataframe(top_df, use_container_width=True)
