import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

st.set_page_config(page_title="Oxford Pets Dashboard v2", layout="wide")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_ROOT = BASE_DIR / "artifacts_v2"
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


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ImprovedCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def create_transfer_model(num_classes: int):
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def build_model(model_key, num_classes):
    if model_key == "baseline":
        return BaselineCNN(num_classes)
    if model_key == "improved":
        return ImprovedCNN(num_classes)
    if model_key == "transfer":
        return create_transfer_model(num_classes)
    raise ValueError(model_key)


def load_summary():
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    # Fallback: składanie danych z folderów eksperymentów, gdy summary.json nie istnieje
    experiment_dirs = sorted([p for p in ARTIFACTS_ROOT.iterdir() if p.is_dir()])
    if not experiment_dirs:
        return None

    experiments = []
    class_names = []
    for exp_dir in experiment_dirs:
        metrics_path = exp_dir / "metrics.json"
        history_path = exp_dir / "history.json"
        if not metrics_path.exists() or not history_path.exists():
            continue

        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)

        exp_name = exp_dir.name
        experiments.append(
            {
                "name": exp_name,
                "title": exp_name,
                "model_key": "transfer",
                "changes": ["Dane załadowane z artifacts_v2"],
                "history": history,
                "metrics": metrics,
            }
        )

        ckpt_path = exp_dir / "best_model.pth"
        if ckpt_path.exists() and not class_names:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            class_names = checkpoint.get("class_names", [])

    return {"class_names": class_names, "experiments": experiments} if experiments else None


def load_experiment_model(exp_name, class_count):
    ckpt_path = ARTIFACTS_ROOT / exp_name / "best_model.pth"
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model = build_model(checkpoint["model_key"], class_count).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def predict(model, image):
    x = eval_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return pred_idx, float(probs[pred_idx]), probs


MODEL_ARCHITECTURE_DETAILS = {
    "baseline": {
        "name": "Bazowa CNN",
        "layers": [
            "Wejście: 3x128x128",
            "Conv2d(3->32, k=3, p=1) + ReLU + MaxPool2d(2) -> 32x64x64",
            "Conv2d(32->64, k=3, p=1) + ReLU + MaxPool2d(2) -> 64x32x32",
            "Conv2d(64->128, k=3, p=1) + ReLU + MaxPool2d(2) -> 128x16x16",
            "Flatten",
            "Linear(32768->256) + ReLU",
            "Linear(256->37)",
        ],
    },
    "improved": {
        "name": "Ulepszona CNN",
        "layers": [
            "Wejście: 3x128x128",
            "Conv2d(3->32) + BatchNorm2d(32) + ReLU + MaxPool2d(2) -> 32x64x64",
            "Conv2d(32->64) + BatchNorm2d(64) + ReLU + MaxPool2d(2) -> 64x32x32",
            "Conv2d(64->128) + BatchNorm2d(128) + ReLU + MaxPool2d(2) -> 128x16x16",
            "Conv2d(128->256) + BatchNorm2d(256) + ReLU + MaxPool2d(2) -> 256x8x8",
            "Flatten",
            "Dropout(0.5)",
            "Linear(16384->512) + ReLU",
            "Dropout(0.3)",
            "Linear(512->37)",
        ],
    },
    "transfer": {
        "name": "Transfer Learning (ResNet18)",
        "layers": [
            "Wejście: 3x128x128",
            "Backbone: ResNet18 (pretrained, fine-tuning całej sieci)",
            "ResNet18: conv1 + bn1 + relu + maxpool",
            "ResNet18: layer1 (2 bloki residual)",
            "ResNet18: layer2 (2 bloki residual)",
            "ResNet18: layer3 (2 bloki residual)",
            "ResNet18: layer4 (2 bloki residual)",
            "AdaptiveAvgPool2d + Flatten",
            "Nowa głowica: Dropout(0.3) + Linear(512->37)",
        ],
    },
}

CHANGES_VS_PREVIOUS = {
    "exp1_baseline": [
        "Model startowy (brak poprzednika).",
    ],
    "exp2_improved": [
        "Dodano 4. blok konwolucyjny (128->256), więc sieć jest głębsza.",
        "Dodano BatchNorm po każdej konwolucji (stabilniejsze uczenie).",
        "Dodano Dropout w klasyfikatorze (mniejsze overfitting).",
        "Zmieniono głowicę z Linear(32768->256->37) na Linear(16384->512->37) + dropout.",
    ],
    "exp3_transfer": [
        "Zamiast własnej CNN użyto backbone ResNet18 (pretrained).",
        "Dodano transfer learning i fine-tuning całej sieci.",
        "Głowica końcowa zmieniona na Dropout(0.3) + Linear(512->37).",
        "Pipeline treningowy oparty na augmentacji z poprzedniego eksperymentu.",
    ],
}


def inject_styles():
    st.markdown(
        """
        <style>
        .main > div {
            padding-top: 1rem;
        }
        .hero {
            padding: 1.1rem 1.2rem;
            border-radius: 14px;
            background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            color: #f9fafb;
            font-size: 1.7rem;
        }
        .hero p {
            margin: 0.4rem 0 0 0;
            color: #d1d5db;
            font-size: 0.95rem;
        }
        .small-muted {
            color: #9ca3af;
            font-size: 0.85rem;
        }
        .block-title {
            font-weight: 700;
            font-size: 1.02rem;
            margin-bottom: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def fmt_pct(value):
    return f"{value * 100:.2f}%"


def fmt_sec(value):
    return f"{value:.1f} s"


st.title("Oxford-IIIT Pet — Dashboard eksperymentów CNN (v2)")
st.caption("Opis eksperymentów, metryki, wykresy i test modeli.")
inject_styles()

payload = load_summary()
if payload is None:
    st.error("Brak artifacts_v2/summary.json. Najpierw uruchom notebook pet_classification_project_v2.ipynb.")
    st.stop()

class_names = payload["class_names"]
experiments = payload["experiments"]

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
best_row = df.iloc[0]

st.markdown(
    f"""
    <div class="hero">
        <h1>Oxford-IIIT Pet — Analiza eksperymentów</h1>
        <p>Najlepszy model: <b>{best_row['experiment']}</b> | Test accuracy: <b>{fmt_pct(best_row['test_acc'])}</b></p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Podsumowanie")
    st.metric("Najlepszy test_acc", fmt_pct(best_row["test_acc"]), best_row["experiment"])
    st.metric("Średni test_acc", fmt_pct(df["test_acc"].mean()))
    fastest = df.loc[df["training_time_sec"].idxmin()]
    st.metric("Najszybszy trening", fmt_sec(fastest["training_time_sec"]), fastest["experiment"])
    st.markdown("---")
    st.markdown('<div class="small-muted">Źródło danych: artifacts_v2</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Przegląd", "Architektury", "Test modelu"])

with tab1:
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Najlepszy test_acc", fmt_pct(best_row["test_acc"]), best_row["experiment"])
    col_b.metric("Najwyższy val_acc", fmt_pct(df["val_acc_best"].max()))
    col_c.metric("Średni czas treningu", fmt_sec(df["training_time_sec"].mean()))

    st.markdown("### Ranking eksperymentów")
    st.dataframe(df, use_container_width=True)

    st.markdown("### Test accuracy")
    st.bar_chart(df.set_index("experiment")["test_acc"], use_container_width=True)

    st.markdown("### Postęp względem maksimum 100%")
    for _, row in df.iterrows():
        st.write(f"**{row['experiment']}** — {fmt_pct(row['test_acc'])}")
        st.progress(float(row["test_acc"]))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Krzywe walidacyjne accuracy")
        fig, ax = plt.subplots(figsize=(7, 4))
        for exp in experiments:
            y = exp["history"]["val_acc"]
            ax.plot(y, linewidth=2, label=f"{exp['name']}")
            ax.fill_between(range(len(y)), y, alpha=0.1)
        ax.set_xlabel("Epoka")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.markdown("### Krzywe walidacyjne loss")
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        for exp in experiments:
            y2 = exp["history"]["val_loss"]
            ax2.plot(y2, linewidth=2, label=f"{exp['name']}")
            ax2.fill_between(range(len(y2)), y2, alpha=0.1)
        ax2.set_xlabel("Epoka")
        ax2.set_ylabel("Loss")
        ax2.grid(alpha=0.25)
        ax2.legend()
        st.pyplot(fig2)

with tab2:
    st.subheader("Architektury i zmiany między modelami")
    for exp in experiments:
        model_key = exp.get("model_key", "baseline")
        details = MODEL_ARCHITECTURE_DETAILS.get(model_key)
        with st.expander(f"{exp['title']} ({exp['name']})", expanded=(exp["name"] == "exp3_transfer")):
            m = exp["metrics"]
            k1, k2, k3 = st.columns(3)
            k1.metric("Train accuracy (ostatnia epoka)", fmt_pct(m["train_acc_last"]))
            k2.metric("Best val accuracy", fmt_pct(m["val_acc_best"]))
            k3.metric("Test accuracy", fmt_pct(m["test_acc"]))

            st.markdown('<div class="block-title">Architektura warstwa po warstwie</div>', unsafe_allow_html=True)
            if details:
                st.caption(f"Model: {details['name']}")
                for layer_desc in details["layers"]:
                    st.write(f"- {layer_desc}")
            else:
                st.write("- Brak szczegółowego opisu architektury dla tego modelu.")

            st.markdown('<div class="block-title">Zmiany względem poprzedniego eksperymentu</div>', unsafe_allow_html=True)
            for change_desc in CHANGES_VS_PREVIOUS.get(exp["name"], exp.get("changes", [])):
                st.write(f"- {change_desc}")

with tab3:
    st.subheader("Predykcja na własnym zdjęciu")
    exp_options = [e["name"] for e in experiments]
    selected_exp = st.selectbox("Wybierz model", exp_options, index=len(exp_options) - 1)
    uploaded = st.file_uploader("Wgraj obraz (jpg/png)", type=["jpg", "jpeg", "png"])

    selected_ckpt = ARTIFACTS_ROOT / selected_exp / "best_model.pth"
    st.caption(f"Ścieżka checkpointu: `{selected_ckpt}`")
    if not selected_ckpt.exists():
        st.info(
            "W artifacts_v2 nie ma pliku best_model.pth dla wybranego eksperymentu. "
            "Zakładka testu modelu wymaga checkpointu, ale metryki i wykresy działają poprawnie."
        )
        uploaded = None

    if not class_names:
        st.warning("Brak class_names w artifacts_v2. Predykcja klas może nie działać poprawnie.")
        uploaded = None

    if uploaded is not None:
        st.markdown("### Podgląd i wynik")
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Wgrany obraz", width=360)

        model = load_experiment_model(selected_exp, len(class_names))
        pred_idx, pred_prob, probs = predict(model, image)
        st.success(f"Predykcja: **{class_names[pred_idx]}** (pewność: **{pred_prob:.2%}**)")

        topk = np.argsort(probs)[::-1][:5]
        top_df = pd.DataFrame(
            {
                "klasa": [class_names[i] for i in topk],
                "prawdopodobieństwo": [float(probs[i]) for i in topk],
            }
        )
        col_top, col_bar = st.columns([1, 1])
        with col_top:
            st.markdown("**Top-5 klas**")
            st.dataframe(top_df, use_container_width=True)
        with col_bar:
            st.markdown("**Rozkład Top-5**")
            chart_df = top_df.set_index("klasa")
            st.bar_chart(chart_df["prawdopodobieństwo"], use_container_width=True)
