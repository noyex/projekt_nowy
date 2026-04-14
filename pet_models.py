import torch.nn as nn
from torchvision import models


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
        x = self.features(x)
        return self.classifier(x)


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
        x = self.features(x)
        return self.classifier(x)


def create_transfer_model(num_classes: int, freeze_backbone: bool = False):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model
import torch.nn as nn
from torchvision import models


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
        x = self.features(x)
        return self.classifier(x)


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
        x = self.features(x)
        return self.classifier(x)


def create_transfer_model(num_classes: int, freeze_backbone: bool = False):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model
import torch.nn as nn
from torchvision import models


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
        x = self.features(x)
        return self.classifier(x)


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
        x = self.features(x)
        return self.classifier(x)


def create_transfer_model(num_classes: int, freeze_backbone: bool = False):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model
"""
Wspólna architektura CNN dla Oxford-IIIT Pet (37 klas).
Używana w notebooku treningowym i w aplikacji Streamlit.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Literal

ActivationName = Literal["relu", "gelu", "leaky_relu"]


def _act(name: ActivationName) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.01, inplace=True)
    raise ValueError(name)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        use_batchnorm: bool,
        activation: ActivationName = "relu",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(_act(activation))
        layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PetCNN(nn.Module):
    """
    Prosta sieć konwolucyjna: N bloków conv + pooling, potem FC.
    """

    def __init__(
        self,
        num_classes: int = 37,
        *,
        channels: tuple[int, ...] = (32, 64, 128),
        use_batchnorm: bool = False,
        dropout_p: float = 0.0,
        activation: ActivationName = "relu",
        input_size: int = 224,
    ) -> None:
        super().__init__()
        assert len(channels) >= 2
        self.channels = channels
        blocks: list[nn.Module] = []
        in_ch = 3
        for out_ch in channels:
            blocks.append(
                ConvBlock(
                    in_ch,
                    out_ch,
                    use_batchnorm=use_batchnorm,
                    activation=activation,
                )
            )
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        # Po k kolejnych MaxPool2d(2): rozmiar = input_size // 2^k
        k = len(channels)
        feat_hw = input_size // (2**k)
        flat_dim = channels[-1] * feat_hw * feat_hw
        head: list[nn.Module] = [nn.Flatten(), nn.Linear(flat_dim, 256), _act(activation)]
        if dropout_p > 0:
            head.append(nn.Dropout(dropout_p))
        head.append(nn.Linear(256, num_classes))
        self.classifier = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


EXPERIMENT_CONFIGS = {
    "exp1_baseline": {
        "label": "Eksperyment 1 — baseline",
        "channels": (32, 64),
        "use_batchnorm": False,
        "dropout_p": 0.0,
        "activation": "relu",
    },
    "exp2_bn_dropout": {
        "label": "Eksperyment 2 — BatchNorm + Dropout",
        "channels": (32, 64),
        "use_batchnorm": True,
        "dropout_p": 0.4,
        "activation": "relu",
    },
    "exp3_aug_adam": {
        "label": "Eksperyment 3 — głębsza sieć + augmentacja + Adam",
        "channels": (32, 64, 128),
        "use_batchnorm": True,
        "dropout_p": 0.5,
        "activation": "gelu",
    },
}


def build_model(key: str, num_classes: int = 37, input_size: int = 224) -> PetCNN:
    cfg = EXPERIMENT_CONFIGS[key].copy()
    cfg.pop("label", None)
    return PetCNN(num_classes=num_classes, input_size=input_size, **cfg)


def list_experiment_keys() -> list[str]:
    return list(EXPERIMENT_CONFIGS.keys())


# Opisy do frontendu (warstwy, trening, porównanie z poprzednim eksperymentem)
EXPERIMENT_DOCS: dict[str, dict[str, str | None]] = {
    "exp1_baseline": {
        "title": "Eksperyment 1 — baseline",
        "vs_previous": None,
        "improvement_summary": (
            "Model referencyjny: najprostsza sieć i najprostszy pipeline treningowy "
            "(bez regularyzacji w części konwolucyjnej i bez augmentacji)."
        ),
        "training_summary": (
            "**Optymalizacja:** SGD (lr=0.01, momentum=0.9). "
            "**Batch size:** 32. **Epoki:** 8 (w notebookzie). "
            "**Dane:** resize 224×224, normalizacja ImageNet — **bez augmentacji**."
        ),
        "architecture_md": """
**Przepływ tensora** (wejście `3×224×224`):

| Krok | Warstwy | Wyjście (C×H×W) |
|------|---------|-----------------|
| Blok 1 | Conv 3→32, kernel 3, padding 1 → **ReLU** → MaxPool 2×2 | 32×112×112 |
| Blok 2 | Conv 32→64 → **ReLU** → MaxPool 2×2 | 64×56×56 |
| Klasyfikacja | Flatten → **Linear** 200704→256 → **ReLU** → **Linear** 256→37 | logity 37 klas |

*200704 = 64 × 56 × 56 (po dwóch poolingach rozmiar przestrzenny 224/4 = 56).*

**Uwagi:** brak BatchNorm, brak Dropout — sieć może szybciej przeuczać się na zbiorze treningowym.
""".strip(),
    },
    "exp2_bn_dropout": {
        "title": "Eksperyment 2 — BatchNorm + Dropout",
        "vs_previous": "exp1_baseline",
        "improvement_summary": (
            "Względem baseline: **BatchNorm** po każdej konwolucji (stabilniejsze skale aktywacji, często szybsza zbieżność) "
            "oraz **Dropout** w głowicy klasyfikującej przed ostatnią warstwą liniową (redukcja przeuczenia)."
        ),
        "training_summary": (
            "Te same **SGD** (lr=0.01, momentum=0.9), batch 32, 8 epok, **te same dane** co w eksperymencie 1 (bez augmentacji), "
            "żeby porównać wyłącznie wpływ architektury/regularyzacji."
        ),
        "architecture_md": """
**Przepływ** — **te same kanały i rozmiary co w exp1**, zmieniona jest tylko zawartość bloków i głowicy:

| Krok | Warstwy | Uwagi |
|------|---------|--------|
| Blok 1 | Conv 3→32 → **BatchNorm2d** → ReLU → MaxPool 2×2 | BN po konwolucji |
| Blok 2 | Conv 32→64 → **BatchNorm2d** → ReLU → MaxPool 2×2 | j.w. |
| Klasyfikacja | Flatten → Linear 200704→256 → ReLU → **Dropout(p=0.4)** → Linear 256→37 | Dropout tylko w głowicy |

Spłaszczenie nadal **200704** — ta sama głębokość co baseline, inna regularyzacja.
""".strip(),
    },
    "exp3_aug_adam": {
        "title": "Eksperyment 3 — głębsza sieć + augmentacja + Adam",
        "vs_previous": "exp2_bn_dropout",
        "improvement_summary": (
            "Względem exp2: **trzeci blok konwolucyjny** (więcej pojemności), aktywacja **GELU** zamiast ReLU w blokach i FC, "
            "**Adam** zamiast SGD, **augmentacja** obrazów na treningu (odbicie, rotacja, ColorJitter), batch 24, więcej epok w notebooku."
        ),
        "training_summary": (
            "**Optymalizacja:** Adam (lr=1e-3). **Batch:** 24. **Epoki:** 10. "
            "**Augmentacja (tylko trening):** RandomHorizontalFlip, RandomRotation(15°), ColorJitter. "
            "Walidacja: ten sam podział indeksów co wcześniej, bez augmentacji."
        ),
        "architecture_md": """
**Przepływ** (`3×224×224`):

| Krok | Warstwy | Wyjście (C×H×W) |
|------|---------|-----------------|
| Blok 1 | Conv 3→32 → BN → **GELU** → MaxPool | 32×112×112 |
| Blok 2 | Conv 32→64 → BN → **GELU** → MaxPool | 64×56×56 |
| Blok 3 | Conv 64→128 → BN → **GELU** → MaxPool | 128×28×28 |
| Klasyfikacja | Flatten → Linear **100352**→256 → **GELU** → Dropout(0.5) → Linear 256→37 | logity 37 klas |

*100352 = 128 × 28 × 28 (trzy kolejne MaxPool: 224/8 = 28).*

To jest **głębszy** model niż exp1/exp2; łączy regularyzację z exp2 z bogatszą cechą i mocniejszym pipeline danych.
""".strip(),
    },
}


def get_experiment_doc(key: str) -> dict[str, str | None]:
    return EXPERIMENT_DOCS[key]
