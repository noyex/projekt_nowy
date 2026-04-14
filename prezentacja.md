# Prezentacja projektu — Oxford-IIIT Pet Dataset

## 1. Cel projektu
- Klasyfikacja ras psów i kotów na podstawie obrazów.
- Porównanie 3 podejść CNN, gdzie każdy kolejny eksperyment ulepsza poprzedni.

## 2. Dataset
- Oxford-IIIT Pet Dataset
- 37 klas
- Zbiory: `trainval` i `test`

## 3. Analiza danych
- Rozkład liczności klas.
- Wizualizacja przykładowych obrazów.
- Wniosek: zróżnicowane pozycje, oświetlenie i tła, co uzasadnia augmentację.

## 4. Eksperymenty

### Eksperyment 1 — Bazowa CNN
- Prosta architektura konwolucyjna.
- Brak augmentacji.
- Punkt odniesienia.

### Eksperyment 2 — Ulepszona CNN
- Dodano BatchNorm i Dropout.
- Dodano Data Augmentation.
- Ulepszenie stabilności uczenia i generalizacji.

### Eksperyment 3 — Transfer Learning (ResNet18)
- Start od eksperymentu 2 (augmentacja).
- Zastąpienie custom CNN modelem pretrained.
- Fine-tuning z mniejszym learning rate.

## 5. Wyniki
- Tabela metryk: train/val/test accuracy + loss.
- Wykresy przebiegu uczenia.
- Confusion matrix dla najlepszego modelu.
- Przykładowe predykcje na obrazach testowych.

## 6. Wnioski
- Najlepszy wynik zwykle osiąga eksperyment 3 (transfer learning).
- BatchNorm + Dropout + augmentacja poprawiają model względem bazy.
- Transfer learning znacząco przyspiesza dojście do wysokiej accuracy.

## 7. Co można poprawić
- Dłuższy fine-tuning i scheduler cosine.
- Silniejsza augmentacja (np. RandomErasing).
- Sprawdzenie innych backbone'ów (EfficientNet/MobileNet).
- Early stopping i szerszy tuning hiperparametrów.
