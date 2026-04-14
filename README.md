# Projekt: Oxford-IIIT Pet Classification

Repo zawiera:
- `pet_classification_project.ipynb` — główny notebook do analizy i treningu 3 eksperymentów.
- `pet_models.py` — definicje modeli.
- `frontend_app.py` — dashboard Streamlit z opisem eksperymentów, metrykami, wykresami i testem modeli.

## Jak uruchomić

1. Zainstaluj zależności:
```bash
pip install -r requirements.txt
```

2. Uruchom notebook:
- `pet_classification_project.ipynb`
- wykonaj wszystkie komórki, aby:
  - pobrać dataset,
  - wytrenować modele,
  - zapisać wyniki do `artifacts/`.

3. Uruchom frontend:
```bash
streamlit run frontend_app.py
```

## Co zawiera notebook

1. Analiza datasetu (statystyki klas + przykładowe obrazy).
2. Trzy eksperymenty:
   - `exp1_baseline` — baza.
   - `exp2_improved` — BatchNorm + Dropout + augmentacja.
   - `exp3_transfer` — transfer learning (ResNet18), ulepszenie eksperymentu 2.
3. Wyniki:
   - numeryczne (`train/val/test accuracy`, `test loss`),
   - wizualne (wykresy uczenia, confusion matrix, predykcje obrazów).
4. Wnioski i pomysły na dalsze ulepszenia.
