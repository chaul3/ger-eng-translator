# 📚 Sprachmodellierung mit PyTorch

## 🔍 Projektübersicht
Dieses Projekt implementiert ein einfaches Sprachmodell mit PyTorch. Das Modell kann auf verschiedenen Datensätzen trainiert werden (z. B. motivierende Zitate, mathematische Gleichungen oder ein Sanity-Check-Datensatz) und lernt, Textsequenzen vorherzusagen. Es basiert auf einem RNN und unterstützt sowohl Wort- als auch Zeichenebene.

## 🛠️ Funktionen
- RNN-basiertes Sprachmodell (mit LSTM)
- Unterstützung für Wort- und Zeichenmodellierung
- Training mit Mini-Batches und SGD
- Validierung und Testauswertung
- Modell-Checkpointing (bestes Modell wird gespeichert)
- Textgenerierung aus einem Start-Prompt
- Visualisierung von Verlust- und Perplexitätskurven

## 📁 Projektstruktur
```plaintext
projekt/
├── data/
│   └── language_modeling/
│       └── sanity_check.txt
│       └── equations.txt
│       └── motivational_quotes.txt
│
├── experiments/
│   └── language_modeling/
│       └── sanity_check-model.pt
│       └── equations-model.pt
│       └── motivational_quotes-model.pt
│
├── helper_functions.py         # Hilfsfunktionen für Visualisierung & Wortdictionary
├── language_model.py           # RNN-Modell mit Trainings- und Inferenzlogik
└── main.py                     # Hauptskript zum Trainieren und Ausführen
```
## 📦 Abhängigkeiten
- Python 3.8 oder höher
- PyTorch
- `more_itertools`
- `matplotlib`

Installiere die Abhängigkeiten mit:
```bash
pip install torch more-itertools matplotlib
```
## ⚙️ Konfiguration
Wähle in main.py den gewünschten Datensatz und Modellierungsmodus:
```python
dataset = 'sanity_check'  # oder 'equations', 'motivational_quotes'
character_level = False   # oder True für Zeichenmodellierung
```
