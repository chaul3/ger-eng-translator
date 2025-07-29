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
projekt/
│
├── data/
│ └── language_modeling/
│ └── <dataset>.txt
│
├── experiments/
│ └── language_modeling/
│ └── <dataset>-model.pt
│
├── helper_functions.py
├── language_model.py
└── main.py

## 📦 Abhängigkeiten
- Python 3.8 oder höher
- PyTorch
- `more_itertools`
- `matplotlib`

Installiere die Abhängigkeiten mit:
```bash
pip install torch more-itertools matplotlib

## ⚙️ Konfiguration
Wähle in main.py den gewünschten Datensatz und Modellierungsmodus:
dataset = 'sanity_check'  # oder 'equations', 'motivational_quotes'
character_level = False   # oder True für Zeichenmodellierung
