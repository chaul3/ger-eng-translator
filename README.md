# 📚 Sprachmodellierung mit PyTorch

## 🔍 Projektübersicht
Dieses Projekt implementiert ein Sprachmodell mit PyTorch, das auf verschiedenen Datensätzen trainiert werden kann, wie z. B. motivierende Zitate, mathematische Gleichungen oder ein Sanity-Check-Datensatz. Das Modell basiert auf einem rekurrenten neuronalen Netzwerk (RNN) und unterstützt sowohl Wort- als auch Zeichenmodellierung. Ziel ist es, Textsequenzen vorherzusagen und neue Inhalte basierend auf einem Start-Prompt zu generieren.

## 🛠️ Funktionen
- **RNN-basiertes Sprachmodell**: Implementierung mit LSTM für bessere Langzeitabhängigkeiten.
- **Wort- und Zeichenmodellierung**: Flexibilität bei der Auswahl der Modellierungsebene.
- **Training mit Mini-Batches**: Effizientes Training mit Stochastic Gradient Descent (SGD).
- **Validierung und Testauswertung**: Überprüfung der Modellleistung auf separaten Datensätzen.
- **Modell-Checkpointing**: Speichert das beste Modell basierend auf Validierungsverlust.
- **Textgenerierung**: Erzeugt neue Textsequenzen basierend auf einem Start-Prompt.
- **Visualisierung**: Darstellung von Verlust- und Perplexitätskurven zur Analyse des Trainingsprozesses.

## 📁 Projektstruktur
```plaintext
projekt/
├── data/
│   └── language_modeling/
│       └── sanity_check.txt          # Kleiner Testdatensatz
│       └── equations.txt             # Mathematische Gleichungen
│       └── motivational_quotes.txt   # Motivierende Zitate
│
├── experiments/
│   └── language_modeling/
│       └── sanity_check-model.pt     # Modell für Sanity-Check-Datensatz
│       └── equations-model.pt        # Modell für Gleichungen
│       └── motivational_quotes-model.pt # Modell für Zitate
│
├── helper_functions.py               # Hilfsfunktionen für Visualisierung und Wörterbucherstellung
├── language_model.py                 # RNN-Modell mit Trainings- und Inferenzlogik
└── main.py                           # Hauptskript zum Trainieren und Ausführen des Modells
```

## 📦 Abhängigkeiten
Um das Projekt auszuführen, müssen folgende Abhängigkeiten installiert werden:
- **Python**: Version 3.8 oder höher
- **PyTorch**: Für die Implementierung des neuronalen Netzwerks
- **`more_itertools`**: Für erweiterte Iterationsfunktionen
- **`matplotlib`**: Für die Visualisierung von Trainingsmetriken

Installiere die Abhängigkeiten mit:
```bash
pip install torch more-itertools matplotlib
```

## ⚙️ Konfiguration
Die Konfiguration des Modells erfolgt im Skript `main.py`. Wähle den gewünschten Datensatz und die Modellierungsebene:
```python
dataset = 'sanity_check'  # Optionen: 'sanity_check', 'equations', 'motivational_quotes'
character_level = False   # True für Zeichenmodellierung, False für Wortmodellierung
```

## 🚀 Nutzung
### Training
Starte das Training des Modells mit:
```bash
python main.py
```
Das Skript trainiert das Modell basierend auf der Konfiguration und speichert das beste Modell in `experiments/language_modeling/`.

### Textgenerierung
Nach dem Training kannst du Textsequenzen generieren:
```bash
python language_model.py --generate --prompt "Dein Start-Prompt"
```
Ersetze `"Dein Start-Prompt"` durch den gewünschten Anfangstext.

### Visualisierung
Um Verlust- und Perplexitätskurven zu visualisieren, führe die Hilfsfunktionen aus:
```bash
python helper_functions.py --plot
```

## 📊 Ergebnisse
Die Ergebnisse des Modells werden in der Konsole ausgegeben und die besten Modelle werden in `experiments/language_modeling/` gespeichert. Die generierten Texte können direkt aus dem Modell abgerufen werden.

## 🛠️ Weiterentwicklung
- Integration von Transformer-Modellen für bessere Leistung.
- Unterstützung für größere Datensätze und Multi-GPU-Training.
- Erweiterung der Visualisierungsoptionen.

## 📄 Lizenz
Dieses Projekt ist unter der MIT-Lizenz veröffentlicht. Siehe die Datei `LICENSE` für weitere Informationen.

## ✨ Autor
Dieses Projekt wurde von **Chau Le** entwickelt. Verbinde dich auf [LinkedIn](https://www.linkedin.com/in/chaule0702/) für weitere Informationen oder Zusammenarbeit.