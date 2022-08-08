from pathlib import Path

import torch

model_path = Path('experiments') / 'language_modeling' / 'motivational_quotes-model.pt'

# load the model again
language_model = torch.load(model_path)

print("\n--- life is ---")
for temperature in [0.6, 0.7, 0.8, 0.9, 1.0]:
    language_model.generate_text(prefix="life is", temperature=temperature)

print("\n--- marriage is ---")
for temperature in [0.6, 0.7, 0.8, 0.9, 1.0]:
    language_model.generate_text(prefix="marriage is", temperature=temperature)

print("\n--- luck is ---")
for temperature in [0.6, 0.7, 0.8, 0.9, 1.0]:
    language_model.generate_text(prefix="luck is", temperature=temperature)
