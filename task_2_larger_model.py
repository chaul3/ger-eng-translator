from pathlib import Path
import torch


# Task 2 is:
# - to use a subword-level model (BPE or sentence piece)
# - train a larger model on GPU (512 states, 50 epochs, mbz 32, at least 20 epochs) on 15k translations
# - save a model with 512 states as 512-translations-model.pt and commit to repo
# - add a temperature parameter to "translate" function so that the below script executes

model_path = Path('experiments') / 'seq-2-seq' / '512-translations-model.pt'

# load the model again
translator = torch.load(model_path)
print(translator)

source = "Wir sehen uns ."
for temperature in [0.6, 0.7, 0.8, 0.9, 1.0]:
    print(f"{source} is translated as {translator.translate(source, temperature)}")

source = "Wir sind echt gut ."
for temperature in [0.6, 0.7, 0.8, 0.9, 1.0]:
    print(f"{source} is translated as {translator.translate(source, temperature)}")

source = "Wir verstehen das ."
for temperature in [0.6, 0.7, 0.8, 0.9, 1.0]:
    print(f"{source} is translated as {translator.translate(source, temperature)}")