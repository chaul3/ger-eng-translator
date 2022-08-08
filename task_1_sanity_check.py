from pathlib import Path
import torch


# Task 1 is:
# - to implement a sequence-to-sequence model
# - train it on the translations_sanity_check.txt data.
# - save a model with 128 states as sanity-translations-model.pt and commit to repo
# - write a "translate" function so that the model passes the below assertions

model_path = Path('experiments') / 'seq-2-seq' / 'sanity-translations-model.pt'

# load the model again
translator = torch.load(model_path)
print(translator)

# correctly translate "Was ist das?"
source = "Was ist das ?"
target = "What is it ?"
assert translator.translate(source) == target

# correctly translate "Bist du sicher ?"
source = "Bist du sicher ?"
target = "Are you sure ?"
assert translator.translate(source) == target

# correctly translate "Vielen Dank !"
source = "Vielen Dank !"
target = "Thanks a lot !"
assert translator.translate(source) == target