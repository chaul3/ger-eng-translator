import json
import os

tf_config = {
    "cluster": {"worker": ["34.91.129.32:12345", "34.141.202.51:23456"]},
    "task": {"type": "worker", "index": 0},
}

os.environ["TF_CONFIG"] = json.dumps(tf_config)

print('An example TF_CONFIG')
print(json.dumps(tf_config))
