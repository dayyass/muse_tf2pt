# Convert MUSE from TensorFlow to PyTorch and ONNX

This repository contains code to:
1. Convert the mUSE (Multilingual Universal Sentence Encoder) transformer model from [TF Hub](https://www.kaggle.com/models/google/universal-sentence-encoder/tensorFlow2/multilingual-large) format to **PyTorch** and **ONNX** formats.
1. Use these models using **PyTorch** and **ONNX**.

> [!IMPORTANT]
> **The PyTorch model can be used not only for inference, but also for additional training and fine-tuning!**

# Usage

## ONNX

The model was transferred from TF Hub to **ONNX** using the [tensorflow-onnx](https://github.com/onnx/tensorflow-onnx) library:
```bash
python -m tf2onnx.convert --saved-model models/universal-sentence-encoder-multilingual-large-3 --output models/model.onnx --extra_opset ai.onnx.contrib:1
```

The model is available for download via the [link](https://huggingface.co/dayyass/universal-sentence-encoder-multilingual-large-3-pytorch/tree/main), the inference code is available [here](tests/test_inference_torch.py).

## PyTorch

The transfer of the model from TF Hub to **PyTorch** was carried out through manual work of direct translation of the calculation graph (*you can visualize the ONNX version of the model via [Netron](https://netron.app/)*) to PyTorch. Notebooks [convert.ipynb](convert.ipynb) and [onnx_inference_and_debug.ipynb](onnx_inference_and_debug.ipynb) were used for conversion and debugging.

The model is available in [HF Models](https://huggingface.co/dayyass/universal-sentence-encoder-multilingual-large-3-pytorch/tree/main) directly through `torch` (*currently, without native support from the `transformers` library*).

Model initialization and usage code:
```python
import torch
from functools import partial
from src.architecture import MUSE
from src.tokenizer import get_tokenizer, tokenize

PATH_TO_PT_MODEL = "models/model.pt"
PATH_TO_TF_MODEL = "models/universal-sentence-encoder-multilingual-large-3"

tokenizer = get_tokenizer(PATH_TO_TF_MODEL)
tokenize = partial(tokenize, tokenizer=tokenizer)

model_torch = MUSE(
    num_embeddings=128010,
    embedding_dim=512,
    d_model=512,
    num_heads=8,
)
model_torch.load_state_dict(
    torch.load(PATH_TO_PT_MODEL)
)

sentence = "Hello, world!"
res = model_torch(tokenize(sentence))
```
> [!NOTE]
> Currently, the checkpoint of the original TF Hub model is used for tokenization, so it is loaded in the code above.

## Notes
For the TF Hub to work, the model needs the `tensorflow-text` library, which is available in PyPI except for Apple Silicon, so it can be downloaded [here](https://github.com/sun1638650145/Libraries-and-Extensions-for-TensorFlow-for-Apple-Silicon/releases).

Python >= 3.8.

