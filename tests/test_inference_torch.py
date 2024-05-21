import sys
import unittest
from functools import partial

import torch

sys.path.append("..")
from src.architecture import MUSE  # noqa: E402
from src.tokenizer import get_tokenizer, tokenize  # noqa: E402

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
model_torch.load_state_dict(torch.load(PATH_TO_PT_MODEL))

sentence = "Hello, world!"
res = model_torch(tokenize(sentence))  # type: ignore


class TestOutputs(unittest.TestCase):
    def test_with_tf(self):
        import numpy as np
        import tensorflow as tf
        import tensorflow_text  # noqa: F401

        model = tf.saved_model.load(PATH_TO_TF_MODEL)
        res_tf = model(sentence).numpy()

        self.assertTrue(np.allclose(res.detach().numpy(), res_tf, atol=1e-3))
