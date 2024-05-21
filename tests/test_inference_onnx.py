import unittest
from os import cpu_count

import onnxruntime as ort
from onnxruntime_extensions import get_library_path

PATH_TO_ONNX_MODEL = "models/model.onnx"
PATH_TO_TF_MODEL = "models/universal-sentence-encoder-multilingual-large-3"


def load_onnx_model(model_filepath):
    _options = ort.SessionOptions()
    _options.inter_op_num_threads, _options.intra_op_num_threads = (
        cpu_count(),
        cpu_count(),
    )
    _options.register_custom_ops_library(get_library_path())
    _providers = ["CPUExecutionProvider"]  # could use ort.get_available_providers()
    return ort.InferenceSession(
        path_or_bytes=model_filepath, sess_options=_options, providers=_providers
    )


model = load_onnx_model(PATH_TO_ONNX_MODEL)

sentence = "Hello, world!"
res = model.run(output_names=["outputs"], input_feed={"inputs": [sentence]})[0]


class TestOutputs(unittest.TestCase):
    def test_with_tf(self):
        import numpy as np
        import tensorflow as tf
        import tensorflow_text  # noqa: F401

        model = tf.saved_model.load(PATH_TO_TF_MODEL)
        res_tf = model(sentence).numpy()

        self.assertTrue(np.allclose(res, res_tf, atol=1e-3))
