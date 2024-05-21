import torch
from tensorflow.core.protobuf.saved_model_pb2 import SavedModel
from tensorflow.python.saved_model.loader_impl import parse_saved_model
from tensorflow_text.python.ops.sentencepiece_tokenizer import SentencepieceTokenizer


def _get_tokenizer_from_saved_model(saved_model: SavedModel) -> SentencepieceTokenizer:
    """
    Get tokenizer from tf SavedModel.
    :param SavedModel saved_model: tf SavedModel.
    :return: tokenizer.
    :rtype: SentencepieceTokenizer
    """

    # extract functions that contain SentencePiece somewhere in there
    functions_with_sp = [
        f
        for f in saved_model.meta_graphs[0].graph_def.library.function
        if "tokenizer" in str(f).lower()
    ]

    assert (
        len(functions_with_sp) == 1
    ), f"len(functions_with_sp) = {len(functions_with_sp)}"

    # find SentencePieceOp (contains the model) in the found function
    nodes_with_sp = [
        n for n in functions_with_sp[0].node_def if n.op == "SentencepieceOp"
    ]

    assert len(nodes_with_sp) == 1, f"len(nodes_with_sp) = {len(nodes_with_sp)}"

    # we can pretty much save the model into a file since it does not change
    model = nodes_with_sp[0].attr["model"].s

    # instantiate the model
    tokenizer = SentencepieceTokenizer(model)

    return tokenizer


def get_tokenizer(model_path: str) -> SentencepieceTokenizer:
    tokenizer = _get_tokenizer_from_saved_model(parse_saved_model(model_path))
    return tokenizer


def tokenize(
    sentence: str,  # TODO: add batch processing
    tokenizer: SentencepieceTokenizer,
) -> torch.Tensor:
    return torch.LongTensor([1] + tokenizer.tokenize([sentence]).to_list()[0] + [2])
