import tensorflow as tf
from functools import wrap
from einops import rearrange


def _many(fn):
    @wraps(fn)
    def inner(tensors, pattern, **kwargs):
        return (fn(tensor, pattern, **kwargs) for tensor in tensors)

    return inner

rearrange_many = _many(rearrange)

class CompositionalAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        dim,
        dim_head=64,
        num_searches=8,
        num_retrievals=2,
        dropout=0.0,
        prenorm=False,
        causal=False,
        **kwargs
    ):
        super(CompositionalAttention, self).__init__(**kwargs)
