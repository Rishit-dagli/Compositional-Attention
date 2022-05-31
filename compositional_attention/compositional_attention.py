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
        if prenorm:
            self.norm = tf.keras.layers.LayerNormalization(axis=-1)

        self.scale = dim_head**-0.5
        inner_search_dim = dim_head * num_searches
        inner_retrieval_dim = dim_head * num_retrievals

        self.num_searches = num_searches
        self.num_retrievals = num_retrievals

        self.to_searches_queries = tf.keras.layers.Dense(
            inner_search_dim, use_bias=False
        )
        self.to_searches_keys = tf.keras.layers.Dense(inner_search_dim, use_bias=False)
        self.to_retrieval_values = tf.keras.layers.Dense(
            inner_retrieval_dim, use_bias=False
        )

        self.to_retrieval_queries = tf.keras.layers.Dense(
            inner_search_dim, use_bias=False
        )
        self.to_retrieval_keys = tf.keras.layers.Dense(dim_head, use_bias=False)

        self.to_out = tf.keras.layers.Dense(dim, use_bias=False)

        self.search_dropout = tf.keras.layers.Dropout(dropout)
        self.retrieval_dropout = tf.keras.layers.Dropout(dropout)

        self.causal = causal