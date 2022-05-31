from functools import wraps

import tensorflow as tf
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
        self.norm = None
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

    def call(
        self,
        inputs,
        mask=None,
    ):
        """
        This follows the same einstein notation as the publically availaible PyTorch implementation (https://github.com/lucidrains/compositional-attention-pytorch):
        b - batch
        n - sequence dimension
        i - sequence dimension (source)
        j - sequence dimension (target, aggregation dimension)
        s - number of searches
        r - number of retrievals
        d - feature dimension
        """
        if self.norm:
            inputs = self.norm(inputs)

        s = self.num_searches
        r = self.num_retrievals

        sq, sk = self.to_searches_queries(inputs), self.to_searches_keys(inputs)
        sq, sk = rearrange_many((sq, sk), "b n (s d) -> b s n d", s=s)

        sq = sq * self.scale

        search_sim = tf.einsum("b s i d, b s j d -> b s i j", sq, sk)

        if mask is not None:
            mask = rearrange(mask, "b j -> b 1 1 j")
            search_sim = tf.where(
                tf.math.logical_not(mask),
                -tf.experimental.numpy.finfo(search_sim.dtype).max,
                search_sim,
            )

        if self.causal:
            i, j = tf.shape(search_sim)[-2:]
            causal_mask = tf.linalg.band_part(
                tf.ones([i, j], dtype=tf.dtypes.bool), 0, j - i
            )
            causal_mask = tf.linalg.set_diag(
                causal_mask, tf.zeros(tf.shape(causal_mask)[0:-1])
            )
            search_sim = tf.where(
                causal_mask,
                -tf.experimental.numpy.finfo(search_sim.dtype).max,
                search_sim,
            )

        search_attn = search_sim - tf.experimental.numpy.amax(
            search_sim, axis=-1, keepdims=True
        )
        search_attn = tf.nn.softmax(search_attn, axis=-1)
        search_attn = self.search_dropout(search_attn)

        rv = self.to_retrieval_values(inputs)
        rv = rearrange(rv, "b n (r d) -> b r n d", r=r)

        retrieved = tf.einsum("b s i j, b r j d -> b s r i d", search_attn, rv)

        rq, rk = self.to_retrieval_queries(inputs), self.to_retrieval_keys(retrieved)
        rq = rearrange(rq, "b n (s d) -> b s n d", s=s)
        rq = rq * self.scale

        retrieval_sim = tf.einsum("b s n d , b s r n d -> b s n r", rq, rk)

        retrieval_attn = retrieval_sim - tf.experimental.numpy.amax(
            retrieval_sim, axis=-1, keepdims=True
        )
        retrieval_attn = tf.nn.softmax(retrieval_attn, axis=-1)
        retrieval_attn = self.retrieval_dropout(retrieval_attn)

        out = tf.einsum("b s n r, b s r n d -> b s n d", retrieval_attn, retrieved)

        out = rearrange(out, "b s n d -> b n (s d)")
        return self.to_out(out)
