import tensorflow as tf
from functools import wrap
from einops import rearrange


def _many(fn):
    @wraps(fn)
    def inner(tensors, pattern, **kwargs):
        return (fn(tensor, pattern, **kwargs) for tensor in tensors)

    return inner

rearrange_many = _many(rearrange)



