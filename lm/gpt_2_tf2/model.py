"""
Based on https://github.com/openai/gpt-2/blob/master/src/model.py,
converted to TF 2.0 Keras API.
"""
import attr
import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal, Constant


@attr.s(auto_attribs=True, frozen=True)
class HParams:
    n_vocab: int
    n_ctx: int
    n_embed: int
    n_head: int
    n_layer: int


class Model(tf.keras.Model):
    def __init__(self, hparams: HParams, name=None):
        super().__init__(name=name)
        self.hparams = hparams
        self.wpe = self.add_weight(
            'wpe', [hparams.n_ctx, hparams.n_embed],
            initializer=RandomNormal(stddev=0.01))
        self.wte = self.add_weight(
            'wte', [hparams.n_vocab, hparams.n_embed],
            initializer=RandomNormal(stddev=0.02))
        self.blocks = tf.keras.Sequential([
            Block(hparams, name=f'h{i}') for i in range(hparams.n_layer)])
        self.ln_f = Norm()

    def call(self, x, past=None):
        results = {}
        batch, sequence = shape_list(x)
        past_length = 0 if past is None else tf.shape(input=past)[-2]
        h = (tf.gather(self.wte, x) +
             tf.gather(self.wpe, positions_for(x, past_length)))

        # Transformer
        presents = []
        pasts = (tf.unstack(past, axis=1) if past is not None else
                 [None] * self.hparams.n_layer)
        assert len(pasts) == self.hparams.n_layer
        for i, past in enumerate(pasts):
            h, present = self.blocks.layers[i](h, past=past)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        h = self.ln_f(h)

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch * sequence, self.hparams.n_embed])
        logits = tf.matmul(h_flat, self.wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, self.hparams.n_vocab])
        results['logits'] = logits
        return results


class Block(tf.keras.layers.Layer):
    def __init__(self, hparams: HParams, name=None):
        super().__init__(name=name)
        self.hparams = hparams

    def build(self, input_shape):
        nx = input_shape[-1]
        self.ln_1 = Norm()
        self.ln_2 = Norm()
        self.mlp = MLP(nx * 4)
        self.attn = Attention(nx, hparams=self.hparams)

    def call(self, x, past):
        a, present = self.attn(self.ln_1(x), past=past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present


class Norm(tf.keras.layers.Layer):
    """ Normalize to mean = 0, std = 1, then do a diagonal affine transform.
    """
    def __init__(self, *, axis=-1, epsilon=1e-5, name=None):
        super().__init__(name=name)
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        n_state = input_shape[-1]
        self.g = self.add_weight(
            'g', [n_state], tf.float32, initializer=Constant(1))
        self.b = self.add_weight(
            'b', [n_state], dtype=tf.float32, initializer=Constant(0))

    def call(self, x):
        u = tf.reduce_mean(input_tensor=x, axis=self.axis, keepdims=True)
        s = tf.reduce_mean(input_tensor=tf.square(x-u), axis=self.axis,
                           keepdims=True)
        x = (x - u) * tf.math.rsqrt(s + self.epsilon)
        x = x * self.g + self.b
        return x


class MLP(tf.keras.layers.Layer):
    def __init__(self, n_state: int, name=None):
        super().__init__(name=name)
        self.n_state = n_state

    def build(self, input_shape):
        nx = input_shape[-1]
        self.c_fc = Conv1D(self.n_state)
        self.c_proj = Conv1D(nx)

    def call(self, x):
        h = gelu(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2


class Conv1D(tf.keras.layers.Layer):
    def __init__(self, n_state: int, w_init_stdev=0.02, name=None):
        super().__init__(name=name)
        self.n_state = n_state
        self.w_init_stdev = w_init_stdev

    def build(self, input_shape):
        nx = input_shape[-1]
        self.w = self.add_weight(
            'w', [1, nx, self.n_state],
            initializer=RandomNormal(stddev=self.w_init_stdev))
        self.b = self.add_weight(
            'b', [self.n_state], tf.float32, initializer=Constant(0))

    def call(self, x):
        *start, nx = shape_list(x)
        return tf.reshape(
            tf.matmul(tf.reshape(x, [-1, nx]),
                      tf.reshape(self.w, [-1, self.n_state])) + self.b,
            start + [self.n_state])


class Attention(tf.keras.layers.Layer):
    def __init__(self, n_state: int, hparams: HParams, name=None):
        super().__init__(name=name)
        self.hparams = hparams
        self.c_attn = Conv1D(n_state * 3)
        self.c_proj = Conv1D(n_state)
        assert n_state % hparams.n_head == 0

    def call(self, x, past):
        assert x.shape.ndims == 3  # Should be [batch, sequence, features]
        if past is not None:
            # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]
            assert past.shape.ndims == 5
        c = self.c_attn(x)
        q, k, v = map(self.split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = self.multihead_attn(q, k, v)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present

    def split_heads(self, x):
        """ From [batch, sequence, features] to
        [batch, heads, sequence, features].
        """
        return tf.transpose(
            a=split_states(x, self.hparams.n_head), perm=[0, 2, 1, 3])

    def merge_heads(self, x):
        """ Reverse of split_heads.
        """
        return merge_states(tf.transpose(a=x, perm=[0, 2, 1, 3]))

    def mask_attn_weights(self, w):
        # w has shape [batch, heads, dst_sequence, src_sequence],
        # where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    def multihead_attn(self, q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.math.rsqrt(tf.cast(v.shape[-1], w.dtype))

        w = self.mask_attn_weights(w)
        w = softmax(w)
        a = tf.matmul(w, v)
        return a


def shape_list(x):
    """ Deal with dynamic shape in tensorflow cleanly.
    """
    # TODO review usages, maybe something easier will do
    static = x.shape.as_list()
    dynamic = tf.shape(input=x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def softmax(x, axis=-1):
    x = x - tf.reduce_max(input_tensor=x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(input_tensor=ex, axis=axis, keepdims=True)


def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))


def split_states(x, n):
    """ Reshape the last dimension of x into [n, x.shape[-1]/n].
    """
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])


def merge_states(x):
    """ Smash the last two dimensions of x into a single dimension.
    """
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])


def attention_mask(nd, ns, *, dtype):
    """ 1's in the lower triangle, counting from the lower right corner.
    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd),
    but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def expand_tile(value, size):
    """Add a new axis of given size.
    """
    value = tf.convert_to_tensor(value=value, name='value')
    n_dims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1] * n_dims)


def positions_for(tokens, past_length):
    batch_size = tf.shape(input=tokens)[0]
    n_steps = tf.shape(input=tokens)[1]
    return expand_tile(past_length + tf.range(n_steps), batch_size)


