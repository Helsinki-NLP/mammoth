Modules
=============

Layer stacks
------------

.. autoclass:: mammoth.modules.layer_stack.StackXcoder
    :members:

.. autoclass:: mammoth.modules.transformer.TransformerStack
    :members:

.. autoclass:: mammoth.modules.transformer.NativeTransformerWrapper
    :members:


Transformer blocks
------------------

.. autoclass:: mammoth.modules.transformer.EncoderBlock
    :members:

.. autoclass:: mammoth.modules.transformer.DecoderBlock
    :members:


Sublayers
---------

.. autoclass:: mammoth.modules.transformer.MultiHeadAttention
    :members:

.. autoclass:: mammoth.modules.transformer.FeedForward
    :members:

.. autoclass:: mammoth.modules.transformer.RotaryEmbedding
    :members:

.. autofunction:: mammoth.modules.transformer.apply_rotary


KV cache
--------

.. autoclass:: mammoth.modules.transformer.LayerCache
    :members:

.. autoclass:: mammoth.modules.transformer.KVCache
    :members:


Attention Bridge
----------------

.. autoclass:: mammoth.modules.attention_bridge.AttentionBridge
    :members:

.. autoclass:: mammoth.modules.attention_bridge.BaseAttentionBridgeLayer
    :members:
