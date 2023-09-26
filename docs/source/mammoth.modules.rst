Modules
=============

Core Modules
------------

.. autoclass:: mammoth.modules.Embeddings
    :members:


Encoders
---------

.. autoclass:: mammoth.encoders.EncoderBase
    :members:

.. autoclass:: mammoth.encoders.MeanEncoder
    :members:

.. autoclass:: mammoth.encoders.RNNEncoder
    :members:


Decoders
---------


.. autoclass:: mammoth.decoders.DecoderBase
    :members:
    
.. autoclass:: mammoth.decoders.decoder.RNNDecoderBase
    :members:

.. autoclass:: mammoth.decoders.StdRNNDecoder
    :members:

.. autoclass:: mammoth.decoders.InputFeedRNNDecoder
    :members:

Attention
----------

.. autoclass:: mammoth.modules.AverageAttention
    :members:

.. autoclass:: mammoth.modules.GlobalAttention
    :members:



Architecture: Transformer
----------------------------

.. autoclass:: mammoth.modules.PositionalEncoding
    :members:

.. autoclass:: mammoth.modules.position_ffn.PositionwiseFeedForward
    :members:

.. autoclass:: mammoth.encoders.TransformerEncoder
    :members:

.. autoclass:: mammoth.decoders.TransformerDecoder
    :members:

.. autoclass:: mammoth.modules.MultiHeadedAttention
    :members:
    :undoc-members:


Architecture: Conv2Conv
----------------------------

(These methods are from a user contribution
and have not been thoroughly tested.)


.. autoclass:: mammoth.encoders.CNNEncoder
    :members:


.. autoclass:: mammoth.decoders.CNNDecoder
    :members:

.. autoclass:: mammoth.modules.ConvMultiStepAttention
    :members:

.. autoclass:: mammoth.modules.WeightNormConv2d
    :members:

Architecture: SRU
----------------------------

.. autoclass:: mammoth.models.sru.SRU
    :members:


Copy Attention
--------------

.. autoclass:: mammoth.modules.CopyGenerator
    :members:


Structured Attention
-------------------------------------------

.. autoclass:: mammoth.modules.structured_attention.MatrixTree
    :members:
