from typing import Any, Optional, Tuple
import math
from abc import ABC
from transformers.file_utils import ModelOutput
import copy
import torch
from torch import nn, Tensor, device
from dataclasses import dataclass

from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel

try:
    from transformers.modeling_utils import apply_chunking_to_forward
except ImportError:
    from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.bert.modeling_bert import BertEmbeddings, BertLayer, BertPooler
import torch.nn.functional as F
from einops import rearrange, repeat

POS_MODEL = {'text': 0, 'visual': 1, 'center': 2}


# some function
def get_extended_attention_mask(attention_mask: Tensor, input_shape: Tuple[int], device: device) -> Tensor:
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (:obj:`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (:obj:`Tuple[int]`):
            The shape of the input to the model.
        device: (:obj:`torch.device`):
            The device of the input to the model.

    Returns:
        :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=torch.long)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


def get_head_mask(
        head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
) -> Tensor:
    """
    Prepare the head mask if needed.

    Args:
        head_mask (:obj:`torch.Tensor` with shape :obj:`[num_heads]` or :obj:`[num_hidden_layers x num_heads]`, `optional`):
            The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
        num_hidden_layers (:obj:`int`):
            The number of hidden layers in the model.
        is_attention_chunked: (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the attentions scores are computed by chunks or not.

    Returns:
        :obj:`torch.Tensor` with shape :obj:`[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or
        list with :obj:`[None]` for each layer.
    """
    head_mask = [None] * num_hidden_layers

    return head_mask


# models
class IFAConfig(PretrainedConfig):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class IFAPreTrainedModel(PreTrainedModel):
    config_class = IFAConfig
    base_model_prefix = "clip"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init_weights(self, module):
        pass


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size  # 768
        self.image_size = config.image_size  # 224
        self.patch_size = config.patch_size  # 32

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

        self.pixel_position_embedding = nn.Embedding(49, self.embed_dim)
        self.register_buffer("pixel_position_ids", torch.arange(49).expand((1, -1)))

        self.aux_position_embedding = nn.Embedding(48, self.embed_dim)
        self.register_buffer("aux_position_ids", torch.arange(48).expand((1, -1)))

        self.rcnn_position_embedding = nn.Embedding(12, self.embed_dim)
        self.register_buffer("rcnn_position_ids", torch.arange(12).expand((1, -1)))

    def forward(self, pixel_values, aux_embeddings=None, rcnn_embeddings=None):
        batch_size = pixel_values.shape[0]

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = class_embeds

        pixel_values = self.patch_embedding(pixel_values)
        pixel_values = pixel_values.flatten(2).transpose(1, 2)
        pixel_values = pixel_values + self.pixel_position_embedding(self.pixel_position_ids)
        embeddings = torch.cat((embeddings, pixel_values), dim=1)

        if aux_embeddings is not None:
            aux_embeds = []
            for aux_embedding in aux_embeddings:
                aux_embed = self.patch_embedding(aux_embedding)
                aux_embed = aux_embed.flatten(2).transpose(1, 2).flatten(0, 1)  # 3*16, 768 3个子图
                aux_embeds.append(aux_embed)
            aux_embeds = torch.stack(aux_embeds)  # bsz, 48, 768
            aux_embeds = aux_embeds + self.aux_position_embedding(self.aux_position_ids)
            embeddings = torch.cat((embeddings, aux_embeds), dim=1)

        if rcnn_embeddings is not None:
            rcnn_embeds = []
            for rcnn_embedding in rcnn_embeddings:
                rcnn_embed = self.patch_embedding(rcnn_embedding)
                rcnn_embed = rcnn_embed.flatten(2).transpose(1, 2).flatten(0, 1)  # 3*4, 768 3个子图
                rcnn_embeds.append(rcnn_embed)
            rcnn_embeds = torch.stack(rcnn_embeds)  # bsz, 12, 768
            rcnn_embeds = rcnn_embeds + self.rcnn_position_embedding(self.rcnn_position_ids)
            embeddings = torch.cat((embeddings, rcnn_embeds), dim=1)
        return embeddings


class BertEmbeddings_my(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(
            self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            output_attentions: bool = False,
            past_key_values: torch.Tensor = None,
            current_layer: int = None,
            output_qks=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        qks = (key_states, value_states) if output_qks else None

        if past_key_values is not None:
            key_states = torch.cat([past_key_values[0], key_states], dim=2)
            value_states = torch.cat([past_key_values[1], value_states], dim=2)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz)

        query_states = query_states.view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        key_states = self._shape(self.k_proj(attn_output), -1, bsz)
        value_states = self._shape(self.v_proj(attn_output), -1, bsz)
        qks = (key_states, value_states) if output_qks else None

        return attn_output, attn_weights_reshaped, qks


def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


class CLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = quick_gelu
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads  # 12
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)  # 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 768

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
            visual_hidden_state=None,
            output_qks=None,
            current_layer=None,
            past_key_values=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        qks = (key_layer, value_layer) if output_qks else None

        if past_key_values is not None:
            key_layer = torch.cat([past_key_values[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_values[0], value_layer], dim=2)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            bsz, nheads, length, dsize = past_key_values[0].size()
            visual_attention_mask = torch.ones((bsz, 1, 1, length)).to(attention_mask.device)  # bsz, 12, len, 64
            visual_attention_mask = (1.0 - visual_attention_mask) * -10000.0
            attention_mask = torch.cat((visual_attention_mask, attention_mask), dim=-1)
            attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # bsz, 128, 768

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs, qks


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
            visual_hidden_state=None,
            output_qks=None,
            current_layer=None,
            past_key_values=None,
    ):
        self_outputs, qks = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
            visual_hidden_state,
            output_qks,
            current_layer,
            past_key_values,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them

        return outputs, qks


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            output_attentions: bool = False,
            past_key_values: torch.Tensor = None,
            current_layer: int = None,
            output_qks=None
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape :obj:`(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                :obj:`(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                :obj:`(config.encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights, qks = self.self_attn(
            hidden_states=hidden_states,
            output_attentions=output_attentions,
            past_key_values=past_key_values,
            output_qks=output_qks,
            current_layer=current_layer,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        if output_qks:
            outputs += (qks,)

        return outputs


class BertLayer_slefdefine(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.add_cross_attention = config.add_cross_attention
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        self.num_attention_heads = config.num_attention_heads  # 12
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)  # 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
            visual_hidden_state=None,
            output_qks=None,
            current_layer=None,
            past_key_values=None,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs, qks = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            visual_hidden_state=visual_hidden_state,
            output_qks=output_qks,
            current_layer=current_layer,
            past_key_values=past_key_values,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        key_layer = self.transpose_for_scores(self.key(outputs[0]))
        value_layer = self.transpose_for_scores(self.value(outputs[0]))

        qks = (key_layer, value_layer) if output_qks else None

        if output_qks:
            outputs += (qks,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


@dataclass
class BottleneckEncoderOutput(ModelOutput):
    '''
    Base class for BottleneckEncoder's outputs.

    Args:
        last_hidden_state (:obj:`tuple(torch.FloatTensor)`, `optional`):
            Tuple of :obj:`torch.FloatTensor` (the output of each modality)
            of shape :obj:`(batch_siize, sequence_len, hidden_size)

        last_bottlen_state (:obj:`tuple(torch.FloatTensor)`, `optional`):
            Tuple of :obj:`torch.FloatTensor` (three for the output of each
                bottleunit for each modality, one for the output of center bottle unit)
            of shape :obj:`(batch_size, unit_num, hidden_size)`.

        hidden_states (:obj:`tuple(tuple(torch.FloatTensor))`, `optional):
            Tuple of :obj:`tuple(torch.FloatTensor)` (three for the output of each modality, three for the bottle unit
                output correspones to each modality, one for the output of center bottle unit)
                of length :obj:`max(config.n_layers, visual_config.n_layers, audio_config.n_layers)`
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.

        attentions (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tuple(torch.FloatTensor)` (attentions weights after the attention softmax for each layer),
            with each tuple having 3 tensors of shape :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

        cross_attentions (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_attentions=True`` and ``config.add_cross_attention=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tuple(torch.FloatTensor)` (cross_attentions weights after the attention softmax for the bottle layer),
            with each tuple having 6 tensors (each modality hidden state as value, center unit hidden state as query,
            and each modality hidden state as query, center unit hidden stats as value) of
            of shape :obj:`(batch_size, num_heads, bottle_unit_num, center_unit_num)

    '''

    last_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    last_bottle_state: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


class Bottleneck_Attention(nn.Module):
    def __init__(self, vision_config, text_config):
        super().__init__()
        self.vision_config = vision_config
        self.text_config = text_config
        self.unit_num_t = 16

        self.vision_layers = nn.ModuleList(
            [CLIPEncoderLayer(vision_config) for _ in range(vision_config.num_hidden_layers)])
        self.text_layer = nn.ModuleList([BertLayer(text_config) for _ in range(text_config.num_hidden_layers)])

        self.cross_t = nn.ModuleList([BottleNet(768, 768), BottleNet(768, 768)])
        self.cross_v = nn.ModuleList([BottleNet(768, 768), BottleNet(768, 768)])

        self.bottle_start = (4, 4)  # type(common_config.bottle_start) = Tuple
        self.bottle_end = (8, 8)  # type(common_config.bottle_end) = Tuple

    def forward(
            self,
            hidden_states: Tuple,
            bottle_states: Tuple,
            add_bottleneck=False,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            cross_attention=False,
            concat_tensor=None
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None

        device = hidden_states[0].device
        raw_seq_len = ()
        for i in range(2):
            raw_seq_len = raw_seq_len + (hidden_states[i].size()[1],)
        # raw_seq_len = [hidden_states[i].size()[1] for i in range(3)]

        t_start, v_start = self.bottle_start
        t_end, v_end = self.bottle_end

        def earlate_process(module, start, end, mode, **kargs):
            pos = POS_MODEL[mode]
            input_hidden_states = hidden_states[pos]
            if start >= end:
                return input_hidden_states
            for i in range(start, end):
                layer_module = module[i]
                output = layer_module(input_hidden_states, **kargs)
                input_hidden_states = output[0]
            return input_hidden_states

        early_output_t = earlate_process(self.text_layer, 0, t_start, 'text', attention_mask=attention_mask,
                                         head_mask=head_mask, encoder_attention_mask=encoder_attention_mask,
                                         encoder_hidden_states=encoder_hidden_states, past_key_value=past_key_values,
                                         output_attentions=output_attentions)
        early_output_v = earlate_process(self.vision_layers, 0, v_start, 'visual', output_attentions=output_attentions,
                                         past_key_values=past_key_values, output_qks=None, )

        ## attention_mask not None说明需要进行attention mask，在后面直接拼0，表示对attention bottleneck部分不进行mask
        if attention_mask is not None:
            mask_shape = attention_mask.size()[:-1] + (self.unit_num_t,)
            bottle_mask_t = torch.zeros(mask_shape, device=device)
            # bottle_mask_v = torch.zeros(bottle_states[POS_MODEL['visual']].shape, device=device)
            # bottle_mask_a = torch.zeros(bottle_states[POS_MODEL['audio']].shape, device=device)

            mask_t = torch.cat([attention_mask, bottle_mask_t], dim=-1)
            # mask_v = torch.cat([attention_mask[POS_MODEL['visual']], bottle_mask_v], dim = -2)
            # mask_a = torch.cat([attention_mask[POS_MODEL['audio']], bottle_mask_a], dim = -2)

            # attention_mask = (mask_t, mask_v, mask_a)
            extend_attention_mask = mask_t

        hidden_states = (early_output_t, early_output_v)

        if add_bottleneck:
            for i in range(0, t_end - t_start):

                layer_module_t = self.text_layer[i + t_start]
                layer_module_v = self.vision_layers[i + v_start]

                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_head_mask = head_mask[i] if head_mask is not None else None

                ## concate hidden_state and bottle_state, attention_mask is generated in outer function
                hidden_states_t = torch.cat([hidden_states[POS_MODEL['text']], bottle_states[POS_MODEL['text']]],
                                            dim=-2)
                hidden_states_v = torch.cat([hidden_states[POS_MODEL['visual']], bottle_states[POS_MODEL['visual']]],
                                            dim=-2)

                layer_outputs = layer_module_t(
                    hidden_states_t,
                    extend_attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_values,
                    output_attentions,
                )

                hidden_states_t, bottle_states_t = layer_outputs[0][:, :raw_seq_len[POS_MODEL['text']], :], \
                    layer_outputs[0][:, raw_seq_len[POS_MODEL['text']]:, :]

                visual_layer_outputs = layer_module_v(hidden_states_v)[0]
                hidden_states_v, bottle_states_v = visual_layer_outputs[:, :raw_seq_len[POS_MODEL['visual']],
                                                   :], visual_layer_outputs[:, raw_seq_len[POS_MODEL['visual']]:, :]

                hidden_states = (hidden_states_t, hidden_states_v)

                ## cross attention
                bottle_states_center = bottle_states[POS_MODEL['center']]
                bottle_states_center = self.cross_t[0](bottle_states_center, bottle_states_t)
                bottle_states_center = self.cross_v[0](bottle_states_center, bottle_states_v)

                bottle_states_t = self.cross_t[1](bottle_states_t, bottle_states_center)
                bottle_states_v = self.cross_v[1](bottle_states_v, bottle_states_center)

                bottle_states = (bottle_states_t, bottle_states_v, bottle_states_center)

        else:
            t_end, v_end = t_start, v_start
        hidden_states_t = earlate_process(self.text_layer, t_end, 12, 'text',
                                          attention_mask=attention_mask,
                                          head_mask=head_mask, encoder_attention_mask=encoder_attention_mask,
                                          encoder_hidden_states=encoder_hidden_states, past_key_value=past_key_values,
                                          output_attentions=output_attentions)
        hidden_states_v = earlate_process(self.vision_layers, v_end, 12, 'visual')

        hidden_states = (hidden_states_t, hidden_states_v)

        '''
        处理完成后,各模态的处理结果在hidden_states(Tuple)中,bottle unit的结果在bottle_states(Tuple)中
        '''

        #     if use_cache:
        #         next_decoder_cache += (layer_outputs[-1],)
        #     if output_attentions:
        #         all_self_attentions = all_self_attentions + (layer_outputs[1],)
        #         if self.config.add_cross_attention:
        #             all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # if output_hidden_states:
        #     all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    bottle_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BottleneckEncoderOutput(
            last_hidden_state=hidden_states,
            last_bottle_state=bottle_states,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )


class BertPooler_selfdefine(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # first_token_tensor = hidden_states[:, 0]
        avarage_token_tensor = torch.mean(hidden_states, dim=1)
        pooled_output = self.dense(avarage_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class VisionPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class _AbstractCoAttentionModule(nn.Module):
    side = None

    def forward(self, lang, obj, lang_mask, obj_mask, co_mask):
        raise NotImplementedError


class _AbstractSAModule(_AbstractCoAttentionModule):
    def __init__(self, dim, dim_ff, n_head, msa_dropout, ffn_dropout):
        super().__init__()
        self.dim = dim
        self.msa = MultiheadAttn(dim, n_head, dropout=msa_dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(dim_ff, dim)
        )

    def _forward(self, q, k, v, mask, add_q=True):
        msa = self.msa(q, k, v, mask)
        if add_q:
            x = self.norm1(q + msa)
        else:
            x = self.norm1(v + msa)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x


class LangSaModule(_AbstractSAModule):
    side = 'lang'

    def forward(self, lang, obj, lang_mask, obj_mask, co_mask):
        x = lang
        mask = lang_mask
        return self._forward(x, x, x, mask)


class ObjSaModule(_AbstractSAModule):
    side = 'obj'

    def forward(self, lang, obj, lang_mask, obj_mask, co_mask):
        x = obj
        mask = obj_mask
        return self._forward(x, x, x, mask)


class LangGaModule(_AbstractSAModule):
    side = 'lang'

    def forward(self, lang, obj, lang_mask, obj_mask, co_mask):
        x_value = lang
        x_guide = obj
        return self._forward(x_value, x_guide, x_guide, obj_mask)


class ObjGaModule(_AbstractSAModule):
    side = 'obj'

    def forward(self, lang, obj, lang_mask, obj_mask, co_mask):
        x_value = obj
        x_guide = lang
        return self._forward(x_value, x_guide, x_guide, lang_mask)


class ObjEdgeGaModule(_AbstractCoAttentionModule):
    side = 'obj'

    def __init__(self, dim, dim_ff, n_head, msa_dropout, ffn_dropout):
        super().__init__()
        self.dim = dim
        self.msa_edge = MultiheadAttn(dim, n_head, dropout=msa_dropout)
        self.msa_node = MultiheadAttn(dim, n_head, dropout=msa_dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.norme = nn.LayerNorm(dim)
        self.normq = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim * 2, dim_ff),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(dim_ff, dim)
        )

    def _forward(self, q, node, edge):
        xe = self.norme(q + edge)
        xn = self.normq(q + node)
        x = torch.cat((xn, xe), -1)
        x = xn + self.ffn(x)
        x = self.norm2(x)
        return x

    def _forward_edge(self, lang, obj, lang_mask, obj_mask, co_mask):
        x_value = obj  # N nobj ndim
        x_value = x_value.unsqueeze(2)
        x_expand_t = x_value.transpose(1, 2)  # N 1 nobj ndim
        x_value = x_value - x_expand_t  # N nobj nobj ndim

        N, nobj, _, ndim = x_value.shape
        x_value = x_value.transpose(0, 1).reshape(-1, nobj, ndim)

        x_guide = lang.repeat(nobj, 1, 1)

        mask = lang_mask.repeat(nobj, 1, 1, 1)
        edge = self.msa_edge(x_value, x_guide, x_guide, mask)
        edge = edge.max(1)[0]
        edge = edge.reshape(nobj, N, ndim).transpose(0, 1)
        return edge

    def _forward_node(self, lang, obj, lang_mask, obj_mask, co_mask):
        x_value = obj
        x_guide = lang
        return self.msa_node(x_value, x_guide, x_guide, lang_mask)

    def forward(self, lang, obj, *args):
        node = self._forward_node(lang, obj, *args)
        edge = self._forward_edge(lang, obj, *args)
        return self._forward(obj, node, edge)


class MultiheadAttn(nn.Module):
    def __init__(self, dim, nhead, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.nhead = nhead
        self.head_dim = int(dim // nhead)
        assert self.nhead * self.head_dim == self.dim
        self.linears = nn.ModuleList([nn.Linear(dim, dim) for _ in range(4)])
        self.linear_q = nn.Linear(dim, dim)
        if dropout == 0:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(dropout)

    def attention(self, queries, keys, values, mask=None, dropout=None):
        """
            queries: B x H x S x headdim
            keys: B x H x L x headdim
            values: B x H x L x headdim
            mask: B x 1 x S x L
        """
        headdim = queries.size(-1)
        scores = queries @ keys.transpose(-1, -2) / math.sqrt(headdim)
        if mask is not None:
            scores = scores + mask
            # scores = scores.masked_fill(mask == 0, -1e9)
        scores = torch.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        return scores @ values

    def forward(self, query, key, value, mask=None, sum_seq=False):
        """
            query: B x S x D
            key: B x L x D
            value: B x L x D
            mask: B x S x L
        """
        batch_size = query.size(0)
        # if mask is not None:
        #     mask = mask.unsqueeze(1)

        # device=query.device
        # quires=self.linear_q(query)
        # quires=quires.view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        queries, keys, values = [
            layer(x).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
            for layer, x in zip(self.linears[:3], (query, key, value))
        ]
        result = self.attention(queries, keys, values, mask, self.dropout)
        if sum_seq:
            result = result.sum(2, keepdim=True)
        result = result.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        return self.linears[-1](result)


class CoAttentionLayer(_AbstractCoAttentionModule):
    def __init__(self, lang_layer, obj_layer):
        super().__init__()
        self.obj_layer = obj_layer
        self.lang_layer = lang_layer
        assert self.obj_layer is None or self.obj_layer.side == 'obj'
        assert self.lang_layer is None or self.lang_layer.side == 'lang'

    def forward(self, *args):
        lang, obj = args[:2]
        if self.obj_layer:
            obj = self.obj_layer(*args)
        if self.lang_layer:
            lang = self.lang_layer(*args)
        return lang, obj


class CrossAttention(nn.Module):
    def __init__(self, query_dim, value_dim, heads=4, dim_head=64, attn_dropout=0.1):
        super().__init__()
        hidden_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.q_proj = nn.Linear(query_dim, hidden_dim, bias=False)
        self.kv_proj = nn.Linear(value_dim, hidden_dim * 2, bias=False)

        self.dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(hidden_dim, query_dim)

    def forward(self, query, value=None, mask=None):
        '''
        query.shape = (bsz, seq_len, q_dim)
        key.shape = (bsz, seq_len, k_dim)  value.shape = (bsz, seq_len, v_dim)
        '''
        q = self.q_proj(query)
        value = value if value is not None else query
        k, v = self.kv_proj(value).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=self.heads)
            mask.masked_fill_(mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)

        return self.out_proj(out)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FFN(nn.Module):
    def __init__(self, input_dim, mult=4, dropout=0.1, act_fn=GEGLU()):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim * mult * 2),
            act_fn,
            nn.Linear(input_dim * mult, input_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class BottleNet(nn.Module):
    def __init__(self, query_dim, value_dim, mult=4, **karges) -> None:
        super().__init__()
        self.layerNorm_q = nn.LayerNorm(query_dim)
        self.layerNorm_v = nn.LayerNorm(value_dim)
        self.layerNorm_ffn = nn.LayerNorm(query_dim)
        self.cross = CrossAttention(query_dim, value_dim, **karges)
        self.ffn = FFN(query_dim, mult, dropout=karges.pop('dropout', 0.1), act_fn=karges.pop('act_fn', GEGLU()))

    def forward(self, query, value, mask=None):
        query = self.layerNorm_q(query)
        value = self.layerNorm_v(value)
        cross_input = self.cross(query, value, mask) + query
        ffn_input = self.layerNorm_ffn(cross_input)
        return self.ffn(ffn_input) + ffn_input


class IFAModel(nn.Module):
    def __init__(self, vision_config, text_config, add_pooling_layer=True):
        super(IFAModel, self).__init__()
        # vision model
        self.vision_config = vision_config
        self.vision_embeddings = CLIPVisionEmbeddings(vision_config)
        self.vision_pre_layrnorm = nn.LayerNorm(vision_config.hidden_size)
        self.vision_post_layernorm = nn.LayerNorm(vision_config.hidden_size)
        self.vision_pooler = VisionPooler(text_config) if add_pooling_layer else None

        # text model
        self.text_config = text_config
        self.text_embeddings = BertEmbeddings(text_config)
        self.text_pooler = BertPooler(text_config) if add_pooling_layer else None

        self.bottleneck_t = nn.Parameter(torch.randn(16, 768))
        self.bottleneck_v = nn.Parameter(torch.randn(16, 768))
        self.bottleneck_center = nn.Parameter(torch.randn(16, 768))

        # encoders
        self.encoder = IFAEncoder(vision_config, text_config)

        self.device = vision_config.device
        # self.layers.to(device=self.device)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            use_bottleneck=True,
            pixel_values=None,
            aux_values=None,
            rcnn_values=None,
            encoder_hidden_states=None,
            encoder_extended_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        # pre vision
        vision_embedding_output = self.vision_embeddings(pixel_values, aux_values, rcnn_values)
        vision_embedding_output = self.vision_pre_layrnorm(vision_embedding_output)

        # pre text
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        if token_type_ids is None:
            raise ValueError("token_type_ids is None!")

        extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, input_shape, device)
        head_mask = get_head_mask(head_mask, self.text_config.num_hidden_layers)  # [None]*12

        text_embedding_output = self.text_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        hideden_state = (text_embedding_output, vision_embedding_output)

        bottleneck_t = self.bottleneck_v.repeat(batch_size, 1, 1)
        bottleneck_v = self.bottleneck_v.repeat(batch_size, 1, 1)
        bottleneck_center = self.bottleneck_center.repeat(batch_size, 1, 1)

        bottleneck_state = (bottleneck_t, bottleneck_v, bottleneck_center)

        encoder_outputs = self.encoder(
            hideden_state,
            bottleneck_state,
            add_bottleneck=use_bottleneck,
            attention_mask=extended_attention_mask,
            head_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        text_sequence_output, vision_sequence_output = encoder_outputs[0]
        bottleneck_t, bottleneck_v, bottleneck_center = encoder_outputs[1]

        text_pooled_output = self.text_pooler(text_sequence_output) if self.text_pooler is not None else None
        vision_pooled_output = torch.mean(vision_sequence_output, dim=1)
        vision_pooled_output = self.vision_pooler(vision_pooled_output) if self.vision_pooler is not None else None

        # if not return_dict:
        #     return (text_sequence_output, text_pooled_output,vision_sequence_output,vision_pooled_output,text_pooled_output_0,vision_pooled_output_0) + encoder_outputs[1:]
        if not return_dict:
            return (text_sequence_output, text_pooled_output, vision_sequence_output, vision_pooled_output,
                    text_embedding_output) + encoder_outputs[2:]

        return BaseModelOutputWithPooling(
            last_hidden_state=text_sequence_output,
            pooler_output=text_pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        # ----------type1:当使用基于transformer的层次化交互时  结束----------

        #
        '''
        # ----------type2:只使用全局信息瓶颈约束  开始----------
        text_sequence_output = text_embedding_output
        text_pooled_output = self.text_pooler(text_sequence_output) if self.text_pooler is not None else None

        vision_sequence_output = vision_embedding_output
        vision_pooled_output = torch.mean(vision_sequence_output, dim=1)
        vision_pooled_output = self.vision_pooler(vision_pooled_output) if self.vision_pooler is not None else None


        if not return_dict:
            return (text_sequence_output, text_pooled_output,vision_sequence_output,vision_pooled_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=text_sequence_output,
            pooler_output=text_pooled_output
        )

        # ----------type2:只使用全局信息瓶颈约束  结束----------
        '''

    def _init_text_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.text_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.text_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self):
        return self.text_embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.text_embeddings.word_embeddings = value

    def resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

    def _get_resized_embeddings(
            self, old_embeddings: nn.Embedding, new_num_tokens: Optional[int] = None
    ) -> nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (:obj:`torch.nn.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (:obj:`int`, `optional`):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or :obj:`None`, just returns a pointer to the input tokens
                :obj:`torch.nn.Embedding`` module of the model without doing anything.

        Return:
            :obj:`torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
            :obj:`new_num_tokens` is :obj:`None`
        """
        if new_num_tokens is None:
            return old_embeddings
        else:
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        if old_num_tokens == new_num_tokens:
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}."
                f"You should either use a different resize function or make sure that `old_embeddings` are an instance of {nn.Embedding}."
            )

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim).to(
            self.device, dtype=old_embeddings.weight.dtype
        )

        # initialize all new embeddings (in particular added tokens)
        self._init_text_weights(new_embeddings)

        # Copy token embeddings from the previous weights

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        return new_embeddings