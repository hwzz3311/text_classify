import os
import platform
import string
from collections import OrderedDict
from typing import Optional, List, Union, Tuple

import torch
from torch import nn
from transformers import BertConfig
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, \
    BaseModelOutputWithPastAndCrossAttentions

from src.models.modeling_bert import BertModel, BertLayer, BertPooler, BertEmbeddings, logger
from src.models import Bert


class MYBertEncoder(nn.Module):
    def __init__(self, config, bert_layer_nums):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(bert_layer_nums)])
        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
            return_dict: Optional[bool] = True,
            to_layer_num: Optional[int] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            # 跑到指定层之后 就跳出 layer
            if to_layer_num is not None and i >= to_layer_num:
                print(f"encode_forward ： curren layer :{i}, to_layer_num :{to_layer_num} , break")
                break
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        return hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions

    def encode_continue(self,
                        hidden_states: torch.Tensor,
                        next_decoder_cache: torch.Tensor,
                        all_hidden_states: torch.Tensor,
                        all_self_attentions: torch.Tensor,
                        all_cross_attentions: torch.Tensor,
                        from_layer_num: int,
                        to_layer_num: int,

                        attention_mask: Optional[torch.FloatTensor] = None,
                        head_mask: Optional[torch.FloatTensor] = None,
                        encoder_hidden_states: Optional[torch.FloatTensor] = None,
                        encoder_attention_mask: Optional[torch.FloatTensor] = None,
                        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
                        use_cache: Optional[bool] = None,
                        output_attentions: Optional[bool] = False,
                        output_hidden_states: Optional[bool] = False,
                        ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        """

        :param hidden_states: 来自上层bert 的输出
        :param next_decoder_cache: 来自上层bert 的输出
        :param all_hidden_states: 来自上层bert 的输出
        :param all_self_attentions: 来自上层bert 的输出
        :param all_cross_attentions: 来自上层bert 的输出


        :param attention_mask:
        :param head_mask:
        :param encoder_hidden_states:
        :param encoder_attention_mask:
        :param past_key_values:
        :param use_cache:
        :param output_attentions:
        :param output_hidden_states:
        :param return_dict:
        :return:
        """

        # hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions
        for i, layer_module in enumerate(self.layer):
            if i < from_layer_num:
                print(f"encode_continue ： curren layer :{i}, from_layer_num :{from_layer_num} , continue")
                continue
            if i >= to_layer_num:
                print(f"encode_continue ： curren layer :{i}, to_layer_num :{to_layer_num} , break")
                break
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        return hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions


class MYBertModel(BertModel):
    def __init__(self, config, bert_layer_nums, bert_split_dir, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.bert_layer_nums = bert_layer_nums
        self.bert_split_dir = bert_split_dir

        self.embeddings = BertEmbeddings(config)
        self.encoder = MYBertEncoder(config, self.bert_layer_nums)

        # Initialize weights and apply final processing
        self.post_init()

    def post_init(self):
        # 构建一个 本地 的layer和self layer的对应关系
        self.bert_encoder_layer_dict = {}
        # Initialize weights 考虑将 权重的加载放在外面
        weight_index = [index for index in range(self.bert_layer_nums)]

        weight_names = []
        # 添加对应层的名字
        for layer in ['encoder.layer.{}.attention.self.query.weight', 'encoder.layer.{}.attention.self.query.bias',
                      'encoder.layer.{}.attention.self.key.weight', 'encoder.layer.{}.attention.self.key.bias',
                      'encoder.layer.{}.attention.self.value.weight', 'encoder.layer.{}.attention.self.value.bias',
                      'encoder.layer.{}.attention.output.dense.weight', 'encoder.layer.{}.attention.output.dense.bias',
                      'encoder.layer.{}.attention.output.LayerNorm.weight',
                      'encoder.layer.{}.attention.output.LayerNorm.bias', 'encoder.layer.{}.intermediate.dense.weight',
                      'encoder.layer.{}.intermediate.dense.bias', 'encoder.layer.{}.output.dense.weight',
                      'encoder.layer.{}.output.dense.bias', 'encoder.layer.{}.output.LayerNorm.weight',
                      'encoder.layer.{}.output.LayerNorm.bias']:
            for self_bert_encoder_index, saved_bert_encoder_index in enumerate(weight_index):
                self_weight_name = layer.format(self_bert_encoder_index)
                saved_weight_name = layer.format(saved_bert_encoder_index)
                self.bert_encoder_layer_dict[saved_weight_name] = self_weight_name
                weight_names.append(saved_weight_name)
        # 加载对应的文件

        layer_dict = OrderedDict()
        for saved_weight_name in weight_names:
            saved_weight_file = os.path.join(self.bert_split_dir, f"{saved_weight_name}")
            saved_weight = torch.load(saved_weight_file)
            # print(f"mybert torch load weight file: {saved_weight_file}")
            self_weight_name = self.bert_encoder_layer_dict[saved_weight_name]
            layer_dict[self_weight_name] = torch.Tensor(saved_weight)
        for weight_name in ['embeddings.position_ids', 'embeddings.word_embeddings.weight',
                            'embeddings.position_embeddings.weight', 'embeddings.token_type_embeddings.weight',
                            'embeddings.LayerNorm.weight', 'embeddings.LayerNorm.bias']:
            saved_weight_file = os.path.join(self.bert_split_dir, f"{weight_name}")
            saved_weight = torch.load(saved_weight_file)
            # print(f"mybert torch load weight file: {saved_weight_file}")
            layer_dict[weight_name] = torch.Tensor(saved_weight)
        state_dict = self.state_dict(destination=None)
        state_dict.update(layer_dict)
        self.load_state_dict(state_dict)
        layer_dict.clear()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            to_layer_num: Optional[int] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            to_layer_num=to_layer_num
        )
        # extended_attention_mask,
        # head_mask = head_mask,
        # encoder_hidden_states = encoder_hidden_states,
        # encoder_attention_mask = encoder_extended_attention_mask,
        # past_key_values = past_key_values,
        # use_cache = use_cache,
        # output_attentions = output_attentions,
        # output_hidden_states = output_hidden_states,
        hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions = encoder_outputs
        return hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions, \
               extended_attention_mask, head_mask, encoder_hidden_states, \
               encoder_extended_attention_mask, past_key_values, use_cache, output_attentions, output_hidden_states

    def encoder_continue(self, hidden_states: torch.Tensor,
                         next_decoder_cache: torch.Tensor,
                         all_hidden_states: torch.Tensor,
                         all_self_attentions: torch.Tensor,
                         all_cross_attentions: torch.Tensor,
                         from_layer_num: int,
                         to_layer_num: int,

                         extended_attention_mask: Optional[torch.FloatTensor] = None,
                         head_mask: Optional[torch.FloatTensor] = None,
                         encoder_hidden_states: Optional[torch.FloatTensor] = None,
                         encoder_extended_attention_mask: Optional[torch.FloatTensor] = None,
                         past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
                         use_cache: Optional[bool] = None,
                         output_attentions: Optional[bool] = False,
                         output_hidden_states: Optional[bool] = False, ):
        encoder_outputs = self.encoder.encode_continue(hidden_states, next_decoder_cache, all_hidden_states,
                                                       all_self_attentions, all_cross_attentions, from_layer_num,
                                                       to_layer_num, extended_attention_mask, head_mask, encoder_hidden_states,
                                                       encoder_extended_attention_mask, past_key_values, use_cache,
                                                       output_attentions, output_hidden_states)

        hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions = encoder_outputs
        return hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions,\
               extended_attention_mask, head_mask, encoder_hidden_states, \
               encoder_extended_attention_mask, past_key_values, use_cache, output_attentions, output_hidden_states


class MYBertEncoderContinue(nn.Module):
    def __init__(self, config, continue_layer_nums):
        super(MYBertEncoderContinue, self).__init__()
        self.config = config
        # TODO: 注意修改为 12 - N层
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(continue_layer_nums)])
        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states: torch.Tensor,
            next_decoder_cache: torch.Tensor,
            all_hidden_states: torch.Tensor,
            all_self_attentions: torch.Tensor,
            all_cross_attentions: torch.Tensor,

            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
            return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        """

        :param hidden_states: 来自上层bert 的输出
        :param next_decoder_cache: 来自上层bert 的输出
        :param all_hidden_states: 来自上层bert 的输出
        :param all_self_attentions: 来自上层bert 的输出
        :param all_cross_attentions: 来自上层bert 的输出


        :param attention_mask:
        :param head_mask:
        :param encoder_hidden_states:
        :param encoder_attention_mask:
        :param past_key_values:
        :param use_cache:
        :param output_attentions:
        :param output_hidden_states:
        :param return_dict:
        :return:
        """

        # hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class MYBertContinue(nn.Module):
    def __init__(self, config: BertConfig, user_config: Bert.Config, add_pooling_layer=True):
        super(MYBertContinue, self).__init__()
        # super().__init__()
        self.config = config
        self.user_config = user_config
        self.continue_layer_nums = user_config.continue_layer_nums

        self.encoder = MYBertEncoderContinue(config=config, continue_layer_nums=self.continue_layer_nums)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def post_init(self):
        # 构建一个 本地 的layer和self layer的对应关系
        self.bert_encoder_layer_dict = {}
        # Initialize weights 考虑将 权重的加载放在外面
        weight_index = [index for index in
                        range(self.config.num_hidden_layers - self.continue_layer_nums, self.config.num_hidden_layers)]

        weight_names = []
        # 添加对应层的名字
        for layer in ['encoder.layer.{}.attention.self.query.weight', 'encoder.layer.{}.attention.self.query.bias',
                      'encoder.layer.{}.attention.self.key.weight', 'encoder.layer.{}.attention.self.key.bias',
                      'encoder.layer.{}.attention.self.value.weight', 'encoder.layer.{}.attention.self.value.bias',
                      'encoder.layer.{}.attention.output.dense.weight', 'encoder.layer.{}.attention.output.dense.bias',
                      'encoder.layer.{}.attention.output.LayerNorm.weight',
                      'encoder.layer.{}.attention.output.LayerNorm.bias', 'encoder.layer.{}.intermediate.dense.weight',
                      'encoder.layer.{}.intermediate.dense.bias', 'encoder.layer.{}.output.dense.weight',
                      'encoder.layer.{}.output.dense.bias', 'encoder.layer.{}.output.LayerNorm.weight',
                      'encoder.layer.{}.output.LayerNorm.bias']:
            for self_bert_encoder_index, saved_bert_encoder_index in enumerate(weight_index):
                self_weight_name = layer.format(self_bert_encoder_index)
                saved_weight_name = layer.format(saved_bert_encoder_index)
                self.bert_encoder_layer_dict[saved_weight_name] = self_weight_name
                weight_names.append(saved_weight_name)
        # 加载对应的文件

        layer_dict = OrderedDict()
        for saved_weight_name in weight_names:
            saved_weight_file = os.path.join(self.user_config.bert_split_dir, f"{saved_weight_name}")
            saved_weight = torch.load(saved_weight_file)
            self_weight_name = self.bert_encoder_layer_dict[saved_weight_name]
            layer_dict[self_weight_name] = torch.Tensor(saved_weight)
        for weight_name in ["pooler.dense.bias", "pooler.dense.weight"]:
            saved_weight_file = os.path.join(self.user_config.bert_split_dir, f"{weight_name}")
            saved_weight = torch.load(saved_weight_file)
            layer_dict[weight_name] = torch.Tensor(saved_weight)
        state_dict = self.state_dict(destination=None)
        state_dict.update(layer_dict)
        self.load_state_dict(state_dict)
        layer_dict.clear()

    def forward(
            self,
            hidden_states: torch.Tensor,
            next_decoder_cache: torch.Tensor,
            all_hidden_states: torch.Tensor,
            all_self_attentions: torch.Tensor,
            all_cross_attentions: torch.Tensor,
            extended_attention_mask: Optional[torch.Tensor],
            head_mask,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_extended_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        """

        :param hidden_states: 来自于上层bert模型的encoder层的输出
        :param next_decoder_cache: 来自于上层bert模型的encoder层的输出
        :param all_hidden_states: 来自于上层bert模型的encoder层的输出
        :param all_self_attentions: 来自于上层bert模型的encoder层的输出
        :param all_cross_attentions: 来自于上层bert模型的encoder层的输出
        :param extended_attention_mask: 上层bert模型的中间结果
        :param head_mask: 上层bert模型的中间结果
        :param encoder_hidden_states: 上层bert模型的中间结果
        :param encoder_extended_attention_mask: 上层bert模型的中间结果
        :param past_key_values: 上层bert模型参数
        :param use_cache: 上层bert模型参数
        :param output_attentions: 上层bert模型参数
        :param output_hidden_states: 上层bert模型参数
        :param return_dict: 上层bert模型参数
        :return:
        """

        encoder_outputs = self.encoder(hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions,
                                       all_cross_attentions,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask,
                                       encoder_hidden_states=encoder_hidden_states,
                                       encoder_attention_mask=encoder_extended_attention_mask,
                                       past_key_values=past_key_values,
                                       use_cache=use_cache,
                                       output_attentions=output_attentions,
                                       output_hidden_states=output_hidden_states,
                                       return_dict=return_dict,
                                       )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
