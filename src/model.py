# -*- coding: utf-8 -*-

from typing import List, Optional, Tuple, Union

from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import get_peft_model

from torch import nn
import torch


class MultiHeadCLM(nn.Module):  # TODO: Args for model params
    def __init__(
            self, 
            clm_model, 
            context_size=4096, 
            num_labels=4, 
            label_weigths=None,
            hidden_states_idx=-1,
            cls_token_idx=-1,
    ) -> None:
        super().__init__()

        self.clm_model = clm_model

        self.context_size = context_size
        self.num_labels = num_labels
        self.label_weights = label_weigths
        self.hidden_states_idx = hidden_states_idx
        self.cls_token_idx = cls_token_idx

        self.score = nn.Linear(self.context_size, self.num_labels)
        self.loss_fct = nn.CrossEntropyLoss(self.label_weights)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        label_id: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:

        # print("input_ids: ", input_ids.shape, input_ids.dtype)
        # print("attention_mask: ", attention_mask.shape, attention_mask.dtype)

        outputs = self.clm_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # print("outputs.hidden_states[self.hidden_states_idx]: ", outputs['hidden_states'][-1].shape)

        logits = self.score(outputs['hidden_states'][self.hidden_states_idx])
        # print("logits: ", logits.shape, logits.dtype)

        pooled_logits = logits[:, self.cls_token_idx]
        # print("pooled_logits: ", pooled_logits.shape, pooled_logits.dtype)

        # TODO: Should be put elsewhere
        labels = torch.nn.functional.one_hot(label_id, num_classes=self.num_labels).to(pooled_logits.device, pooled_logits.dtype)
        # print("labels: ", labels.shape, labels.dtype)

        loss_fct = self.loss_fct(self.label_weights)
        loss = loss_fct(pooled_logits, labels)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits
        )
    

def load_mh(
        clm_model_name=None,
        quantization_config=None,
        lora_config=None,
        label_weigths=None,
        torch_dtype=None, 
        device_map=None,
        hf_token=None
    ):


    tokenizer = AutoTokenizer.from_pretrained(clm_model_name, padding_side="left", token=hf_token)
    tokenizer.add_special_tokens({'sep_token':'<SEP>', 'pad_token':'<PAD>', 'cls_token':'<CLS>', 'mask_token':'<MASK>'})
    tokenizer.use_default_system_prompt = False


    clm_model = AutoModelForCausalLM.from_pretrained(
        clm_model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        token=hf_token
    )


    clm_model.config.pad_token_id = tokenizer.pad_token_id
    clm_model.resize_token_embeddings(len(tokenizer))


    mh_model = MultiHeadCLM(clm_model, label_weigths=label_weigths)
    mh_model = get_peft_model(mh_model, lora_config)
    mh_model = mh_model.to(torch_dtype)
    mh_model.print_trainable_parameters()


    return tokenizer, mh_model

