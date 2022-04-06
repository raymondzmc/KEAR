# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from path_explain import EmbeddingExplainerTorch
from .layers import ChoicePredictor
import torch
import torch.nn as nn
import torch.nn.functional as F
from .perturbation import SmartPerturbation
from .sift import SiFTAdversarialLearner, hook_sift_layer
from .loss import LOSS_REGISTRY, LossCriterion
from torch.cuda.amp import autocast
import torch.distributed as dist
from transformers import (
    PreTrainedModel, AlbertModel, AlbertConfig, DebertaModel, DebertaConfig, 
    DebertaV2Config, DebertaV2Model,
    ElectraModel, ElectraConfig, RobertaModel, RobertaConfig,
    load_tf_weights_in_electra
)
from transformers.modeling_outputs import BaseModelOutput
import logging

import gc, pdb

from captum.attr import LayerIntegratedGradients

logger = logging.getLogger(__name__)
CSQA_CHOICE_NUM = 5
VERY_NEGATIVE_NUMBER = -1e7


class Model(PreTrainedModel):
    """
    AlBert-AttentionMerge-Classifier

    1. self.forward(input_ids, attention_mask, token_type_ids, label)
    2. self.predict(input_ids, attention_mask, token_type_ids)
    """
    def __init__(self, config, opt):
        super(Model, self).__init__(config)

        self.my_config = opt
        # self.safe_deberta = True

        model_type = Model.model_type
        print('model_type=', model_type)
        if model_type == 'albert':
            self.albert = AlbertModel(config)  
        elif model_type == 'deberta':
            self.deberta = DebertaModel(config)  
        elif model_type == 'electra':
            self.electra = ElectraModel(config)  
        elif model_type == 'roberta':
            self.roberta = RobertaModel(config)
        elif model_type == 'debertav2':
            self.deberta = DebertaV2Model(config)
        else:
            raise ValueError('Model type not supported.')
        scorer = {}
        scorer[opt['data_version']] = ChoicePredictor(config, opt)
        self.scorer = nn.ModuleDict(scorer)
        self.hidden_size = config.hidden_size
        self.num_choices = opt['num_choices']
        if self.my_config.get('adv_train', False):
            if self.my_config.get('adv_sift', False):
                adv_modules = hook_sift_layer(self, hidden_size=self.hidden_size, 
                                              learning_rate=self.my_config['adv_step_size'],
                                              init_perturbation=self.my_config['adv_noise_var'])
                self.adv_teacher = SiFTAdversarialLearner(self, adv_modules)
            else:
                cs = self.my_config['adv_loss']
                assert cs is not None
                lc = LOSS_REGISTRY[LossCriterion[cs]](name='Adv Loss func: {}'.format(cs))
                self.adv_task_loss_criterion = [lc]
                self.adv_teacher = SmartPerturbation(self.my_config['adv_epsilon'],
                        self.my_config['adv_step_size'],
                        self.my_config['adv_noise_var'],
                        self.my_config['adv_p_norm'],
                        self.my_config['adv_k'],
                        loss_map=self.adv_task_loss_criterion,
                        norm_level=self.my_config['adv_norm_level'])

        self.init_weights()
        self.requires_grad = {}
        print('init model finished.')

    def normalize_name(self, name):
        return name.replace('-', '_')

    def lm(self):
        if Model.base_model_prefix == 'albert':
            return self.albert
        elif Model.base_model_prefix == 'deberta':
            return self.deberta
        elif Model.base_model_prefix == 'electra':
            return self.electra
        elif Model.base_model_prefix == 'roberta':
            return self.roberta

    @classmethod
    def set_config(cls, model_type='albert'):
        print('set config, model_type=', model_type)
        cls.model_type = model_type
        if model_type == 'deberta':
            cls.config_class = DebertaConfig
            cls.base_model_prefix = "deberta"
            cls._keys_to_ignore_on_load_missing = ["position_ids"]  
        elif model_type == 'albert':
            cls.config_class = AlbertConfig
            cls.base_model_prefix = "albert"
            cls._keys_to_ignore_on_load_missing = [r"position_ids"]      
        elif model_type == 'electra':
            cls.config_class = ElectraConfig
            cls.load_tf_weights = load_tf_weights_in_electra
            cls.base_model_prefix = "electra"
            cls._keys_to_ignore_on_load_missing = [r"position_ids"]
            cls._keys_to_ignore_on_load_unexpected = [r"electra\.embeddings_project\.weight", r"electra\.embeddings_project\.bias"]
        elif model_type == 'roberta':
            cls.config_class = RobertaConfig
            cls.base_model_prefix = "roberta"
        elif model_type == 'debertav2':
            cls.config_class = DebertaV2Config
            cls.base_model_prefix = "deberta"
            cls._keys_to_ignore_on_load_missing = ["position_ids"]
            cls._keys_to_ignore_on_load_unexpected = ["position_embeddings"]

    def _init_weights(self, module):
        """Initialize the weights."""
        model_type = Model.model_type
        if model_type == 'debertav2':
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()    
        else:        
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if isinstance(module, (nn.Linear)) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)        

    def embed_encode(self, input_ids, token_type_ids=None, attention_mask=None):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        lm = self.lm()
        embedding_output = lm.embeddings(flat_input_ids, flat_token_type_ids)
        return embedding_output

    def _interpret_forward(self, embedding_output, attention_mask, dataset_name='csqa'):
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        lm = self.lm()
        encoder_outputs = lm.encoder(
            embedding_output,
            flat_attention_mask,
        )
        encoded_layers = encoder_outputs[1]
        sequence_output = encoded_layers[-1]
        return self.scorer[dataset_name]((sequence_output,), attention_mask)


    def forward(self, *batch):
        """
        batch: (0:idx, 1:input_ids, 2:attention_mask, 3:token_type_ids, 4:question_mask, 5:choice_mask, 6:choice labels, 7:dataset_name, 8:mode)
        """
        choice_mask, labels, dataset_name, method = batch[-4:]
        idx, input_ids, attention_mask, token_type_ids, question_mask = batch[:-4]
        input_size = self._to_tensor(idx.size(0), idx.device)

        embedding_output = self.embed_encode(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        saliency_map = []
        for choice_idx in range(choice_mask.shape[-1]):
            input_len = int(attention_mask[:, [choice_idx]].sum().item())
            explainer = EmbeddingExplainerTorch(
                model=lambda x: self._interpret_forward(x, attention_mask[:, [choice_idx]], dataset_name=dataset_name),
                embedding_axis=-1,
            )

            baseline = torch.zeros((1,) + embedding_output[[choice_idx]].shape[1:], device=embedding_output.device)

            if method == 'attribution':
                saliency_map.append(
                    explainer.attributions(
                        embedding_output[[choice_idx]],
                        baseline,
                        batch_size=1,
                        num_samples=10,
                        use_expectation=False,
                    ).squeeze(0)[:input_len].tolist()
                )
            elif method == 'interaction':
                saliency_map.append(
                    explainer.interactions(
                        embedding_output[[choice_idx]],
                        baseline,
                        batch_size=1,
                        num_samples=10,
                        use_expectation=False,
                    ).squeeze(0)[:input_len, :input_len].tolist()
                )
            else:
                raise NotImplementedError("\"interpretation_method\" not supported!")


        # scores = self._interpret_forward(lm_embeddings, attention_mask)
        # def forward_func(input_ids, index=0):
        #     pred = self._forward(idx,
        #                          input_ids[:, [index]],
        #                          attention_mask[:, [index]],
        #                          token_type_ids[:, [index]],
        #                          question_mask,
        #                          dataset_name)
        #     return pred[0]


        # attritions = []
        # lig = LayerIntegratedGradients(forward_func, self.deberta.embeddings.word_embeddings)
        # lm_embeddings.requires_grad_()
        # for index in range(choice_mask.size(1)):
        #     torch.cuda.empty_cache()
        #     attr = lig.attribute(inputs=(input_ids),
        #                          additional_forward_args=(0),
        #                          n_steps=50,
        #                          internal_batch_size=1)
        #     attr = attr.sum(dim=-1).squeeze(0)
        #     attr = attr / torch.norm(attr)
        #     attritions.append(attr.tolist())


        with torch.no_grad():
            logits = self._forward(idx, input_ids, attention_mask, token_type_ids, question_mask, dataset_name)

        label_to_use = labels
        clf_logits = choice_mask * VERY_NEGATIVE_NUMBER + logits

        loss = F.cross_entropy(clf_logits, label_to_use.view(-1), reduction='none')

        # Not used for interpretation
        adv_loss = torch.zeros_like(loss)
        adv_norm = torch.zeros_like(loss)

        with torch.no_grad():
            predicts = torch.argmax(clf_logits, dim=1)
            right_num = (predicts == labels)
        return loss, right_num, input_size, clf_logits, adv_norm, saliency_map


    # def forward(self, *batch):
    #     """
    #     batch: (0:idx, 1:input_ids, 2:attention_mask, 3:token_type_ids, 4:question_mask, 5:choice_mask, 6:choice labels, 7:dataset_name, 8:mode)
    #     """
    #     choice_mask, labels, dataset_name, mode = batch[-4:]
    #     idx, input_ids, attention_mask, token_type_ids, question_mask = batch[:-4]

    #     if self.my_config['break_input']:
    #         gc.collect()
    #         torch.cuda.empty_cache()
    #         logits = torch.cat([
    #             self._forward(
    #                 idx,
    #                 input_ids[:, [i]],
    #                 attention_mask[:, [i]],
    #                 token_type_ids[:, [i]],
    #                 question_mask[:, [i]],
    #                 dataset_name
    #             )
    #             for i in range(self.num_choices)
    #         ], dim=-1)
    #     else:
    #         logits = self._forward(idx, input_ids, attention_mask, token_type_ids, question_mask, dataset_name)
        
    #     label_to_use = labels
    #     clf_logits = choice_mask * VERY_NEGATIVE_NUMBER + logits

    #     loss = F.cross_entropy(clf_logits, label_to_use.view(-1), reduction='none')
    #     adv_loss = torch.zeros_like(loss)
    #     adv_norm = torch.zeros_like(loss)
    #     if self.my_config.get('adv_train', False) and mode == 'train':
    #         if self.my_config.get('adv_sift', False):
    #             adv_loss, adv_norm = self.adv_teacher.loss(logits, self._forward, self.my_config['grad_adv_loss'], 
    #                 self.my_config['adv_loss'], idx, input_ids, attention_mask, token_type_ids, question_mask, dataset_name)
    #             loss = loss + self.my_config['adv_alpha'] * adv_loss
    #         else:
    #             adv_loss, adv_norm, adv_logits = self.adv_teacher.forward(self, logits, idx, input_ids, token_type_ids, attention_mask, question_mask, dataset_name)
    #             if adv_loss is None:
    #                 adv_loss = torch.zeros_like(loss)
    #                 adv_norm = adv_loss
    #             loss = loss + self.my_config['adv_alpha'] * adv_loss
    #     input_size = self._to_tensor(idx.size(0), idx.device)
    #     with torch.no_grad():
    #         predicts = torch.argmax(clf_logits, dim=1)
    #         right_num = (predicts == labels)
    #     return loss, right_num, input_size, clf_logits, adv_norm

    def _forward(self, idx, input_ids, attention_mask, token_type_ids, question_mask, dataset_name='csqa', inputs_embeds=None, return_raw=False):
        if inputs_embeds is None:
            flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        else:
            flat_input_ids = None

        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        lm = self.lm()

        # if self.my_config['break_input']:

        #     if input_ids.size(0) != 1:
        #         raise Exception("Break input only supported for batch size = 1!") 

        #     scores = []

        #     # Perform a separate inference for each choice to save memory (more time)
        #     for i in range(self.num_choices):

        #         # Clear cache to save memory
        #         gc.collect()
        #         torch.cuda.empty_cache()
        #         oom = False


        #         try:
        #             last_hidden_state = lm(flat_input_ids[[i]], flat_attention_mask[[i]], flat_token_type_ids[[i]], inputs_embeds)['last_hidden_state']
        #             outputs = BaseModelOutput(last_hidden_state=last_hidden_state)
        #             scores.append(self.scorer[dataset_name](outputs, attention_mask[:, [i]]))

        #             del last_hidden_state, outputs

        #         except RuntimeError: # Out of memory
        #             oom = True

        #         if oom:
        #             stats = torch.cuda.memory_stats(device=0)
        #             pdb.set_trace()

        #     return torch.cat(scores, dim=-1)

        # else:
        outputs = lm(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        return self.scorer[dataset_name](outputs, attention_mask)


    def predict(self, idx, input_ids, attention_mask, token_type_ids):
        """
        return: [B, 2]
        """
        return self._forward(idx, input_ids, attention_mask, token_type_ids)

    def _to_tensor(self, it, device): return torch.tensor(it, device=device, dtype=torch.float)
