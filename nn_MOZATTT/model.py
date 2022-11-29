from poolings import *
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from helper_functions import *


class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
            LOGGER.info(self.config)
        else:
            # self.config = torch.load(config_path)
            self.config = AutoConfig.from_pretrained(config_path, output_hidden_states=True)
            LOGGER.info(self.config)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if cfg.pooling == 'mean':
            self.pool = MeanPooling()
        elif cfg.pooling == 'max':
            self.pool = MaxPooling()
        elif cfg.pooling == 'min':
            self.pool = MinPooling()
        elif cfg.pooling == 'attention':
            self.pool = AttentionPooling(self.config.hidden_size)
        elif cfg.pooling == 'weightedlayer':
            self.pool = WeightedLayerPooling(self.config.num_hidden_layers, layer_start=cfg.layer_start,
                                             layer_weights=None)

        self.fc = nn.Linear(self.config.hidden_size, 6)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        feature = self.pool(last_hidden_states, inputs['attention_mask'])
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(feature)
        return output