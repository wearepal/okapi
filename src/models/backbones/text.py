from dataclasses import dataclass
from enum import Enum
from typing import cast

from ranzen import implements
from torch import Tensor
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.distilbert.modeling_distilbert import (
    DistilBertForSequenceClassification,
    DistilBertModel,
)

from src.models.base import BackboneFactory, ModelFactoryOut
from src.types import BertInput, DistilBertInput

__all__ = [
    "Bert",
    "DistilBert",
]


class _Bert(BertModel):
    def forward(self, x: BertInput) -> Tensor:
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]
        token_type_ids = x["token_type_ids"]
        pooled_outputs = cast(
            BaseModelOutputWithPoolingAndCrossAttentions,
            super().forward(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            ),
        )
        return pooled_outputs.pooler_output


class _DistilBert(DistilBertModel):
    def forward(self, x: DistilBertInput) -> Tensor:
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]
        hidden_state = cast(
            BaseModelOutput,
            super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ),
        )
        return hidden_state.last_hidden_state[:, 0]


class BertVersion(Enum):
    BASE_UNC = "bert-base-uncased"


@dataclass
class Bert(BackboneFactory):
    version: BertVersion = BertVersion.BASE_UNC

    @implements(BackboneFactory)
    def __call__(self) -> ModelFactoryOut[_Bert]:
        model = cast(_Bert, _Bert.from_pretrained(pretrained_model_name_or_path=self.version.value))
        return model, model.config.hidden_size


class DistilBertVersion(Enum):
    BASE_UNC = "distilbert-base-uncased"


@dataclass
class DistilBert(BackboneFactory):
    version: DistilBertVersion = DistilBertVersion.BASE_UNC

    @implements(BackboneFactory)
    def __call__(self) -> ModelFactoryOut[_DistilBert]:
        model = cast(
            _DistilBert,
            _DistilBert.from_pretrained(pretrained_model_name_or_path=self.version.value),
        )
        return model, model.config.hidden_size
