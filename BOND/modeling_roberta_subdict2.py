from transformers import RobertaModel, BertPreTrainedModel, RobertaConfig
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, KLDivLoss
from label_smothing import LMCritierion

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
    "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
    "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}

def convert(inp, mask):
    if inp is not None and inp.sum()!=0:
        if len(inp.view(-1)) == mask.shape[0]:
            return inp.view(-1)[mask]
        else:
            return inp.view(-1, inp.shape[-1])[mask]
    return None

class RobertaForTokenClassification_v2(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForTokenClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]
    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        label_mask=None,
        p1=None, p2=None, n1=None, n2=None, p2_labels=None, n1_labels=None,
        p2_1=1.0, p2_2=1.0, n1_1=1.0, n1_2=1.0
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        final_embedding = outputs[0]
        sequence_output = self.dropout(final_embedding)
        logits = self.classifier(sequence_output)

        outputs = (logits, final_embedding, ) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # Only keep active parts of the loss
            if attention_mask is not None or label_mask is not None:
                active_loss = True
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                if label_mask is not None:
                    active_loss = active_loss & label_mask.view(-1)
                active_logits = logits.view(-1, self.num_labels)[active_loss]
            
            p1 = convert(p1, active_loss)
            p2 = convert(p2, active_loss)
            n1 = convert(n1, active_loss)
            n2 = convert(n2, active_loss)
            if p1 is None and p2 is None and n1 is None and n2 is None:
              
                if labels.shape == logits.shape:
                    loss_fct = KLDivLoss()
                    if attention_mask is not None or label_mask is not None:
                        active_labels = labels.view(-1, self.num_labels)[active_loss]
                        loss = loss_fct(active_logits, active_labels)
                    else:
                        loss = loss_fct(logits, labels)
                else:
                    loss_fct = CrossEntropyLoss()
                    if attention_mask is not None or label_mask is not None:
                        active_labels = labels.view(-1)[active_loss]
                        loss = loss_fct(active_logits, active_labels)
                    else:
                        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

                outputs = (loss, ) + outputs
                return outputs
            
            if labels.shape == logits.shape:
                loss_fct = KLDivLoss()
                
                if attention_mask is not None or label_mask is not None:
                    active_labels = labels.view(-1, self.num_labels)[active_loss]
                    if p2_labels is not None:
                        p2_labels = p2_labels.view(-1, self.num_labels)[active_loss]

                        p2_labels = p2_labels.to(active_labels.device)
                    else:
                        p2_labels = active_labels
                    if n1_labels is not None:
                        n1_labels = n1_labels.view(-1, self.num_labels)[active_loss]
                        n1_labels = n1_labels.to(active_labels.device)
                    else:
                        n1_labels = active_labels

                    l_p1 = loss_fct(active_logits[p1], active_labels[p1]) if p1 is not None else torch.FloatTensor([0]).to(active_labels.device)
                    l_p2 = loss_fct(active_logits[p2], p2_labels[p2]) if p2 is not None else torch.FloatTensor([0]).to(active_labels.device)
                    l_n1 = loss_fct(active_logits[n1], n1_labels[n1]) if n1 is not None else torch.FloatTensor([0]).to(active_labels.device)
                    l_n2 = loss_fct(active_logits[n2], active_labels[n2]) if n2 is not None else torch.FloatTensor([0]).to(active_labels.device)

                    loss = ( l_p1 + l_p2 + l_n1 + l_n2 ) / ((len(p1) if p1 is not None else 0)+(len(p2) if p2 is not None else 0)+(len(n1) if n1 is not None else 0)+(len(n2) if n2 is not None else 0))
                else:
                    loss = loss_fct(logits, labels)
            else:
                loss_fct = CrossEntropyLoss(reduction='sum')
                lm_fct_p = LMCritierion(0.5, p2_1, p2_2)
                lm_fct_n = LMCritierion(0.5, n1_1, n1_2)
                if attention_mask is not None or label_mask is not None:
                    active_labels = labels.view(-1)[active_loss]
                    if p2_labels is not None:
                        p2_labels = p2_labels.view(-1)[active_loss]
                        p2_labels = p2_labels.to(active_labels.device)
                    else:
                        p2_labels = active_labels
                    if n1_labels is not None:
                        n1_labels = n1_labels.view(-1)[active_loss]
                        n1_labels = n1_labels.to(active_labels.device)
                    else:
                        n1_labels = active_labels

                    l_p1 = loss_fct(active_logits[p1], active_labels[p1]) if p1 is not None else torch.FloatTensor([0]).to(active_labels.device)

                    l_p2 = lm_fct_p(active_logits[p2], p2_labels[p2]) if p2 is not None else torch.FloatTensor([0]).to(active_labels.device)
                    
                    l_n1 = lm_fct_n(active_logits[n1], n1_labels[n1]) if n1 is not None else torch.FloatTensor([0]).to(active_labels.device)
                    l_n2 = loss_fct(active_logits[n2], active_labels[n2]) if n2 is not None else torch.FloatTensor([0]).to(active_labels.device)

                    loss = (l_p1 + l_n2 + l_p2 + l_n1) / ((len(p1) if p1 is not None else 0)+(len(p2) if p2 is not None else 0)+(len(n1) if n1 is not None else 0)+(len(n2) if n2 is not None else 0))
                    
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))


            outputs = (loss,) + outputs

        return outputs  # (loss), scores, final_embedding, (hidden_states), (attentions)
