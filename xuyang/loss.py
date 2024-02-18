import torch
from torch import nn

class LabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, model_output, labels, shift_labels=False):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum(dim=-2) / num_active_elements
        smoothed_loss = smoothed_loss.sum(dim=-2) / (num_active_elements * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss

def sprk_compute_loss(outputs, labels, virtual_token_num, margin=0):
    relu = nn.ReLU()
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    shift_labels = shift_labels.to(shift_logits.device)
    # loss.shape = (1,4)
    loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
    loss_mask = loss != 0
    loss = (loss*loss_mask).sum(dim=1)/loss_mask.sum(dim=1)
    pos_neg_penalty = torch.mean(relu(loss[::2]-loss[1::2]))
    poins_loss = torch.mean(loss[::2])
    return pos_neg_penalty+poins_loss

def pairwise_compute_loss(outputs, labels, virtual_token_num, margin=0):
    relu = nn.ReLU()
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
    prefix_labels = torch.full((labels.shape[0], virtual_token_num), -100).to(labels.device)
    labels = torch.cat((prefix_labels, labels), dim=1)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    shift_labels = shift_labels.to(shift_logits.device)
    # loss.shape = (1,4)
    loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
    loss_mask = loss != 0
    loss = (loss*loss_mask).sum(dim=1)/loss_mask.sum(dim=1)
    pos_neg_penalty = torch.mean(relu(loss[::2]-loss[1::2]))
    return pos_neg_penalty

def constrastive_loss(outputs, labels, virtual_token_num, margin=0, device=None):
    relu = nn.ReLU()
    prefix_labels = torch.full((labels.shape[0], virtual_token_num), -100).to(labels.device)
    labels = torch.cat((prefix_labels, labels), dim=1)
    label_smoothed_loss = LabelSmoother()(outputs, labels)
    pos_neg_penalty = relu(torch.mean(label_smoothed_loss[::2]-label_smoothed_loss[1::2]) + margin)
    return pos_neg_penalty