import torch
import torch.nn as nn
 
class LMCritierion(nn.Module):

    def __init__(self, label_smoothing, p1, p2):
        super(LMCritierion, self).__init__()
        self.label_smoothing = label_smoothing
        self.LogSoftmax = nn.LogSoftmax(dim=-1)
        self.p1 = p1
        self.p2 = p2
 
        if label_smoothing > 0:
            self.criterion = nn.KLDivLoss(size_average=False)
        else:
            self.criterion = nn.NLLLoss(size_average=False, ignore_index=-100)
        self.confidence = 1.0 - label_smoothing
 
    def _smooth_label(self, num_tokens, labels, confidence):
        
        gtruth = labels.view(-1)
        tdata = gtruth.detach()
        one_hot = self._one_hot(num_tokens)
        one_hot = one_hot.to(labels.device)
        tmp_ = one_hot.repeat(gtruth.size(0), 1)
        tmp_.scatter_(1, tdata.unsqueeze(1), confidence)
        gtruth = tmp_.detach()
        return gtruth

    def _one_hot(self, num_tokens):
        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.p2)
        return one_hot
 
    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def forward(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        num_tokens = scores.size(-1)

        gtruth = labels.view(-1)
        
        tdata = gtruth.detach()
        one_hot = self._one_hot(num_tokens) 
        if labels.is_cuda:
            one_hot = one_hot.to(labels.device)
        tmp_ = one_hot.repeat(gtruth.size(0), 1)  
        tmp_.scatter_(1, tdata.unsqueeze(1), self.p1)
        gtruth = tmp_.detach()
        loss = self.criterion(scores, gtruth)

        return loss

if __name__ == "__main__":

    crit = LMCritierion(0.5)
    logits = torch.randn(10, 5)
    labels = torch.randint(0,5,(10,))
    print(crit(logits, labels))
