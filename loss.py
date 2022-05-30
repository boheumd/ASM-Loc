import torch 
import torch.nn as nn 
from collections import defaultdict

# The base loss function without uncertainty
class ASMLoc_Base_Loss(nn.Module):
    def __init__(self, args, dataset="THUMOS14", beta=None):
        super(ASMLoc_Base_Loss, self).__init__()
        self.dataset = dataset
        self.lamb_fg = args.lamb_fg
        self.lamb_bg = args.lamb_bg
        self.lamb_abg = args.lamb_abg

    def cls_criterion(self, inputs, label):
        return -torch.mean(torch.sum(torch.log(inputs.clamp(min=1e-7)) * label, dim=-1))

    def forward(self, vid_label, fg_cls, bg_cls, temp_att, cas, fg_cas, bg_cas, pseudo_instance_label=None, uncertainty=None, dynamic_weights=None):
        device = fg_cls.device 
        batch_size = fg_cls.shape[0]
        fg_label = torch.hstack((vid_label, torch.zeros((batch_size, 1), device=device)))
        bg_label = torch.hstack((torch.zeros_like(vid_label), torch.ones((batch_size, 1), device=device)))
        
        # normalize the label vector by L1 norm
        fg_label = fg_label / torch.sum(fg_label, dim=1, keepdim=True)
        bg_label = bg_label / torch.sum(bg_label, dim=1, keepdim=True)
        
        fg_loss = self.cls_criterion(fg_cls, fg_label)
        bg_loss = self.cls_criterion(bg_cls, bg_label)
        abg_loss = self.cls_criterion(bg_cls, fg_label)
        
        cls_loss = self.lamb_fg * fg_loss + self.lamb_bg * bg_loss + self.lamb_abg * abg_loss
            
        loss = cls_loss
        loss_dict = defaultdict(float)
        loss_dict["fg_loss"] = fg_loss.cpu().item()
        loss_dict["bg_loss"] = bg_loss.cpu().item()
        loss_dict["abg_loss"] = abg_loss.cpu().item()        
        return loss, loss_dict

# The loss function with uncertainty guidance
class ASMLoc_Loss(nn.Module):
    def __init__(self, args, dataset="THUMOS14"):
        super(ASMLoc_Loss, self).__init__()
        self.dataset = dataset
        self.lamb_fg = args.lamb_fg
        self.lamb_bg = args.lamb_bg
        self.lamb_abg = args.lamb_abg
        self.beta = args.beta

    def cls_criterion(self, inputs, label, uncertainty=None):
        if not uncertainty is None:
            loss1 = -torch.mean(torch.sum(torch.exp(-uncertainty) * torch.log(inputs.clamp(min=1e-7)) * label, dim=-1)) #(B, T, C) -> (B, T) -> 1
            loss2 = self.beta * torch.mean(uncertainty) #(B, T, 1) -> 1
            return loss1 + loss2
        else:
            return -torch.mean(torch.sum(torch.log(inputs.clamp(min=1e-7)) * label, dim=-1))

    def forward(self, vid_label, fg_cls, bg_cls, temp_att, cas, fg_cas, bg_cas, pseudo_instance_label=None, uncertainty=None):
        device = fg_cls.device 
        batch_size = fg_cls.shape[0]
        fg_label = torch.hstack((vid_label, torch.zeros((batch_size, 1), device=device)))
        bg_label = torch.hstack((torch.zeros_like(vid_label), torch.ones((batch_size, 1), device=device)))
        
        # normalize the label vector by L1 norm
        fg_label = fg_label / torch.sum(fg_label, dim=1, keepdim=True)
        bg_label = bg_label / torch.sum(bg_label, dim=1, keepdim=True)
        
        fg_loss = self.cls_criterion(fg_cls, fg_label)
        bg_loss = self.cls_criterion(bg_cls, bg_label)
        abg_loss = self.cls_criterion(bg_cls, fg_label)
        
        cls_loss = self.lamb_fg * fg_loss + self.lamb_bg * bg_loss + self.lamb_abg * abg_loss
        pseudo_instance_loss = self.cls_criterion(cas, pseudo_instance_label, uncertainty)
        
        loss = cls_loss + pseudo_instance_loss
        loss_dict = defaultdict(float)
        loss_dict["fg_loss"] = fg_loss.cpu().item()
        loss_dict["bg_loss"] = bg_loss.cpu().item()
        loss_dict["abg_loss"] = abg_loss.cpu().item()
        loss_dict["pseudo_instance_loss"] = pseudo_instance_loss.cpu().item()
        return loss, loss_dict
