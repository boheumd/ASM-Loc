import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
import pdb

class MHSA_Intra(nn.Module):
    """
    compute intra-segment attention
    """
    def __init__(self, dim_in, heads, num_pos, pos_enc_type='relative', use_pos=True):
        super(MHSA_Intra, self).__init__()

        self.dim_in = dim_in
        self.dim_inner = self.dim_in
        self.heads = heads
        self.dim_head = self.dim_inner // self.heads
        self.num_pos = num_pos

        self.scale = self.dim_head ** -0.5

        self.conv_query = nn.Conv1d(
            self.dim_in, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_key = nn.Conv1d(
            self.dim_in, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_value = nn.Conv1d(
            self.dim_in, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_out = nn.Conv1d(
            self.dim_inner, self.dim_in, kernel_size=1, stride=1, padding=0
        )
        self.bn = nn.BatchNorm1d(
            num_features=self.dim_in, eps=1e-5, momentum=0.1
        )
        self.bn.weight.data.zero_()
        self.bn.bias.data.zero_()
    
    def forward(self, input, intra_attn_mask):
        B, C, T = input.shape
        query = self.conv_query(input).view(B, self.heads, self.dim_head, T).permute(0, 1, 3, 2).contiguous() #(B, h, T, dim_head)
        key = self.conv_key(input).view(B, self.heads, self.dim_head, T) #(B, h, dim_head, T)
        value = self.conv_value(input).view(B, self.heads, self.dim_head, T).permute(0, 1, 3, 2).contiguous() #(B, h, T, dim_head)

        query *= self.scale
        sim = torch.matmul(query, key) #(B, h, T, T)
        intra_attn_mask = intra_attn_mask.view(B, 1, T, T)
        sim.masked_fill_(intra_attn_mask == 0, -np.inf)
        attn = F.softmax(sim, dim=-1) #(B, h, T, T)
        attn = torch.nan_to_num(attn, nan=0.0)
        output = torch.matmul(attn, value) #(B, h, T, dim_head)

        output = output.permute(0, 1, 3, 2).contiguous().view(B, C, T) #(B, C, T)
        output = input + self.bn(self.conv_out(output))
        return output

class MHSA_Inter(nn.Module):
    """
    compute inter-segment attention
    """
    def __init__(self, dim_in, heads, num_pos, pos_enc_type='relative', use_pos=True):
        super(MHSA_Inter, self).__init__()

        self.dim_in = dim_in
        self.dim_inner = self.dim_in
        self.heads = heads
        self.dim_head = self.dim_inner // self.heads
        self.num_pos = num_pos

        self.scale = self.dim_head ** -0.5
        # self.pos_emb_t = RelPosEmb1D(self.num_pos, self.dim_head)

        self.conv_query = nn.Conv1d(
            self.dim_in, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_key = nn.Conv1d(
            self.dim_in, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_value = nn.Conv1d(
            self.dim_in, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_out = nn.Conv1d(
            self.dim_inner, self.dim_in, kernel_size=1, stride=1, padding=0
        )
        self.bn = nn.BatchNorm1d(
            num_features=self.dim_in, eps=1e-5, momentum=0.1
        )
        self.bn.weight.data.zero_()
        self.bn.bias.data.zero_()
    
    def forward(self, input, inter_attn_mask, proposal_project_mask):
        B, C, T = input.shape
        B, T, K = proposal_project_mask.shape
        assert K == inter_attn_mask.shape[2], "segment num not equal"

        proposal_project_mask_norm = proposal_project_mask / torch.sum(proposal_project_mask, dim=1, keepdim=True).clamp(min=1e-5) #(B, T, K)
        segment_feature = torch.matmul(input, proposal_project_mask_norm) #(B, C, K)
        query_global = self.conv_query(segment_feature).view(B, self.heads, self.dim_head, K).permute(0, 1, 3, 2).contiguous() #(B, h, K, dim_head)
        key_global = self.conv_key(segment_feature).view(B, self.heads, self.dim_head, K) #(B, h, dim_head, K)
        value_global = self.conv_value(segment_feature).view(B, self.heads, self.dim_head, K).permute(0, 1, 3, 2).contiguous() #(B, h, K, dim_head)

        query_global *= self.scale
        sim_global = torch.matmul(query_global, key_global) #(B, h, K, K)
        inter_attn_mask = inter_attn_mask.view(B, 1, K, K)
        sim_global.masked_fill_(inter_attn_mask == 0, -np.inf)
        attn_global = F.softmax(sim_global, dim=-1) #(B, h, K, K)
        attn_global = torch.nan_to_num(attn_global, nan=0.0)
        output_global = torch.matmul(attn_global, value_global) #(B, h, K, dim_head)

        output_global = output_global.permute(0, 1, 3, 2).contiguous().view(B, C, K) #(B, C, K)
        proposal_project_mask_reverse_norm = proposal_project_mask.permute(0, 2, 1) / torch.sum(proposal_project_mask.permute(0, 2, 1), dim=1, keepdim=True).clamp(min=1e-5) #(B, K, T)
        output_global = torch.matmul(output_global, proposal_project_mask_reverse_norm) #(B, C, T)
        output_global = input + self.bn(self.conv_out(output_global)) #(B, C, T)
        return output_global

class ASMLoc_Base(nn.Module):
    """
    ASMLoc_Base: base model without action-aware segment modules
    """
    def __init__(self, args):
        super(ASMLoc_Base, self).__init__()
        self.dataset = args.dataset
        self.feature_dim = args.feature_dim
        self.action_cls_num = args.action_cls_num # Only the action categories number.
        self.drop_thresh = args.dropout
        self.fg_topk_seg = args.fg_topk_seg 
        self.bg_topk_seg = args.bg_topk_seg 
        
        self.dropout = nn.Dropout(args.dropout)
        if self.dataset == "THUMOS":
            self.feature_embedding = nn.Sequential(
                nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                )
        else:
            # We add more strong regularization operation to the ActivityNet dataset since this dataset contains much more diverse videos.
            self.feature_embedding = nn.Sequential(
                nn.Dropout(args.dropout),
                nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                )
        
        self.att_branch = nn.Conv1d(in_channels=self.feature_dim, out_channels=2, kernel_size=1, padding=0)
        self.snippet_cls = nn.Linear(in_features=self.feature_dim, out_features=(self.action_cls_num + 1))
        
    def forward(self, input_feature, proposal_bbox=None, proposal_count_by_video=None, \
        anchor_index=None, positive_index=None, negative_index=None, vid_len=None, vid_name=None):
        device = input_feature.device
        batch_size, temp_len = input_feature.shape[0], input_feature.shape[1]
        
        fg_topk_num = max(temp_len // self.fg_topk_seg, 1)
        bg_topk_num = max(temp_len // self.bg_topk_seg, 1)
        
        input_feature = input_feature.permute(0, 2, 1)
        embeded_feature = self.feature_embedding(input_feature)  #(B, C, T)
        
        if self.dataset == "THUMOS":
            temp_att = self.att_branch((embeded_feature))
        else:
            temp_att = self.att_branch(self.dropout(embeded_feature))
        
        temp_att = temp_att.permute(0, 2, 1) #(B, T, 3)
        temp_att = torch.softmax(temp_att, dim=2)
        fg_att = temp_att[:, :, 0].unsqueeze(2) #(B, T, 1)
        bg_att = temp_att[:, :, 1].unsqueeze(2)

        embeded_feature = embeded_feature.permute(0, 2, 1) #(B, T, C)
        
        select_idx = torch.ones((batch_size, temp_len, 1), device=device) #(B, T, 1)
        select_idx = self.dropout(select_idx)
        embeded_feature = embeded_feature * select_idx #(B, T, C)

        ##### get classification scores according to CAS * att by selecting top-k snippets, MIL loss #####
        cas = self.snippet_cls(self.dropout(embeded_feature)) #(B, T, K+1)
        fg_cas = cas * fg_att #(B, T, K+1)
        bg_cas = cas * bg_att
        
        sorted_fg_cas, _ = torch.sort(fg_cas, dim=1, descending=True)
        sorted_bg_cas, _ = torch.sort(bg_cas, dim=1, descending=True)
        
        fg_cls = torch.mean(sorted_fg_cas[:, :fg_topk_num, :], dim=1)
        bg_cls = torch.mean(sorted_bg_cas[:, :bg_topk_num, :], dim=1)
        fg_cls = torch.softmax(fg_cls, dim=1) #(B, K+1)
        bg_cls = torch.softmax(bg_cls, dim=1)
        
        fg_cas = torch.softmax(fg_cas, dim=2) #(B, T, K+1)
        bg_cas = torch.softmax(bg_cas, dim=2)
        
        cas = torch.softmax(cas, dim=2) #(B, T, K+1)
        return fg_cls, bg_cls, temp_att, cas, fg_cas, bg_cas, None

class ASMLoc(nn.Module):
    """
    ASMLoc: ASMLoc_Base model with action-aware segment modules
    """
    def __init__(self, args):
        super(ASMLoc, self).__init__()
        self.dataset = args.dataset
        self.feature_dim = args.feature_dim
        self.action_cls_num = args.action_cls_num # Only the action categories number.
        self.drop_thresh = args.dropout
        self.fg_topk_seg = args.fg_topk_seg 
        self.bg_topk_seg = args.bg_topk_seg 
        self.max_segments_num = args.max_segments_num

        self.dropout = nn.Dropout(args.dropout)
        if self.dataset == "THUMOS":
            self.feature_embedding = nn.Sequential(
                nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                )
        else:
            # We add more strong regularization operation to the ActivityNet dataset since this dataset contains much more diverse videos.
            self.feature_embedding = nn.Sequential(
                nn.Dropout(args.dropout),
                nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                )
        self.MHSA_Intra = MHSA_Intra(dim_in=self.feature_dim, num_pos=args.sample_segments_num, heads=8)
        self.MHSA_Inter = MHSA_Inter(dim_in=self.feature_dim, num_pos=args.sample_segments_num, heads=8)
        
        self.att_branch = nn.Conv1d(in_channels=self.feature_dim, out_channels=2, kernel_size=1, padding=0)
        self.snippet_cls = nn.Linear(in_features=self.feature_dim, out_features=(self.action_cls_num + 1))
        self.uncertainty_branch = nn.Conv1d(in_channels=self.feature_dim, out_channels=1, kernel_size=1, padding=0, bias=False)
        self.uncertainty_branch.weight.data.normal_(0.0, 0.0001)

    def forward(self, input_feature, proposal_bbox, proposal_count_by_video, vid_len, vid_name):
        device = input_feature.device
        batch_size, temp_len = input_feature.shape[0], input_feature.shape[1]
        
        fg_topk_num = max(temp_len // self.fg_topk_seg, 1)
        bg_topk_num = max(temp_len // self.bg_topk_seg, 1)

        input_feature = input_feature.permute(0, 2, 1)
        embeded_feature = self.feature_embedding(input_feature)  #(B, C, T)
        intra_attn_mask = torch.zeros((batch_size, temp_len, temp_len), dtype=torch.float32).to(input_feature)
        inter_attn_mask = torch.zeros((batch_size, self.max_segments_num, self.max_segments_num), dtype=torch.float32).to(input_feature)
        proposal_project_mask = torch.zeros((batch_size, temp_len, self.max_segments_num), dtype=torch.float32).to(input_feature)
        for n in range(batch_size):
            for k in range(proposal_count_by_video[n]):
                proposal_start_index = proposal_bbox[n, k, 0]
                proposal_end_index = proposal_bbox[n, k, 1]
                intra_attn_mask[n, proposal_start_index:proposal_end_index+1, proposal_start_index:proposal_end_index+1] = 1.0
                proposal_project_mask[n, proposal_start_index: proposal_end_index+1, k] = 1
            inter_attn_mask[n, 0:proposal_count_by_video[n], 0:proposal_count_by_video[n]] = 1
        embeded_feature = self.MHSA_Intra(embeded_feature, intra_attn_mask)  #(B, C, T)
        embeded_feature = self.MHSA_Inter(embeded_feature, inter_attn_mask, proposal_project_mask)  #(B, C, T)
        
        if self.dataset == "THUMOS":
            temp_att = self.att_branch((embeded_feature))
            uncertainty = self.uncertainty_branch(embeded_feature) #(B, 1, T)
        else:
            temp_att = self.att_branch(self.dropout(embeded_feature))
            uncertainty = self.uncertainty_branch(self.dropout(embeded_feature)) #(B, 1, T)
        
        temp_att = temp_att.permute(0, 2, 1) #(B, T, 2)
        temp_att = torch.softmax(temp_att, dim=2)
        fg_att = temp_att[:, :, 0].unsqueeze(2) #(B, T, 1)
        bg_att = temp_att[:, :, 1].unsqueeze(2)

        embeded_feature = embeded_feature.permute(0, 2, 1) #(B, T, C)
        
        select_idx = torch.ones((batch_size, temp_len, 1), device=device)
        select_idx = self.dropout(select_idx)
        embeded_feature = embeded_feature * select_idx #(B, T, C)

        ##### get classification scores according to CAS * att by selecting top-k snippets, MIL loss #####
        cas = self.snippet_cls(self.dropout(embeded_feature)) #(B, T, K+1)
        fg_cas = cas * fg_att
        bg_cas = cas * bg_att
        
        sorted_fg_cas, _ = torch.sort(fg_cas, dim=1, descending=True)
        sorted_bg_cas, _ = torch.sort(bg_cas, dim=1, descending=True)
        
        fg_cls = torch.mean(sorted_fg_cas[:, :fg_topk_num, :], dim=1)
        bg_cls = torch.mean(sorted_bg_cas[:, :bg_topk_num, :], dim=1)
        fg_cls = torch.softmax(fg_cls, dim=1) #(B, K+1)
        bg_cls = torch.softmax(bg_cls, dim=1)
        
        fg_cas = torch.softmax(fg_cas, dim=2) #(B, T, K+1)
        bg_cas = torch.softmax(bg_cas, dim=2)
        
        cas = torch.softmax(cas, dim=2) #(B, T, K+1)
        return fg_cls, bg_cls, temp_att, cas, fg_cas, bg_cas, uncertainty.permute(0, 2, 1)
