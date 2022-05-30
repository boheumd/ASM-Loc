import os 
import json 
import torch 
import numpy as np 
from torch.utils.data import Dataset 
from net_evaluation import temporal_interpolation
from scipy import interpolate
from collections import defaultdict
import pdb

def dynamic_segment_sample(input_feature, sample_len, dynamic_segment_weights):
    input_len = input_feature.shape[0]
    if input_len == 1:
        sample_len = 2
        sample_idxs = np.rint(np.linspace(0, input_len-1, sample_len))
        dynamic_segment_weights_cumsum = np.concatenate((np.zeros((1,), dtype=float), np.array([0.5, 1.0], dtype=float)), axis=0)
        return input_feature[sample_idxs.astype(np.int), :], dynamic_segment_weights_cumsum
    else:
        assert sample_len > 0, "WRONG SAMPLE_LEN {}, THIS PARAM MUST BE GREATER THAN 0.".format(sample_len)
        dynamic_segment_weights_cumsum = np.concatenate((np.zeros((1,), dtype=float), np.cumsum(dynamic_segment_weights)), axis=0)
        max_dynamic_segment_weights_cumsum = np.round(dynamic_segment_weights_cumsum[-1]).astype(int)
        f_upsample = interpolate.interp1d(dynamic_segment_weights_cumsum, np.arange(input_len+1), kind='linear', axis=0, fill_value='extrapolate')
        scale_x = np.linspace(1, max_dynamic_segment_weights_cumsum, max_dynamic_segment_weights_cumsum)
        sampled_time = f_upsample(scale_x)
        f_feature = interpolate.interp1d(np.arange(1, input_len+1), input_feature, kind='linear', axis=0, fill_value='extrapolate')
        sampled_feature = f_feature(sampled_time)
        return sampled_feature, dynamic_segment_weights_cumsum

def uniform_sample(input_feature, sample_len):
    input_len = input_feature.shape[0]
    assert sample_len > 0, "WRONG SAMPLE_LEN {}, THIS PARAM MUST BE GREATER THAN 0.".format(sample_len)
    if input_len == 1:
        sample_len = 2
        sample_idxs = np.rint(np.linspace(0, input_len-1, sample_len))
    elif input_len <= sample_len:
        sample_idxs = np.arange(input_len)
    else:
        sample_scale = input_len / sample_len
        sample_idxs = np.arange(sample_len) * sample_scale
        sample_idxs = np.floor(sample_idxs)
    return input_feature[sample_idxs.astype(np.int), :]

def random_sample(input_feature, sample_len):
    input_len = input_feature.shape[0]
    assert sample_len > 0, "WRONG SAMPLE_LEN {}, THIS PARAM MUST BE GREATER THAN 0.".format(sample_len)
    
    if input_len < sample_len:
        return temporal_interpolation(input_feature, sample_len)
    elif input_len > sample_len:
        index_list = np.rint(np.linspace(0, input_len-1, sample_len+1)).astype(np.int)
        sample_idxs = np.zeros(sample_len)
        for i in range(sample_len):
            sample_idxs[i] = np.random.choice(range(index_list[i], index_list[i+1]))
    else:
        sample_idxs = np.arange(input_len)
    return input_feature[sample_idxs.astype(np.int), :]

class ASMLocDataset(Dataset):
    def __init__(self, args, phase="train", sample="random", step=None, logger=None):
        self.phase = phase 
        self.sample = sample
        self.data_dir = args.data_dir 
        self.sample_segments_num = args.sample_segments_num
        self.save_dir = args.save_dir
        self.pred_segment_path = args.pred_segment_path
        
        self.delta = args.delta
        self.max_segments_num = args.max_segments_num
        self.action_cls_num = args.action_cls_num
        self.dataset = args.dataset
        self.logger = logger

        with open(os.path.join(self.data_dir, "gt.json")) as gt_f:
            self.gt_dict = json.load(gt_f)["database"]
        args.gt_dict = self.gt_dict
        self.pseudo_segment_dict = {}
        self.pseudo_segment_dict['results'] = defaultdict(list)
        if not self.pred_segment_path is None:
            with open(self.pred_segment_path, 'r') as pred_f:
                self.pseudo_segment_dict = json.load(pred_f)

        if 'train' in self.phase:
            self.feature_dir = os.path.join(self.data_dir, "train")
            self.data_list = list(open(os.path.join(self.data_dir, "split_train.txt")))
            self.data_list = [item.strip() for item in self.data_list]
        elif 'test' in self.phase:
            self.feature_dir = os.path.join(self.data_dir, "test")
            self.data_list = list(open(os.path.join(self.data_dir, "split_test.txt")))
            self.data_list = [item.strip() for item in self.data_list]
        elif 'full' in self.phase:
            self.feature_dir = self.data_dir
            self.data_list = list(open(os.path.join(self.data_dir, "split_test.txt"))) + list(open(os.path.join(self.data_dir, "split_train.txt")))
            self.data_list = [item.strip() for item in self.data_list]

        self.dynamic_segment_weight_path = os.path.join(args.save_dir, 'dynamic_segment_weights_pred_step{}'.format(step))
        
        self.class_name_lst = args.class_name_lst
        self.action_class_idx_dict = {action_cls:idx for idx, action_cls in enumerate(self.class_name_lst)}
        
        self.action_class_num = args.action_cls_num
        
        self.get_proposals(self.pseudo_segment_dict)
        
    def get_proposals(self, pseudo_segment_dict):
        self.label_dict = {}
        self.gt_action_dict = defaultdict(list)
        self.pseudo_segment_dict = pseudo_segment_dict
        for vid_name in self.data_list:
            item_label = np.zeros(self.action_class_num)
            for ann in self.gt_dict[vid_name]["annotations"]:
                ann_label = ann["label"]
                item_label[self.action_class_idx_dict[ann_label]] = 1.0
                self.gt_action_dict[vid_name].append([ann['segment'][0], ann['segment'][1], 1.0, ann_label])
            self.label_dict[vid_name] = item_label

        self.pseudo_segment_dict_att = defaultdict(list)
        self.pseudo_segment_dict_pseudo = defaultdict(list)
        self.pseudo_segment_dict_all = defaultdict(list)
        for vid_name in self.data_list:
            label_set = set()
            if self.dataset == 'THUMOS':
                if 'validation' in vid_name:
                    for ann in self.gt_dict[vid_name]['annotations']:
                        label_set.add(ann['label'])
                elif 'test' in vid_name:
                    for pred in self.pseudo_segment_dict['results'][vid_name]:
                        label_set.add(pred['label'])
            elif self.dataset == 'ActivityNet':
                if self.gt_dict[vid_name]['subset'] == 'train':
                    for ann in self.gt_dict[vid_name]['annotations']:
                        label_set.add(ann['label'])
                elif self.gt_dict[vid_name]['subset'] == 'val':
                    for pred in self.pseudo_segment_dict['results'][vid_name]:
                        label_set.add(pred['label'])

            prediction_list_all = []
            for label in label_set:
                prediction_list = []
                for pred in self.pseudo_segment_dict['results'][vid_name]:
                    if pred['label'] == label:
                        t_start = pred["segment"][0]
                        t_end = pred["segment"][1]
                        prediction_list.append([t_start, t_end, pred["score"], pred["label"]])
                prediction_list = sorted(prediction_list, key=lambda k: k[2], reverse=True)
                prediction_list_all += prediction_list
            self.pseudo_segment_dict_all[vid_name] = prediction_list_all
            
            # remove duplicate proposals
            prediction_list_nodup = []
            for pred in prediction_list_all:
                t_start = pred[0]
                t_end = pred[1]
                if [t_start, t_end] not in prediction_list_nodup:
                    prediction_list_nodup.append([t_start, t_end])
            prediction_list_nodup = sorted(prediction_list_nodup, key=lambda k: k[0], reverse=True)
            
            # remove the proposals inside another proposal 
            prediction_list_att = []
            if len(prediction_list_nodup) > 0:
                prediction_list_att.append(prediction_list_nodup.pop(-1))
                while len(prediction_list_nodup) > 0:
                    prev_segment = prediction_list_att.pop(-1)
                    cur_segment = prediction_list_nodup.pop(-1)
                    if prev_segment[1] >= cur_segment[1]:
                        prediction_list_att.append(prev_segment)
                    else:
                        prediction_list_att.append(prev_segment)
                        prediction_list_att.append(cur_segment)
            self.pseudo_segment_dict_att[vid_name] = prediction_list_att

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        vid_name = self.data_list[idx]
        vid_label = self.label_dict[vid_name]
        if 'full' in self.phase:
            if self.dataset == 'THUMOS':
                if 'validation' in vid_name:
                    con_vid_feature = np.load(os.path.join(self.feature_dir, 'train', vid_name+".npy"))
                else:
                    con_vid_feature = np.load(os.path.join(self.feature_dir, 'test', vid_name+".npy"))
            elif self.dataset == 'ActivityNet':
                if os.path.isfile(os.path.join(self.feature_dir, 'train', vid_name+".npy")):
                    con_vid_feature = np.load(os.path.join(self.feature_dir, 'train', vid_name+".npy"))
                elif os.path.isfile(os.path.join(self.feature_dir, 'test', vid_name+".npy")):
                    con_vid_feature = np.load(os.path.join(self.feature_dir, 'test', vid_name+".npy"))
        else:
            con_vid_feature = np.load(os.path.join(self.feature_dir, vid_name+".npy"))
        
        vid_len = con_vid_feature.shape[0]
        
        dynamic_segment_weights_cumsum = None
        if self.sample == "random":
            input_feature = random_sample(con_vid_feature, self.sample_segments_num)
        elif self.sample == 'uniform':
            input_feature = uniform_sample(con_vid_feature, self.sample_segments_num)
        elif self.sample == 'dynamic_random':
            dynamic_segment_weights = np.load(os.path.join(self.dynamic_segment_weight_path, vid_name + ".npy"))
            input_feature, dynamic_segment_weights_cumsum = dynamic_segment_sample(con_vid_feature, self.sample_segments_num, dynamic_segment_weights)
            input_feature = random_sample(input_feature, self.sample_segments_num)
        elif self.sample == 'dynamic_uniform':
            dynamic_segment_weights = np.load(os.path.join(self.dynamic_segment_weight_path, vid_name + ".npy"))
            input_feature, dynamic_segment_weights_cumsum = dynamic_segment_sample(con_vid_feature, self.sample_segments_num, dynamic_segment_weights)
            input_feature = uniform_sample(input_feature, self.sample_segments_num)

        input_feature = torch.as_tensor(input_feature.astype(np.float32)) 
        vid_label = torch.as_tensor(vid_label.astype(np.float32))
        
        output_len = input_feature.shape[0]
        proposal_bbox = torch.zeros((self.max_segments_num, 2), dtype=torch.int32)
        pseudo_instance_label = torch.zeros((output_len, self.action_cls_num+1), dtype=torch.float32)
        # init all the timestep with bg class = 1
        pseudo_instance_label[:, -1] = 1

        time_to_index_factor = 25 / 16
        upsample_scale = time_to_index_factor * output_len / vid_len
        if dynamic_segment_weights_cumsum is not None and (vid_len + 1) == dynamic_segment_weights_cumsum.shape[0]:
            f_upsample = interpolate.interp1d(np.arange(vid_len+1), dynamic_segment_weights_cumsum, kind='linear', axis=0, fill_value='extrapolate')
            upsample_scale = time_to_index_factor * output_len / round(dynamic_segment_weights_cumsum[-1])
        else:
            dynamic_segment_weights_cumsum = None

        ########## generate proposal_bbox from pseudo_segment_att for Intra & Inter-Segment Attention modules ##########
        proposal_list_att = []
        for k, segment in enumerate(self.pseudo_segment_dict_att[vid_name]):
            t_start = segment[0]
            t_end = segment[1]
            t_mid = (t_start + t_end) / 2
            segment_duration = t_end - t_start
            if dynamic_segment_weights_cumsum is not None:
                t_start = (f_upsample(t_start * time_to_index_factor + 1) - 1) / time_to_index_factor
                t_end = (f_upsample(t_end * time_to_index_factor + 1) - 1) / time_to_index_factor
                t_mid = (t_start + t_end) / 2
            segment_duration = t_end - t_start
            index_start = max(round((t_mid - (self.delta + 0.5) * segment_duration) * upsample_scale), 0)
            index_end = min(round((t_mid + (self.delta + 0.5) * segment_duration) * upsample_scale), output_len-1)
            proposal_list_att.append([index_start, index_end])
        proposal_list_att = sorted(proposal_list_att, key=lambda k: k[0], reverse=True)

        proposal_count_by_video = len(proposal_list_att)
        for k, segment in enumerate(proposal_list_att):
            proposal_bbox[k, 0] = segment[0]
            proposal_bbox[k, 1] = segment[1]

        ########## generate pseudo_instance_label from pseudo_segment_all for Pseudo Instance-level Loss ##########
        fg_label_set_gt = np.where(self.label_dict[vid_name] == 1)[0]
        for segment in self.pseudo_segment_dict_all[vid_name]:
            t_start = segment[0]
            t_end = segment[1]
            t_label = self.action_class_idx_dict[segment[3]]
            if not t_label in fg_label_set_gt:
                continue
            if dynamic_segment_weights_cumsum is not None:
                t_start = (f_upsample(t_start * time_to_index_factor + 1) - 1) / time_to_index_factor
                t_end = (f_upsample(t_end * time_to_index_factor + 1) - 1) / time_to_index_factor
            index_start = max(int(round(t_start * upsample_scale)), 0)
            index_end = min(int(round(t_end * upsample_scale)), output_len-1)
            pseudo_instance_label[index_start:index_end+1, t_label] = 1
            pseudo_instance_label[index_start:index_end+1, -1] = 0
        pseudo_instance_label = pseudo_instance_label / torch.sum(pseudo_instance_label, dim=-1, keepdim=True).clamp(min=1e-6)
        return vid_name, input_feature, vid_label, vid_len, proposal_bbox, proposal_count_by_video, pseudo_instance_label, dynamic_segment_weights_cumsum

def my_collate_fn(batch):
    batched_output_list = []
    for i in range(len(batch[0])):
        if torch.is_tensor(batch[0][i]):
            batched_output = torch.stack([item[i] for item in batch], dim=0)
        else:
            batched_output = [item[i] for item in batch]
        batched_output_list.append(batched_output)
    return batched_output_list

def build_dataset(args, phase="train", sample="random", step=None, logger=None):
    return ASMLocDataset(args, phase, sample, step=step, logger=logger)


def grouping(arr):
    """
    Group the connected results
    """
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)

