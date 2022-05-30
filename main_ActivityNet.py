import os 
import json 
import random 
from tqdm import tqdm 

import torch 
import numpy as np 
from torch.utils.data import DataLoader
from config import build_args 
from dataset import build_dataset, my_collate_fn
from model import ASMLoc_Base, ASMLoc
from loss import ASMLoc_Base_Loss, ASMLoc_Loss
from net_evaluation import ANETDetection, upgrade_resolution, get_proposal_oic, nms, result2json, grouping
from datetime import datetime

import pdb

def train(args, model, dataloader, criterion, optimizer, cur_epoch=0, logger=None, step=0, num_steps=0):
    model.train()
    print("-------------------------------------------------------------------------------")
    device = args.device
    
    train_num_correct = 0
    train_num_total = 0
    
    loss_stack = []
    fg_loss_stack = []
    bg_loss_stack = []
    abg_loss_stack = []
    pseudo_instance_loss_stack = []

    train_final_result = dict()
    train_final_result['version'] = 'VERSION 1.3'
    train_final_result['results'] = {}
    train_final_result['class_score'] = {}
    train_final_result['external_data'] = {'used': True, 'details': 'Features from I3D Net'}

    for cur_iter, (vid_name, input_feature, vid_label, vid_len, proposal_bbox, proposal_count_by_video, pseudo_instance_label, dynamic_segment_weights_cumsum) in enumerate(dataloader):

        vid_label = vid_label.to(device)
        input_feature = input_feature.to(device)
        proposal_bbox = proposal_bbox.to(device)
        pseudo_instance_label = pseudo_instance_label.to(device)

        fg_cls, bg_cls, temp_att, cas, fg_cas, bg_cas, uncertainty_pred = \
        model(input_feature, proposal_bbox=proposal_bbox, proposal_count_by_video=proposal_count_by_video, vid_len=vid_len, vid_name=vid_name)

        loss, loss_dict = criterion(vid_label, fg_cls, bg_cls, temp_att, cas, fg_cas, bg_cas, pseudo_instance_label=pseudo_instance_label, uncertainty=uncertainty_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            fg_score = fg_cls[:, :args.action_cls_num]
            label_np = vid_label.cpu().numpy()
            score_np = fg_score.cpu().numpy()
            
            pred_np = np.zeros_like(score_np)
            pred_np[score_np >= args.cls_threshold] = 1
            pred_np[score_np < args.cls_threshold] = 0
            correct_pred = np.sum(label_np == pred_np, axis=1)
            
            train_num_correct += np.sum((correct_pred == args.action_cls_num))
            train_num_total += correct_pred.shape[0]
            
            loss_stack.append(loss.cpu().item())
            fg_loss_stack.append(loss_dict["fg_loss"])
            bg_loss_stack.append(loss_dict["bg_loss"])
            abg_loss_stack.append(loss_dict["abg_loss"])
            pseudo_instance_loss_stack.append(loss_dict["pseudo_instance_loss"])


    train_acc = train_num_correct/train_num_total
    train_log_dict = {}
    train_log_dict["train_fg_cls_loss"] = np.mean(fg_loss_stack)
    train_log_dict["train_bg_cls_loss"] = np.mean(bg_loss_stack)
    train_log_dict["train_abg_cls_loss"] = np.mean(abg_loss_stack)
    train_log_dict["train_pseudo_instance_loss"] = np.mean(pseudo_instance_loss_stack)
    train_log_dict["train_loss"] = np.mean(loss_stack)
    train_log_dict["train_acc"] = train_acc

    print_str = "Epoch:{}/{} step:{}/{}\n".format(cur_epoch, args.epochs, step, num_steps) + \
                "train_fg_cls_loss:{:.3f}  train_bg_cls_loss:{:.3f}  train_abg_cls_loss:{:.3f}\n".format(np.mean(fg_loss_stack), np.mean(bg_loss_stack), np.mean(abg_loss_stack)) + \
                "train_pseudo_instance_loss:{:.3f}\n".format(np.mean(pseudo_instance_loss_stack)) + \
                "train_loss:{:.3f}\n".format(np.mean(loss_stack)) + \
                "train acc:{:.3f}\n".format(train_acc)

    print(print_str, flush=True)
    if logger:
        logger.write(print_str + '\n')
    
    return train_log_dict


def test(args, model, dataloader, criterion, logger=None, step=0):
    model.eval()
    device = args.device
    save_dir = args.save_dir
    
    test_num_correct = 0
    test_num_total = 0
    
    loss_stack = []
    fg_loss_stack = []
    bg_loss_stack = []
    abg_loss_stack = []
    pseudo_instance_loss_stack = []
    
    test_final_result = dict()
    test_final_result['version'] = 'VERSION 1.3'
    test_final_result['results'] = {}
    test_final_result['class_score'] = {}
    test_final_result['external_data'] = {'used': True, 'details': 'Features from I3D Net'}

    for vid_name, input_feature, vid_label, vid_len, proposal_bbox, proposal_count_by_video, pseudo_instance_label, dynamic_segment_weights_cumsum in tqdm(dataloader):
        input_feature = input_feature.to(device)
        vid_label = vid_label.to(device)
        proposal_bbox = proposal_bbox.to(device)
        pseudo_instance_label = pseudo_instance_label.to(device)
        #--------------------------------------------------------------------------#
        #--------------------------------------------------------------------------#
        fg_cls, bg_cls, temp_att, cas, fg_cas, bg_cas, uncertainty_pred = \
            model(input_feature, proposal_bbox=proposal_bbox, proposal_count_by_video=proposal_count_by_video, vid_len=vid_len, vid_name=vid_name)

        loss, loss_dict = criterion(vid_label, fg_cls, bg_cls, temp_att, cas, fg_cas, bg_cas, pseudo_instance_label=pseudo_instance_label, uncertainty=uncertainty_pred)

        vid_len = vid_len[0]
        t_factor = (args.segment_frames_num * vid_len) / (args.frames_per_sec * args.test_upgrade_scale * input_feature.shape[1])
        vid_duration = args.segment_frames_num * vid_len / args.frames_per_sec

        loss_stack.append(loss.cpu().item())
        fg_loss_stack.append(loss_dict["fg_loss"])
        bg_loss_stack.append(loss_dict["bg_loss"])
        abg_loss_stack.append(loss_dict["abg_loss"])
        pseudo_instance_loss_stack.append(loss_dict["pseudo_instance_loss"])
        
        #--------------------------------------------------------------------------#
        #--------------------------------------------------------------------------#
        temp_cas = fg_cas
        fg_score = fg_cls[:, :args.action_cls_num]
        label_np = vid_label.cpu().numpy()
        score_np = fg_score.cpu().numpy() # (K)
        pred_np = np.zeros_like(score_np)
        pred_np[score_np >= args.cls_threshold] = 1
        pred_np[score_np < args.cls_threshold] = 0
        correct_pred = np.sum(label_np == pred_np, axis=1)
        test_num_correct += np.sum((correct_pred == args.action_cls_num))
        test_num_total += correct_pred.shape[0]
        #--------------------------------------------------------------------------#
        #--------------------------------------------------------------------------#
        # GENERATE PROPORALS.
        temp_cls_score_np = temp_cas[:, :, :args.action_cls_num].cpu().numpy()
        temp_cls_score_np = np.reshape(temp_cls_score_np, (temp_cas.shape[1], args.action_cls_num, 1))
        temp_att_fg_score_np = temp_att[:, :, 0].unsqueeze(2).expand([-1, -1, args.action_cls_num]).cpu().numpy()
        temp_att_fg_score_np = np.reshape(temp_att_fg_score_np, (temp_cas.shape[1], args.action_cls_num, 1))

        score_np = np.reshape(score_np, (-1))
        if score_np.max() > args.cls_threshold:
            cls_prediction = np.array(np.where(score_np > args.cls_threshold)[0]) #(num_pred_classes)
        else:
            cls_prediction = np.array([np.argmax(score_np)], dtype=int)
        temp_cls_score_np = temp_cls_score_np[:, cls_prediction] #(T, num_pred_classes, 1)
        temp_att_fg_score_np = temp_att_fg_score_np[:, cls_prediction]

        int_temp_cls_scores = upgrade_resolution(temp_cls_score_np, args.test_upgrade_scale) #(T*upscale, num_pred_classes, 1)
        int_temp_att_fg_score_np = upgrade_resolution(temp_att_fg_score_np, args.test_upgrade_scale)
        
        cas_thresh_list = [0.005, 0.01, 0.015, 0.02]
        att_thresh_list = [0.005, 0.01, 0.015, 0.02]
        proposal_dict = {}
        # CAS based proposal generation
        for cas_thresh in cas_thresh_list:
            tmp_int_cas = int_temp_cls_scores.copy() #(T*upscale, num_pred_classes, 1)
            zero_location = np.where(tmp_int_cas < cas_thresh)
            tmp_int_cas[zero_location] = 0
            
            tmp_seg_list = []
            for c_idx in range(len(cls_prediction)):
                pos = np.where(tmp_int_cas[:, c_idx] >= cas_thresh)
                tmp_seg_list.append(pos)
            
            props_list = get_proposal_oic(tmp_seg_list, (0.70*tmp_int_cas + 0.30*int_temp_att_fg_score_np), cls_prediction, \
                    score_np, t_factor, lamb=0.150, gamma=0.0, dynamic_segment_weights_cumsum=dynamic_segment_weights_cumsum[0], vid_duration=vid_duration)
            
            for i in range(len(props_list)):
                if len(props_list[i]) == 0:
                    continue
                class_id = props_list[i][0][0]
                
                if class_id not in proposal_dict.keys():
                    proposal_dict[class_id] = []
                
                proposal_dict[class_id] += props_list[i]
        
        # att based proposal generation
        for att_thresh in att_thresh_list:
            tmp_int_att = int_temp_att_fg_score_np.copy()
            zero_location = np.where(tmp_int_att < att_thresh)
            tmp_int_att[zero_location] = 0
            
            tmp_seg_list = []
            for c_idx in range(len(cls_prediction)):
                pos = np.where(tmp_int_att[:, c_idx] >= att_thresh)
                tmp_seg_list.append(pos)

            props_list = get_proposal_oic(tmp_seg_list, (0.70*int_temp_cls_scores + 0.30*tmp_int_att), cls_prediction, \
                    score_np, t_factor, lamb=0.150, gamma=0.250, dynamic_segment_weights_cumsum=dynamic_segment_weights_cumsum[0], vid_duration=vid_duration)
            
            for i in range(len(props_list)):
                if len(props_list[i]) == 0:
                    continue
                class_id = props_list[i][0][0]
                
                if class_id not in proposal_dict.keys():
                    proposal_dict[class_id] = []
                
                proposal_dict[class_id] += props_list[i]
        
        # NMS 
        final_proposals = []
        for class_id in proposal_dict.keys():
            final_proposals.append(nms(proposal_dict[class_id], args.nms_thresh))
        test_final_result['results'][vid_name[0]] = result2json(final_proposals, args.class_name_lst)
    test_acc = test_num_correct / test_num_total

    if args.test:
        # Final Test
        test_final_json_path = os.path.join(save_dir, "final_test_{}_result.json".format(args.dataset))
    else:
        # Train Evalutaion
        test_final_json_path = os.path.join(save_dir, "{}_lateset_result_step{}.json".format(args.dataset, step))

    with open(test_final_json_path, 'w') as f:
        json.dump(test_final_result, f)

    anet_detection = ANETDetection(ground_truth_file=args.test_gt_file_path,
                    prediction_file=test_final_json_path,
                    tiou_thresholds=args.tiou_thresholds,
                    subset="val", dataset='ActivityNet', logger=logger)
    test_mAP = anet_detection.evaluate()

    print_str = "test_fg_cls_loss:{:.3f}  test_bg_cls_loss:{:.3f}  test_abg_cls_loss:{:.3f}\n".format(np.mean(fg_loss_stack), np.mean(bg_loss_stack), np.mean(abg_loss_stack)) + \
                "test_pseudo_instance_loss:  {:.3f}\n".format(np.mean(pseudo_instance_loss_stack)) + \
                "test_loss:{:.3f}\n".format(np.mean(loss_stack)) + \
                "test acc:{:.3f}\n".format(test_acc) + \
                "test_mAP:{:.3f}\n".format(test_mAP)

    print(print_str, flush=True)
    if logger:
        logger.write(print_str + '\n')

    test_log_dict = {}
    test_log_dict["test_fg_cls_loss"] = np.mean(fg_loss_stack)
    test_log_dict["test_bg_cls_loss"] = np.mean(bg_loss_stack)
    test_log_dict["test_abg_cls_loss"] = np.mean(abg_loss_stack)
    test_log_dict["test_pseudo_instance_loss"] = np.mean(pseudo_instance_loss_stack)
    test_log_dict["test_loss"] = np.mean(loss_stack)
    test_log_dict["test_acc"] = test_acc
    test_log_dict["test_mAP"] = test_mAP
    return test_log_dict


def generate_pseudo_segment(args, model, dataloader, step=0):
    model.eval()
    device = args.device
    save_dir = args.save_dir
    
    test_num_correct = 0
    test_num_total = 0
    
    test_final_result = dict()
    test_final_result['version'] = 'VERSION 1.3'
    test_final_result['results'] = {}
    test_final_result['class_score'] = {}
    test_final_result['external_data'] = {'used': True, 'details': 'Features from I3D Net'}
    
    cas_thresh_list = np.arange(0.1, 0.305, 0.005)
    for vid_name, input_feature, vid_label, vid_len, proposal_bbox, proposal_count_by_video, pseudo_instance_label, dynamic_segment_weights_cumsum in tqdm(dataloader):
        input_feature = input_feature.to(device)
        vid_label = vid_label.to(device)
        proposal_bbox = proposal_bbox.to(device)
        pseudo_instance_label = pseudo_instance_label.to(device)

        #--------------------------------------------------------------------------#
        #--------------------------------------------------------------------------#
        fg_cls, bg_cls, temp_att, cas, fg_cas, bg_cas, uncertainty_pred = \
            model(input_feature, proposal_bbox=proposal_bbox, proposal_count_by_video=proposal_count_by_video, vid_len=vid_len, vid_name=vid_name)

        vid_len = vid_len[0]
        t_factor = (args.segment_frames_num * vid_len) / (args.frames_per_sec * args.test_upgrade_scale * input_feature.shape[1])
        vid_duration = args.segment_frames_num * vid_len / args.frames_per_sec
        #--------------------------------------------------------------------------#
        #--------------------------------------------------------------------------#
        temp_cas = fg_cas
        fg_score = fg_cls[:, :args.action_cls_num]
        label_np = vid_label.cpu().numpy()
        score_np = fg_score.cpu().numpy() # (K)
        pred_np = np.zeros_like(score_np)
        pred_np[score_np >= args.cls_threshold] = 1
        pred_np[score_np < args.cls_threshold] = 0
        correct_pred = np.sum(label_np == pred_np, axis=1)
        test_num_correct += np.sum((correct_pred == args.action_cls_num))
        test_num_total += correct_pred.shape[0]
        #--------------------------------------------------------------------------#
        #--------------------------------------------------------------------------#
        # GENERATE PROPORALS.
        temp_cls_score_np = temp_cas[:, :, :args.action_cls_num].cpu().numpy()
        temp_cls_score_np = np.reshape(temp_cls_score_np, (temp_cas.shape[1], args.action_cls_num, 1))
        temp_att_fg_score_np = temp_att[:, :, 0].unsqueeze(2).expand([-1, -1, args.action_cls_num]).cpu().numpy()
        temp_att_fg_score_np = np.reshape(temp_att_fg_score_np, (temp_cas.shape[1], args.action_cls_num, 1))

        score_np = np.reshape(score_np, (-1))
        if score_np.max() > args.cls_threshold:
            cls_prediction = np.array(np.where(score_np > args.cls_threshold)[0]) #(num_pred_classes)
        else:
            cls_prediction = np.array([np.argmax(score_np)], dtype=int)
        temp_cls_score_np = temp_cls_score_np[:, cls_prediction] #(T, num_pred_classes, 1)
        temp_att_fg_score_np = temp_att_fg_score_np[:, cls_prediction]
        
        int_temp_cls_scores = upgrade_resolution(temp_cls_score_np, args.test_upgrade_scale) #(T*upscale, num_pred_classes, 1)
        int_temp_att_fg_score_np = upgrade_resolution(temp_att_fg_score_np, args.test_upgrade_scale)
        
        proposal_dict = {}
        # CAS based proposal generation
        for cas_thresh in cas_thresh_list:
            tmp_int_cas = int_temp_cls_scores.copy() #(T*upscale, num_pred_classes, 1)
            zero_location = np.where(tmp_int_cas < cas_thresh)
            tmp_int_cas[zero_location] = 0
            
            tmp_seg_list = []
            for c_idx in range(len(cls_prediction)):
                pos = np.where(tmp_int_cas[:, c_idx] >= cas_thresh)
                tmp_seg_list.append(pos)
            
            props_list = get_proposal_oic(tmp_seg_list, (0.70*tmp_int_cas + 0.30*int_temp_att_fg_score_np), cls_prediction, 
                    score_np, t_factor, lamb=0.150, gamma=0.0, dynamic_segment_weights_cumsum=dynamic_segment_weights_cumsum[0], vid_duration=vid_duration)
        
            for i in range(len(props_list)):
                if len(props_list[i]) == 0:
                    continue
                class_id = props_list[i][0][0]
                
                if class_id not in proposal_dict.keys():
                    proposal_dict[class_id] = []
                
                proposal_dict[class_id] += props_list[i]
        
        # NMS 
        final_proposals = []
        for class_id in proposal_dict.keys():
            final_proposals.append(nms(proposal_dict[class_id], args.nms_thresh))
        test_final_result['results'][vid_name[0]] = result2json(final_proposals, args.class_name_lst)

    pseudo_proposals_path = os.path.join(save_dir, "pseudo_proposals_step{}.json".format(step))
    with open(pseudo_proposals_path, 'w') as f:
        json.dump(test_final_result, f)

    return test_final_result


def generate_dynamic_segment_weights(args, pseudo_segment_dict, step=0):
    dynamic_segment_weight_path = os.path.join(args.save_dir, 'dynamic_segment_weights_pred_step{}'.format(step))
    os.makedirs(dynamic_segment_weight_path, exist_ok=True)
    for vid_name in pseudo_segment_dict['results']:
        if os.path.isfile(os.path.join('data/ActivityNet13/train', vid_name+".npy")):
            feature = np.load(os.path.join('data/ActivityNet13/train', vid_name+".npy"))
        elif os.path.isfile(os.path.join('data/ActivityNet13/test', vid_name+".npy")):
            feature = np.load(os.path.join('data/ActivityNet13/test', vid_name+".npy"))
        vid_len = feature.shape[0]

        label_set = set()
        if 'validation' in vid_name:
            for ann in args.gt_dict[vid_name]["annotations"]:
                label_set.add(ann['label'])
        else:
            for pred in pseudo_segment_dict['results'][vid_name]:
                label_set.add(pred['label'])

        prediction_list_all = []
        for label in label_set:
            prediction_list = []
            for pred in pseudo_segment_dict['results'][vid_name]:
                if pred['label'] == label:
                    t_start = pred["segment"][0]
                    t_end = pred["segment"][1]
                    prediction_list.append([t_start, t_end, pred["score"], pred["label"]])
            prediction_list = sorted(prediction_list, key=lambda k: k[2], reverse=True)

            # select top Q% segments
            segment_score_list = []
            for pred in prediction_list:
                segment_score_list.append(pred[2])
            segment_score = np.array(segment_score_list)
            segment_score_cumsum = np.cumsum(segment_score)
            if segment_score_cumsum.shape[0] > 0:
                score_thres = np.max(segment_score_cumsum) * args.alpha
            else:
                score_thres = 0
                assert(len(prediction_list) == 0), 'num_segments not equal to 0'
            selected_proposal_count_by_video = np.where(segment_score_cumsum <= score_thres)[0].shape[0]
            prediction_list_all += prediction_list[:selected_proposal_count_by_video]

        time_to_index_factor = 25 / 16
        proposal_list = []
        for segment in prediction_list_all:
            t_start = segment[0]
            t_end = segment[1]
            t_mid = (t_start + t_end) / 2
            segment_duration = t_end - t_start
            index_start = max(round((t_mid - (args.delta+0.5) * segment_duration) * time_to_index_factor), 0)
            index_end = min(round((t_mid + (args.delta+0.5) * segment_duration) * time_to_index_factor), vid_len-1)
            if index_start < index_end:
                proposal_list.append([index_start, index_end])
        proposal_list = sorted(proposal_list, key=lambda k: k[0], reverse=True)

        upscale_duration = args.gamma * (2 * args.delta + 1)
        dynamic_segment_weights = np.ones((vid_len,), dtype=float)
        for proposal in proposal_list:
            index_start = proposal[0]
            index_end = proposal[1]
            if (index_end - index_start + 1) <= float(upscale_duration):
                for index in range(index_start, index_end+1):
                    dynamic_segment_weights[index] = max(dynamic_segment_weights[index], min(float(upscale_duration) / (index_end - index_start + 1), float(upscale_duration)))

        ### normalize the weights of fg segments ###
        fg_pos = np.where(dynamic_segment_weights > 1.0)
        fg_temp_list = np.array(fg_pos)[0]
        if fg_temp_list.any():
            grouped_fg_temp_list = grouping(fg_temp_list)
            for k in range(len(grouped_fg_temp_list)):
                segment_score_sum = np.sum(dynamic_segment_weights[grouped_fg_temp_list[k]])
                segment_score_sum_round = np.round(segment_score_sum)
                dynamic_segment_weights[grouped_fg_temp_list[k]] = segment_score_sum_round * dynamic_segment_weights[grouped_fg_temp_list[k]] / segment_score_sum
        np.save(os.path.join(dynamic_segment_weight_path, "{}.npy".format(vid_name)), dynamic_segment_weights)
    return



def main(args):
    torch.set_printoptions(precision=4)
    os.environ['CUDA_VIVIBLE_DEVICES'] = args.gpu
    worker_init_fn = np.random.seed(args.seed)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    
    exp_id = 't{}_step{}_e{}'.format(args.sample_segments_num, args.num_steps, args.epochs_per_step)
    exp_id += f'_gamma{args.gamma}_alpha{args.alpha}_delta{args.delta}'
    if args.suffix:
        exp_id += f'_{args.suffix}'
    save_dir = os.path.join("checkpoints", args.dataset, args.outdir, exp_id)

    args.save_dir = save_dir
    args.device = device
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    logger = open('{}/log.txt'.format(save_dir), 'a+')
    logger.write(str(args) + '\n')

    model_list = []
    optimizer_list = []
    criterion_list = []
    train_dataset_list = []
    test_dataset_list = []
    full_dataset_list = []
    for step in range(args.num_steps+1):
        if step == 0:
            # For the first step, we use the base model
            model = ASMLoc_Base(args)
            criterion = ASMLoc_Base_Loss(args)
            train_dataset = build_dataset(args, phase="train", sample="random", step=step, logger=logger) 
            test_dataset = build_dataset(args, phase="test", sample="uniform", step=step, logger=logger)
            full_dataset = build_dataset(args, phase="full", sample="uniform", step=step, logger=logger)
        else:
            # For the following steps, we use the full model
            model = ASMLoc(args)
            criterion = ASMLoc_Loss(args)
            train_dataset = build_dataset(args, phase="train", sample="dynamic_random", step=step, logger=logger) 
            test_dataset = build_dataset(args, phase="dynamic_test", sample="dynamic_uniform", step=step, logger=logger) 
            full_dataset = build_dataset(args, phase="dynamic_full", sample="dynamic_uniform", step=step, logger=logger)
        train_dataset_list.append(train_dataset)
        test_dataset_list.append(test_dataset)
        full_dataset_list.append(full_dataset)

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
        model_list.append(model)
        optimizer_list.append(optimizer)
        criterion_list.append(criterion)

    if args.checkpoint or args.test:
        checkpoint_path = args.checkpoint
    elif not args.no_resume:
        checkpoint_path = os.path.join(save_dir, 'model_latest.pth')
    else:
        checkpoint_path = None

    best_test_mAP = 0
    step = 0
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print("load checkpoint from {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=device)
        best_test_mAP = checkpoint['best_test_mAP']
        if not args.reset_epoch:
            args.start_epoch = checkpoint['epoch']
        step = min((args.start_epoch-1) // args.epochs_per_step, args.num_steps)
    model = model_list[step]
    optimizer = optimizer_list[step]
    criterion = criterion_list[step]
    if checkpoint_path and os.path.isfile(checkpoint_path):
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    if not args.test:
        train_dataloader = DataLoader(train_dataset_list[step], batch_size=args.batch_size, shuffle=True, 
                                    num_workers=args.num_workers, drop_last=False, collate_fn=my_collate_fn, worker_init_fn=worker_init_fn)
        test_dataloader = DataLoader(test_dataset_list[step], batch_size=1, shuffle=False,
                                    num_workers=args.num_workers, drop_last=False, collate_fn=my_collate_fn, worker_init_fn=worker_init_fn)
        full_dataloader = DataLoader(full_dataset_list[step], batch_size=1, shuffle=False,
                                    num_workers=args.num_workers, drop_last=False, collate_fn=my_collate_fn, worker_init_fn=worker_init_fn)

        print(model, flush=True)
        logger.write(str(model) + '\n')
        for epoch_idx in range(args.start_epoch, args.epochs):
            if epoch_idx > 0 and epoch_idx % args.epochs_per_step == 0 and epoch_idx <= args.epochs_per_step * args.num_steps:
                step = epoch_idx // args.epochs_per_step
                with torch.no_grad():
                    pseudo_segment_dict = generate_pseudo_segment(args, model, full_dataloader, step=step)
                    # generate dynamic segment weights according to the predicted pseudo_segment_dict
                    generate_dynamic_segment_weights(args, pseudo_segment_dict, step=step)
                    # pass the generated pseudo_segment_dict into the dataset class to generate the proposal bounding box and pseudo label 
                    train_dataset_list[step].get_proposals(pseudo_segment_dict)
                    test_dataset_list[step].get_proposals(pseudo_segment_dict)
                    full_dataset_list[step].get_proposals(pseudo_segment_dict)

                    train_dataloader = DataLoader(train_dataset_list[step], batch_size=args.batch_size, shuffle=True, 
                                                num_workers=args.num_workers, drop_last=False, collate_fn=my_collate_fn, worker_init_fn=worker_init_fn)
                    test_dataloader = DataLoader(test_dataset_list[step], batch_size=1, shuffle=False,
                                                num_workers=args.num_workers, drop_last=False, collate_fn=my_collate_fn, worker_init_fn=worker_init_fn)
                    full_dataloader = DataLoader(full_dataset_list[step], batch_size=1, shuffle=False,
                                                num_workers=args.num_workers, drop_last=False, collate_fn=my_collate_fn, worker_init_fn=worker_init_fn)
                model = model_list[step]
                optimizer = optimizer_list[step]
                criterion = criterion_list[step]
                print(model, flush=True)
                logger.write(str(model) + '\n')

            train(args, model, train_dataloader, criterion, optimizer, epoch_idx, logger=logger, step=step, num_steps=args.num_steps)
    
            save_checkpoint = {
                'epoch': epoch_idx+1,
                'model_state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_test_mAP': best_test_mAP,
            }

            torch.save(save_checkpoint, '{}/model_latest.pth'.format(save_dir))
            if (epoch_idx+1) % args.epochs_per_step == 0 and epoch_idx <= args.epochs_per_step * args.num_steps:
                torch.save(save_checkpoint, '{}/model_step{}_last_epoch.pth'.format(save_dir, step))
            if (epoch_idx+1) % args.eval_freq == 0 or epoch_idx > args.epochs - 10:
                with torch.no_grad():
                    test_log_dict = test(args, model, test_dataloader, criterion, logger=logger, step=step)
                    test_mAP = test_log_dict["test_mAP"]

                if test_mAP > best_test_mAP:
                    best_test_mAP = test_mAP
                    torch.save(save_checkpoint, '{}/model_step{}_best_epoch.pth'.format(save_dir, step))
                test_log_dict['best_test_mAP'] = best_test_mAP

                time_now_string = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                print_str = "[{}] Epoch:{}/{}, step:{}/{}, test_mAP:{:.4f}, best_test_mAP:{:.4f}\n".format(
                            time_now_string, epoch_idx, args.epochs, step, args.num_steps, test_mAP, best_test_mAP) + \
                            "-------------------------------------------------------------------------------"
                print(print_str, flush=True)
                logger.write(print_str + '\n')
                logger.flush()

    elif args.test:
        with torch.no_grad():
            for step in range(args.num_steps+1):
                model = model_list[step]
                criterion = criterion_list[step]
                if step < args.num_steps:
                    # during each step, we predict the pseudo segments at the last epoch
                    print("load checkpoint from {}".format(os.path.join(checkpoint_path, "model_step{}_last_epoch.pth".format(step))))
                    checkpoint = torch.load(os.path.join(checkpoint_path, "model_step{}_last_epoch.pth".format(step)), map_location=device)
                else:
                    # for the final step, we save the model weights with best mAP 
                    print("load checkpoint from {}".format(os.path.join(checkpoint_path, "model_step{}_best_epoch.pth".format(step))))
                    checkpoint = torch.load(os.path.join(checkpoint_path, "model_step{}_best_epoch.pth".format(step)), map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                test_dataloader = DataLoader(test_dataset_list[step], batch_size=1, shuffle=False,
                                num_workers=args.num_workers, drop_last=False, collate_fn=my_collate_fn, worker_init_fn=worker_init_fn)
                full_dataloader = DataLoader(full_dataset_list[step], batch_size=1, shuffle=False,
                                            num_workers=args.num_workers, drop_last=False, collate_fn=my_collate_fn, worker_init_fn=worker_init_fn)

                test_log_dict = test(args, model, test_dataloader, criterion, logger=logger)
                test_mAP = test_log_dict["test_mAP"]
                print_str = "step:{}/{}, test_mAP:{:.4f}\n".format(step, args.num_steps, test_mAP)
                print(print_str, flush=True)
                logger.write(print_str + '\n')
                logger.flush()

                if step < args.num_steps:
                    pseudo_segment_dict = generate_pseudo_segment(args, model, full_dataloader, step=step)
                    generate_dynamic_segment_weights(args, pseudo_segment_dict, step=step+1)

                    # pass the pseudo_segment_dict into the dataset class to generate the proposal bbox and pseudo label  
                    test_dataset_list[step+1].get_proposals(pseudo_segment_dict)
                    full_dataset_list[step+1].get_proposals(pseudo_segment_dict)


if __name__ == "__main__":
    args = build_args(dataset="ActivityNet")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    main(args)