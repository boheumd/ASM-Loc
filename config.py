import json 
import argparse
import numpy as np

_CLASS_NAME = {
    "THUMOS":['BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
              'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving',
              'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump',
              'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput',
              'SoccerPenalty', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking'],

    "ActivityNet":['Applying sunscreen', 'Archery', 'Arm wrestling', 'Assembling bicycle',
                    'BMX', 'Baking cookies', 'Ballet', 'Bathing dog', 'Baton twirling',
                    'Beach soccer', 'Beer pong', 'Belly dance', 'Blow-drying hair', 'Blowing leaves',
                    'Braiding hair', 'Breakdancing', 'Brushing hair', 'Brushing teeth', 'Building sandcastles',
                    'Bullfighting', 'Bungee jumping', 'Calf roping', 'Camel ride', 'Canoeing', 'Capoeira',
                    'Carving jack-o-lanterns', 'Changing car wheel', 'Cheerleading', 'Chopping wood',
                    'Clean and jerk', 'Cleaning shoes', 'Cleaning sink', 'Cleaning windows', 'Clipping cat claws',
                    'Cricket', 'Croquet', 'Cumbia', 'Curling', 'Cutting the grass', 'Decorating the Christmas tree',
                    'Disc dog', 'Discus throw', 'Dodgeball', 'Doing a powerbomb', 'Doing crunches', 'Doing fencing',
                    'Doing karate', 'Doing kickboxing', 'Doing motocross', 'Doing nails', 'Doing step aerobics',
                    'Drinking beer', 'Drinking coffee', 'Drum corps', 'Elliptical trainer', 'Fixing bicycle', 'Fixing the roof',
                    'Fun sliding down', 'Futsal', 'Gargling mouthwash', 'Getting a haircut', 'Getting a piercing', 'Getting a tattoo',
                    'Grooming dog', 'Grooming horse', 'Hammer throw', 'Hand car wash', 'Hand washing clothes', 'Hanging wallpaper',
                    'Having an ice cream', 'High jump', 'Hitting a pinata', 'Hopscotch', 'Horseback riding', 'Hula hoop',
                    'Hurling', 'Ice fishing', 'Installing carpet', 'Ironing clothes', 'Javelin throw', 'Kayaking', 'Kite flying',
                    'Kneeling', 'Knitting', 'Laying tile', 'Layup drill in basketball', 'Long jump', 'Longboarding',
                    'Making a cake', 'Making a lemonade', 'Making a sandwich', 'Making an omelette', 'Mixing drinks',
                    'Mooping floor', 'Mowing the lawn', 'Paintball', 'Painting', 'Painting fence', 'Painting furniture',
                    'Peeling potatoes', 'Ping-pong', 'Plastering', 'Plataform diving', 'Playing accordion', 'Playing badminton',
                    'Playing bagpipes', 'Playing beach volleyball', 'Playing blackjack', 'Playing congas', 'Playing drums',
                    'Playing field hockey', 'Playing flauta', 'Playing guitarra', 'Playing harmonica', 'Playing ice hockey',
                    'Playing kickball', 'Playing lacrosse', 'Playing piano', 'Playing polo', 'Playing pool', 'Playing racquetball',
                    'Playing rubik cube', 'Playing saxophone', 'Playing squash', 'Playing ten pins', 'Playing violin',
                    'Playing water polo', 'Pole vault', 'Polishing forniture', 'Polishing shoes', 'Powerbocking', 'Preparing pasta',
                    'Preparing salad', 'Putting in contact lenses', 'Putting on makeup', 'Putting on shoes', 'Rafting',
                    'Raking leaves', 'Removing curlers', 'Removing ice from car', 'Riding bumper cars', 'River tubing',
                    'Rock climbing', 'Rock-paper-scissors', 'Rollerblading', 'Roof shingle removal', 'Rope skipping',
                    'Running a marathon', 'Sailing', 'Scuba diving', 'Sharpening knives', 'Shaving', 'Shaving legs',
                    'Shot put', 'Shoveling snow', 'Shuffleboard', 'Skateboarding', 'Skiing', 'Slacklining',
                    'Smoking a cigarette', 'Smoking hookah', 'Snatch', 'Snow tubing', 'Snowboarding', 'Spinning',
                    'Spread mulch','Springboard diving', 'Starting a campfire', 'Sumo', 'Surfing', 'Swimming',
                    'Swinging at the playground', 'Table soccer','Tai chi', 'Tango', 'Tennis serve with ball bouncing',
                    'Throwing darts', 'Trimming branches or hedges', 'Triple jump', 'Tug of war', 'Tumbling', 'Using parallel bars',
                    'Using the balance beam', 'Using the monkey bar', 'Using the pommel horse', 'Using the rowing machine',
                    'Using uneven bars', 'Vacuuming floor', 'Volleyball', 'Wakeboarding', 'Walking the dog', 'Washing dishes',
                    'Washing face', 'Washing hands', 'Waterskiing', 'Waxing skis', 'Welding', 'Windsurfing', 'Wrapping presents',
                    'Zumba'],
}


_DATASET_HYPER_PARAMS = {
    
    "THUMOS":{
        "seed": 2,
        "epochs": 800,
        "epochs_per_step": 100,
        "num_steps": 3,
        "dropout":0.7,
        "lr":1e-4,
        "weight_decay":5e-5,
        "frames_per_sec":25,
        "segment_frames_num":16,
        "sample_segments_num":750,
        "max_segments_num":200,
        
        "feature_dim":2048,
        "action_cls_num":len(_CLASS_NAME["THUMOS"]),
        "action_cls":_CLASS_NAME["THUMOS"],
        "cls_threshold":0.25,
        "test_upgrade_scale":20,
        "data_dir":"./data/THUMOS14/",
        "test_gt_file":"./data/THUMOS14/gt.json",
        "tiou_thresholds":np.arange(0.1, 1.00, 0.10),
        "nms_thresh":0.45,
        
        "fg_topk_seg":8,
        "bg_topk_seg":3,

        "lamb_fg":1.0,
        "lamb_bg":0.5, 
        "lamb_abg":0.5,

        "delta": 0.5,
        "alpha": 0.7,
        "gamma": 6,
        "beta": 0.2,

    },
    
    "ActivityNet":{
        "seed": 2,
        "epochs": 100,
        "epochs_per_step": 50,
        "num_steps": 1,
        "dropout":0.7,
        "lr":1e-4,
        "weight_decay":0.001,
        "frames_per_sec":25,
        "segment_frames_num":16,
        "sample_segments_num":150,
        "max_segments_num":100,
        
        "feature_dim":2048,
        "action_cls_num":len(_CLASS_NAME["ActivityNet"]),
        "action_cls":_CLASS_NAME["ActivityNet"],
        "cls_threshold":0.10,
        "test_upgrade_scale":20,
        "data_dir":"./data/ActivityNet13",
        "test_gt_file":"./data/ActivityNet13/gt.json",
        "tiou_thresholds":np.arange(0.50, 1.00, 0.05),
        "nms_thresh":0.90,
        
        "fg_topk_seg":2,
        "bg_topk_seg":10,

        "lamb_fg":5.0, 
        "lamb_bg":0.5, 
        "lamb_abg":0.5,

        "delta": 0,
        "alpha": 0.3,
        "gamma": 10,
        "beta": 0.2,
    },
} 

def build_args(dataset=None):
    parser = argparse.ArgumentParser("This script is used for the weakly-supervised temporal aciton localization task.")
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--num_workers", '-j', default=4, type=int)
    parser.add_argument("--batch_size", '-b', default=16, type=int)
    parser.add_argument("--eval_freq", default=5, type=int)
    parser.add_argument("--lr", default=None, type=float)
    
    parser.add_argument("--outdir", default='', type=str)
    parser.add_argument("--suffix", default='', type=str)
    parser.add_argument("--pred_segment_path", default=None, type=str)
    
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--no_resume", action="store_true", help='esume from last exp')
    parser.add_argument("--reset_epoch", action="store_true", help='reset epoch to 0')

    args = parser.parse_args()
    if dataset is not None:
        args.dataset = dataset
    # Based on the selected dataset, we set dataset specific hyper-params. 
    args.seed = _DATASET_HYPER_PARAMS[args.dataset]["seed"]
    args.epochs = _DATASET_HYPER_PARAMS[args.dataset]["epochs"]
    args.epochs_per_step = _DATASET_HYPER_PARAMS[args.dataset]["epochs_per_step"]
    args.num_steps = _DATASET_HYPER_PARAMS[args.dataset]["num_steps"]
    args.class_name_lst = _CLASS_NAME[args.dataset]
    args.action_cls_num = _DATASET_HYPER_PARAMS[args.dataset]["action_cls_num"]
    args.action_cls = _DATASET_HYPER_PARAMS[args.dataset]["action_cls"]
    
    args.dropout = _DATASET_HYPER_PARAMS[args.dataset]["dropout"]
    if args.lr is None:
        args.lr = _DATASET_HYPER_PARAMS[args.dataset]["lr"]
    args.weight_decay = _DATASET_HYPER_PARAMS[args.dataset]["weight_decay"]
    
    args.frames_per_sec = _DATASET_HYPER_PARAMS[args.dataset]["frames_per_sec"]
    args.segment_frames_num = _DATASET_HYPER_PARAMS[args.dataset]["segment_frames_num"]
    args.sample_segments_num = _DATASET_HYPER_PARAMS[args.dataset]["sample_segments_num"]
    args.max_segments_num = _DATASET_HYPER_PARAMS[args.dataset]["max_segments_num"]
    args.feature_dim =  _DATASET_HYPER_PARAMS[args.dataset]["feature_dim"]
    
    args.cls_threshold = _DATASET_HYPER_PARAMS[args.dataset]["cls_threshold"]
    args.tiou_thresholds = _DATASET_HYPER_PARAMS[args.dataset]["tiou_thresholds"]
    args.test_gt_file_path = _DATASET_HYPER_PARAMS[args.dataset]["test_gt_file"]
    args.data_dir = _DATASET_HYPER_PARAMS[args.dataset]["data_dir"]

    args.test_upgrade_scale = _DATASET_HYPER_PARAMS[args.dataset]["test_upgrade_scale"]
    args.nms_thresh = _DATASET_HYPER_PARAMS[args.dataset]["nms_thresh"]
    
    args.fg_topk_seg = _DATASET_HYPER_PARAMS[args.dataset]["fg_topk_seg"]
    args.bg_topk_seg = _DATASET_HYPER_PARAMS[args.dataset]["bg_topk_seg"]

    args.lamb_fg = _DATASET_HYPER_PARAMS[args.dataset]["lamb_fg"]
    args.lamb_bg = _DATASET_HYPER_PARAMS[args.dataset]["lamb_bg"]
    args.lamb_abg = _DATASET_HYPER_PARAMS[args.dataset]["lamb_abg"]
    
    args.delta = _DATASET_HYPER_PARAMS[args.dataset]["delta"]
    args.alpha = _DATASET_HYPER_PARAMS[args.dataset]["alpha"]
    args.gamma = _DATASET_HYPER_PARAMS[args.dataset]["gamma"]
    args.beta = _DATASET_HYPER_PARAMS[args.dataset]["beta"]

    return args