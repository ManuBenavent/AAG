from dotmap import DotMap
import os, json, torch, pickle
import numpy as np
import json
import torch.nn.functional as F
from collections import Counter, deque

class SingleFrameDataset(torch.utils.data.Dataset):
    def __init__(self, config: DotMap, split: str="train"):
        if config.data.dataset == "Assembly101" and split == "test":
            split = "val"
        self.input = []
        self.split = split
        self.num_classes = config.data.num_classes
        self.num_verbs = config.data.num_verbs
        self.num_nouns = config.data.num_nouns
        self.n_past_actions = config.data.n_past_actions
        self.past_actions_from_ar = config.data.past_actions_from_ar
        self.video = config.model.video
        self.input_frames = config.data.input_frames
        self.video_pretrained = config.model.video_pretrained

        self.ar_supervised = config.model.ar_supervised
        self.save_inference = config.save_inference
        self.verb_noun_supervised = config.model.verb_noun_supervised

        feature_path = os.path.join("/dataset", config.data.dataset, "features")

        # RGB features: PKL file
        self.rgb = config.data.RGB_features
        if self.rgb:
            self.rgb = os.path.join(feature_path, "rgb", self.rgb) + "/{}.npy"
            os.path.join("rgb", self.rgb)

        # Depth features: PKL file
        self.depth = config.data.depth_features
        if self.depth:
            self.depth = os.path.join(feature_path, "depth_" + config.data.depth_source , self.depth) + "/{}.npy"

        if self.video_pretrained:
            if self.rgb:
                self.rgb = os.path.join(feature_path, config.data.RGB_features, "anticipation" if not config.data.action_recognition else "recognition") + "/{}.npy"
            if self.depth:
                self.depth = os.path.join(feature_path, config.data.depth_features, "anticipation" if not config.data.action_recognition else "recognition") + "/{}.npy"
        
        # Past actions: PKL file
        self.past_actions = None
        if config.data.past_actions:
            with open(config.data.past_actions, "rb") as f:
                self.past_actions = pickle.load(f)

        # Pose features: Not implemented
        self.pose_source = config.data.pose_source
        self.pose_path = os.path.join(config.data.path, config.data.dataset, "RCNN_2DPoseFeatures_front_view_dev2")

        # Load annotations
        if config.data.action_recognition:
            self.generate_input_ar(config.data.anno_path)
        else:
            self.generate_input(config.data.anno_path, config.data.t_ant)

    def generate_input_ar(self, anno_path):
        anno_path = os.path.join(anno_path, self.split + "_labels.json")
        json_labels = json.load(open(anno_path))
        if self.past_actions_from_ar:
            with open(self.past_actions_from_ar, "rb") as f:
                predicted_actions = pickle.load(f)
        if self.past_actions:
            past_actions = deque(maxlen=self.n_past_actions)
        else:
            past_actions = []
        for v_name, v_data in json_labels.items():
            for i, act_seg in enumerate(v_data):
                start, end = act_seg['segment']

                if self.verb_noun_supervised:
                    verb_id = act_seg['verb_id']
                    noun_id = act_seg['noun_id']
                else:
                    verb_id = -1
                    noun_id = -1
                self.input.append((i, v_name, end, act_seg['label_id'], -1, verb_id, noun_id, list(past_actions)))
                
                if self.past_actions_from_ar:
                    act_id = predicted_actions[v_name][end]
                    past_actions.append(act_id)
                    print("hola")
                elif self.past_actions:
                    past_actions.append(act_seg['label_id'])


    def generate_input(self, anno_path, t_ant, downsampling_rate=5):
        if self.ar_supervised:
            ar_annotation_path = os.path.join(anno_path, "ar_t1_labels.json")
            ar_labels = json.load(open(ar_annotation_path))

        anno_path = os.path.join(anno_path, self.split + "_labels.json")
        json_labels = json.load(open(anno_path))


        frame_rate = 30
        downsampled_frame_rate = frame_rate // downsampling_rate # 6 anyway (30/5 ikea and assembly, or 12/2 meccano)

        if self.past_actions_from_ar:
            with open(self.past_actions_from_ar, "rb") as f:
                predicted_actions = pickle.load(f)

        
        for v_name, v_data in json_labels.items():
            if self.past_actions:
                past_actions = deque(maxlen=self.n_past_actions)
            else:
                past_actions = []
            for i, act_seg in enumerate(v_data):
                start, end = act_seg['segment']
                ant_point = start - t_ant*downsampled_frame_rate # START - 6 frames = 1 second

                if self.verb_noun_supervised:
                    verb_id = act_seg['verb_id']
                    noun_id = act_seg['noun_id']
                else:
                    verb_id = -1
                    noun_id = -1
                if ant_point < 0: # Skip if ant_point is negative
                    continue
                if self.ar_supervised:
                    ar_id = int(ar_labels[v_name][str(i)])
                else:
                    ar_id = -1
                self.input.append((i, v_name, ant_point, act_seg['label_id'], ar_id, verb_id, noun_id, list(past_actions)))
                if self.past_actions_from_ar:
                    act_id = predicted_actions[v_name][end]
                    past_actions.append(act_id)
                elif self.past_actions:
                    past_actions.append(act_seg['label_id'])

        

    def __len__(self):
        return len(self.input)

    def get_class_weights(self, mode="action"):
        if mode == "action":
            index = 3
            num_items = self.num_classes
        elif mode == "ar":
            index = 4
            num_items = self.num_classes
        elif mode == "verb":
            index = 5
            num_items = self.num_verbs
        elif mode == "noun":
            index = 6
            num_items = self.num_nouns
        # Count occurrences of each class
        label_counts = Counter(act[index] for act in self.input)
        total_samples = len(self.input)

        # Compute weights (Inverse frequency method)
        class_weights = {label: total_samples / count for label, count in label_counts.items()}
        for i in range(num_items):
            if i not in class_weights:
                class_weights[i] = 0

        # Convert to tensor and normalize
        weight_tensor = torch.tensor([class_weights[i] for i in sorted(class_weights.keys())], dtype=torch.float32)
        return weight_tensor.cuda()

    def __getitem__(self, idx):
        anno_id, v_name, ant_point, ant_label_id, ar_id, verb_id, noun_id, past_actions = self.input[idx]

        data = {}
        data["target"] = torch.tensor(ant_label_id)

        # AR supervision
        if self.ar_supervised:
            data["ar_target"] = torch.tensor(ar_id)

        # Verb-Noun supervision
        if self.verb_noun_supervised:
            data["verb_target"] = torch.tensor(verb_id)
            data["noun_target"] = torch.tensor(noun_id)

        start = max(0, ant_point - self.input_frames) # Get the last n frames
        if self.rgb:
            rgb_features = np.load(self.rgb.format(v_name), allow_pickle=True)
            if self.video:
                data['rgb'] = torch.tensor(rgb_features[start:ant_point]).float()
                # add zero padding if the number of frames is less than input_frames
                if data['rgb'].shape[0] < self.input_frames:
                    pad_len = self.input_frames -  data['rgb'].shape[0]
                    pad = torch.zeros((pad_len, data['rgb'].shape[1]), dtype=data['rgb'].dtype)
                    data['rgb'] = torch.cat((pad, data['rgb']), dim=0)  # pad at beginning
            elif self.video_pretrained:
                data['rgb'] = torch.tensor(rgb_features.item()[anno_id]).float()
                # data['rgb'] = torch.mean(data['rgb'], dim=0)  # Average over tokens
            else:
                data['rgb'] = torch.tensor(rgb_features[ant_point]).float()
            del rgb_features

        if self.depth:
            depth_features = np.load(self.depth.format(v_name), allow_pickle=True)
            if self.video:
                data['depth'] = torch.tensor(depth_features[start:ant_point]).float()
                # add zero padding if the number of frames is less than input_frames
                if data['depth'].shape[0] < self.input_frames:
                    pad_len = self.input_frames -  data['depth'].shape[0]
                    pad = torch.zeros((pad_len, data['depth'].shape[1]), dtype=data['depth'].dtype)
                    data['depth'] = torch.cat((pad, data['depth']), dim=0)  # pad at beginning
            elif self.video_pretrained:
                data['depth'] = torch.tensor(depth_features.item()[anno_id]).float()
                # data['depth'] = torch.mean(data['depth'], dim=0)  # Average over tokens
            else:
                data['depth'] = torch.tensor(depth_features[ant_point]).float()
            del depth_features

        if self.past_actions:
            embds = []
            for i in past_actions:
                embds.append(self.past_actions[i].squeeze())
            if len(embds) < self.n_past_actions:
                embds.extend([np.zeros((768,)) for _ in range(self.n_past_actions - len(embds))])
            data['past_actions'] = torch.tensor(np.array(embds)).float()



        if self.pose_source and len(self.pose_source) > 0:
            # Hardcoded - Not used!
            with open(os.path.join(self.pose_path, v_name, f"scan_video_{(ant_point*5):012d}_keypoints_2Dfeatures.json"), 'r') as f:
                pose_values = json.load(f)[0]
            pose_features = []
            for source in self.pose_source:
                if source != "orientations":
                    pose_features.extend(pose_values[source])
                else:
                    pose_features.extend(list(pose_values[source].values()))
            data['pose'] = torch.tensor(pose_features).float().squeeze()
            # Add padding to match the expected size
            if data['pose'].shape[0] != 768: # TODO: parametereize or move to forward
                data['pose'] = F.pad(data['pose'], (0, 768 - data['pose'].shape[0]))

        if self.save_inference:
            data['v_name'] = v_name
            data['end'] = ant_point

        return data