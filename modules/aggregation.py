import torch
import torch.nn as nn
from .cross_attention_layer import CrossAttentionEncoderLayer, CrossAttentionTransformerEncoder
from .temporal_transformer import TemporalTransformer
from .positional_encoding import LearnedPositionalEncoding, FixedPositionalEncoding

class TextFusion(nn.Module):
    def __init__(self, config):
        super(TextFusion, self).__init__()
        self.fusion = config.model.text.fusion
        self.n_past_actions = config.data.n_past_actions
        self.video = config.model.video

        if self.fusion == "concat":
            emd_input = 768 * self.n_past_actions
            self.proj = nn.Sequential(nn.Linear(emd_input, config.model.embedding_dim), nn.LayerNorm(config.model.embedding_dim))
            
            
        elif self.fusion == "self-attention":
            enc_layer = nn.TransformerEncoderLayer(d_model=config.model.embedding_dim, 
                                                    nhead=config.model.text.num_heads)
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=config.model.text.num_layers)
            if config.model.text.positional_encoding == 'learned':    
                self.positional_encoding = LearnedPositionalEncoding(config.data.n_past_actions, config.model.embedding_dim, config.data.n_past_actions)
            elif config.model.text.positional_encoding == 'fixed':
                self.positional_encoding = FixedPositionalEncoding(config.model.embedding_dim)
            else:
                self.positional_encoding = None
        elif self.fusion == "stack":
            pass
        else:
            raise ValueError("Invalid fusion strategy for text")

    def forward(self, text):
        if self.fusion == "concat":
            # text is BxNxD, where N is the number of past actions
            # convert to Bx(NxD)
            x = text.view(text.size(0), -1)
            x = self.proj(x)
            x = x.unsqueeze(1)
        elif self.fusion == "self-attention":
            if self.positional_encoding is not None:
                text = self.positional_encoding(text)
            x = self.transformer(text)
            x = x.mean(dim=1)
            x = x.unsqueeze(1)
        elif self.fusion == "stack":
            x = text
        return x

class VisualFusion(nn.Module):
    def __init__(self, config):
        super(VisualFusion, self).__init__()
        input_size = config.model.embedding_dim
        self.depth = config.data.depth_features is not None
        self.pose = config.data.pose_source is not None

        self.fusion = config.model.visual.fusion
        if self.fusion == "concat":
            input_size *= 2
        elif self.fusion == "sum":
            pass
        elif self.fusion == "soft-attention":
            pass
        elif self.fusion == "self-attention":
            enc_layer = nn.TransformerEncoderLayer(d_model=config.model.embedding_dim, 
                                                    nhead=config.model.visual.num_heads)
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=config.model.visual.num_layers)
            input_size = config.model.embedding_dim
        elif self.fusion == "cross-attention":
            enc_layer = CrossAttentionEncoderLayer(d_model=config.model.embedding_dim, 
                                                    nhead=config.model.visual.num_heads)
            self.transformer = CrossAttentionTransformerEncoder(enc_layer, num_layers=config.model.visual.num_layers)
            input_size = config.model.embedding_dim
        else:
            pass # Stack
        
        self.output_size = input_size

    def forward(self, visual):
        rgb = visual["rgb"]
        if self.depth:
            depth = visual["depth"]
        if self.pose:
            pose = visual["pose"]
        


        if self.fusion == "concat":
            # x = torch.cat((rgb, depth), dim=1)
            components = [rgb]
            if self.depth:
                components.append(depth)    
            if self.pose:
                components.append(pose)
            x = torch.cat(components, dim=1)
            x = nn.functional.normalize(x, dim=-1)
        elif self.fusion == "sum":
            x = rgb
            if self.depth:
                x += depth
            if self.pose:
                x += pose
            x = nn.functional.normalize(x, dim=-1)
        elif self.fusion == "soft-attention":
            if self.pose:
                raise NotImplementedError("Soft-attention not implemented for pose")
            depth_attention = nn.functional.softmax(depth, dim=-1)
            x = rgb * depth_attention
            x = nn.functional.normalize(x, dim=-1)
        elif self.fusion == "self-attention":
            # combined_seq = torch.stack([rgb, depth], dim=1)
            components = [rgb]
            if self.depth:
                components.append(depth)
            if self.pose:
                components.append(pose)
            combined_seq = torch.stack(components, dim=1)
            x = self.transformer(combined_seq)
            x = x.mean(dim=1)
            x = x.unsqueeze(1)
        elif self.fusion == "cross-attention":
            if self.pose and not self.depth:
                depth = pose
            # elif self.pose and self.depth: # Results in invalid inputs!
            #     rgb = torch.stack([rgb, pose], dim=1)
            #     depth = depth.unsqueeze(1)
            x = self.transformer(rgb, depth) # src (Q), memory  (K,V) (best performance)
            # x = self.transformer(depth, rgb) # src (Q), memory  (K,V)
            x = x.unsqueeze(1)
        else:
            x = torch.stack([rgb, depth], dim=1)
        return x


class AggregationModel(torch.nn.Module):
    def __init__(self, config):
        super(AggregationModel, self).__init__()

        self.ar_supervised = config.model.ar_supervised
        self.verb_noun_supervised = config.model.verb_noun_supervised
        num_classes = config.data.num_classes
        if self.verb_noun_supervised:
            num_verbs = config.data.num_verbs
            num_nouns = config.data.num_nouns
        
        self.rgb = config.data.RGB_features
        self.depth = config.data.depth_features
        self.pose = config.data.pose_source
        self.past_actions = config.data.past_actions
        self.video = config.model.video
        self.video_pretrained = config.model.video_pretrained

        if self.video and self.video_pretrained:
            raise ValueError("Video pretrained features not supported with video input, please set one of them to False")

        if self.video:
            if self.rgb:
                self.rgb_video_transformer = TemporalTransformer(config, "rgb")
            if self.depth:
                self.depth_video_transformer = TemporalTransformer(config, "depth")
        
        if self.video_pretrained:
            pass # TODO: ?

        classifier_input_size = 0
        if self.rgb: # Project to embedding dim and normalize if needed
            input_size = 1536 if config.data.RGB_features == "DINOv2" else 768
            if config.data.RGB_features == "DINOv2":
                input_size = 1536
            elif config.data.RGB_features == "CLIP-ViT-L14-336":
                input_size = 768
            elif config.data.RGB_features == "TSN_BNInception":
                input_size = 1024
            elif config.data.RGB_features == "jepa_rgb":
                input_size = 1408
            elif config.data.RGB_features == "timesformer_rgb":
                input_size = 768
            else:
                raise ValueError("Invalid RGB features")
            
            self.rgb_norm = nn.Sequential(nn.Linear(input_size, config.model.embedding_dim), nn.LayerNorm(config.model.embedding_dim)) if config.model.embedding_dim != input_size else nn.Identity()
            classifier_input_size += config.model.embedding_dim

        if self.depth: # Project to embedding dim and normalize if needed
            if config.data.depth_features == "DINOv2":
                input_size = 1536
            elif config.data.depth_features == "CLIP-ViT-L14-336":
                input_size = 768
            elif config.data.depth_features == "jepa_depth":
                input_size = 1408
            elif config.data.depth_features == "timesformer_depth":
                input_size = 768
            else:
                raise ValueError("Invalid depth features")
            self.depth_norm = nn.Sequential(nn.Linear(input_size, config.model.embedding_dim), nn.LayerNorm(config.model.embedding_dim)) if config.model.embedding_dim != input_size else nn.Identity()
            classifier_input_size += config.model.embedding_dim

        if self.pose:
            classifier_input_size += config.model.embedding_dim
            self.pose_norm = nn.Identity() # Normalization downgrades performance, use Identity to keep the code in forward cleaner
            
            

        if self.rgb and (self.depth or self.pose): # Visual fusion
            self.visual_fusion = VisualFusion(config)
            classifier_input_size = self.visual_fusion.output_size


        if self.past_actions: # No normalization since it's a CLS token
            self.text_fusion = TextFusion(config)
            classifier_input_size += 768
            

        self.fusion = config.model.fusion
        if self.fusion and (not self.past_actions): # Single test  and not self.pose
            raise ValueError("Fusion strategy specified but no additional modalities to fuse with")

        if self.fusion == "concat":
            self.reshape = classifier_input_size
        elif self.fusion == "sum":
            classifier_input_size = config.model.embedding_dim
        elif self.fusion == "self-attention":
            enc_layer = nn.TransformerEncoderLayer(d_model=config.model.embedding_dim, 
                                                nhead=config.model.num_heads)
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=config.model.num_layers)
            classifier_input_size = config.model.embedding_dim
        elif self.fusion:
            raise ValueError("Invalid fusion strategy")


        if self.ar_supervised:
            self.ar_classifier = torch.nn.Sequential(
                torch.nn.Linear(classifier_input_size, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, num_classes)
            )

        if self.verb_noun_supervised:
            self.verb_classifier = torch.nn.Sequential(
                torch.nn.Linear(classifier_input_size, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, num_verbs)
            )
            self.noun_classifier = torch.nn.Sequential(
                torch.nn.Linear(classifier_input_size, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, num_nouns)
            )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(classifier_input_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )


    def forward(self, data):
        if self.video:
            if self.rgb:
                data["rgb"] = self.rgb_video_transformer(data["rgb"])
            if self.depth:
                data["depth"] = self.depth_video_transformer(data["depth"])

        visual = {}
        if self.rgb:
            visual["rgb"] = self.rgb_norm(data["rgb"])
        
        if self.depth:
            visual["depth"] = self.depth_norm(data["depth"])

        if self.pose:
            visual["pose"] = self.pose_norm(data["pose"])
        if len(visual) >= 2:
            combined_seq = self.visual_fusion(visual)
        elif len(visual) == 1:
            visual = list(v for k, v in visual.items())
            combined_seq = visual[0].unsqueeze(1)
        else:
            combined_seq = None

        
        if self.past_actions: # No normalization since it's a CLS token
            text_data = self.text_fusion(data["past_actions"])
            combined_seq = torch.cat([combined_seq, text_data], dim=1) if combined_seq is not None else text_data # TODO: review for sum and concat
        elif self.fusion is not None:
            raise ValueError("Fusion strategy specified but no past actions")
        
        if self.fusion is None:
            x = combined_seq.squeeze(1)


        

        if self.fusion == "concat":
            x = combined_seq.view(combined_seq.size(0), self.reshape)
            x = nn.functional.normalize(x, dim=-1)
        elif self.fusion == "sum":
            x = nn.functional.normalize(combined_seq, dim=-1)
            x = x.sum(dim=1) # (batch, seq_len, embedding_dim) -> (batch, embedding_dim)
        elif self.fusion == "self-attention":
            x = self.transformer(combined_seq)
            x = x.mean(dim=1)

        if self.ar_supervised:
            return {"ar": self.ar_classifier(x), "fut": self.classifier(x)}
        
        if self.verb_noun_supervised:
            return {"verb": self.verb_classifier(x), "noun": self.noun_classifier(x), "fut": self.classifier(x)}

        return {"fut": self.classifier(x)}

