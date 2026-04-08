'''
Binary change subheads for Cross-modal CD.
'''
import torch
import torch.nn as nn
from mmcv.utils import build_from_cfg
from mmcv.cnn import ConvModule
import torch.nn.functional as F

from mmseg.models.cd.fhd import split_batches
from mmseg.models.decode_heads import SegformerHead
from ..bc_heads import BaseHeadBC, ConcatModule, KConcatModule, ContrastiveModule
from ..map_encoders import MAP_ENCODERS
from ...builder import HEADS, build_loss
from ....ops import resize

@HEADS.register_module()
class CrossModalConcathead(BaseHeadBC):
    '''
    BC head for cross-modal concatentation baseline.
    '''
    def __init__(self, feature_strides, map_encoder, **kwargs):
        super(CrossModalConcathead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        num_inputs = len(self.in_channels)

        map_encoder['num_scales'] = len(self.in_index)
        map_encoder['ignore_index'] = self.ignore_index
        map_encoder['norm_cfg'] = self.norm_cfg
        self.map_encoder = build_from_cfg(map_encoder, MAP_ENCODERS)

        self.temporal_fusion_modules = nn.ModuleList(
            [ConcatModule(
                in_channels=self.in_channels[s] + self.map_encoder.out_channels[s],
                out_channels=self.channels,
                norm_cfg=self.norm_cfg
            ) for s in range(num_inputs)]
        )
        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs, gt_semantic_seg_pre):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32; len=3, 1/4,1/8,1/16
        map_features = self.map_encoder(gt_semantic_seg_pre)

        bitemporal_features = []
        for s, module in enumerate(self.temporal_fusion_modules):
            f2 = x[s]
            m1 = map_features[s]
            if m1.shape[2:] != f2.shape[2:]:
                m1 = resize(m1, size=f2.shape[2:], mode='bilinear', align_corners=self.align_corners)

            f = module(features=[f2, m1])
            f = resize(input=f, size=x[0].shape[2:], mode='bilinear', align_corners=self.align_corners)
            bitemporal_features.append(f)

        out = self.fusion_conv(torch.cat(bitemporal_features, dim=1))
        out = self.cls_seg(out)

        return out

@HEADS.register_module()
class CrossModalAttentionHead(BaseHeadBC):
    '''
    BC subhead for cross-modal CD based on our attention-style feature fusion.
    '''
    def __init__(self, feature_strides, map_encoder, extra_branch=False, k=None, **kwargs):
        '''
        :param k: How many channels each feature can attend to.
        '''
        super(CrossModalAttentionHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        num_inputs = len(self.in_channels)
        self.n_semantic_classes = map_encoder['n_semantic_classes']

        map_encoder['num_scales'] = len(self.in_index)
        map_encoder['ignore_index'] = self.ignore_index # TODO: this look at the wrong ignore index (bc instead of sem), but doesn't matter as long as the two indices are equal
        map_encoder['norm_cfg'] = self.norm_cfg
        self.map_encoder = build_from_cfg(map_encoder, MAP_ENCODERS)

        self.extra_branch = extra_branch
        if k is None:
            self.k = self.n_semantic_classes + 1
        else:
            self.k = k

        self.temporal_fusion_modules = nn.ModuleList(
            [KConcatModule(
                in_channels=self.in_channels[s] + self.map_encoder.out_channels[s],
                out_channels=self.channels,
                k=self.k + (1 if self.extra_branch else 0),
                norm_cfg=self.norm_cfg
            ) for s in range(num_inputs)]
        )
        self.attention_weights = nn.ModuleList(
            [nn.Conv2d(
                in_channels=self.map_encoder.out_channels[s],
                out_channels=self.k * self.channels,
                kernel_size=1,
                ) for s in range(num_inputs)]
        )
        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs, gt_semantic_seg_pre):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32; len=3, 1/4,1/8,1/16
        map_features = self.map_encoder(gt_semantic_seg_pre)
        bitemporal_features = []
        for s, module in enumerate(self.temporal_fusion_modules):
            f2 = x[s]
            m1 = map_features[s]
            if m1.shape[2:] != f2.shape[2:]:
                m1 = resize(m1, size=f2.shape[2:], mode='bilinear', align_corners=self.align_corners)

            h = module(features=[f2, m1])

            if self.extra_branch:
                f_extra = h[:,-self.channels:]
                h = h[:,:-self.channels]

            h_k = h.reshape(
                h.shape[0],
                self.k,
                self.channels,
                h.shape[2],
                h.shape[3]
            ) # (B,K,C,H,W)
            attn_weights = self.attention_weights[s](m1) # (B,KC, H, W)
            attn_weights = attn_weights.reshape(
                h_k.shape[0], 
                self.k, 
                h_k.shape[2],
                h_k.shape[3],
                h_k.shape[4]).softmax(dim=1) # (B,K,C,H,W)
            f = (h_k * attn_weights).sum(dim=1)  # (B,C,H,W)
            if self.extra_branch:
                f = f + f_extra
            f = resize(input=f, size=x[0].shape[2:], mode='bilinear', align_corners=self.align_corners)
            bitemporal_features.append(f)

        out = self.fusion_conv(torch.cat(bitemporal_features, dim=1))
        out = self.cls_seg(out)

        return out

    
@HEADS.register_module()
class CrossModalMapFormerHead(CrossModalAttentionHead):
    '''
    BC subhead for cross-modal MapFormer.
    '''
    def __init__(
        self,
        feature_strides, 
        map_encoder, 
        extra_branch=False, 
        k=None, 
        contrastive_loss_weight=1.0,
        balance_pos_neg=True,
        change_classes=(1,),
        focal_loss=None,
        dice_loss=None,
        separable_loss_weight=0.0,
        hard_negative_ratio=1.0,
        hard_negative_min_kept=0,
        **kwargs
    ):
        super(CrossModalMapFormerHead, self).__init__(
            feature_strides=feature_strides,
            map_encoder=map_encoder,
            extra_branch=extra_branch,
            k=k,
            **kwargs
        )
        self.contrastive_img_forward = SegformerHead(
            align_corners = self.align_corners,
            channels=self.channels,
            dropout_ratio=self.dropout_ratio,
            ignore_index=None,
            in_channels=self.in_channels,
            in_index=self.in_index,
            loss_decode={'type': 'CrossEntropyLoss'}, # not used
            norm_cfg=self.norm_cfg,
            num_classes=self.map_encoder.out_channels[0] # embedding dim here
        )
        self.contrastive_module = ContrastiveModule(
            in_channels_map=None, #self.map_encoder.out_channels[0],
            in_channels_img=None,
            proj_channels=self.map_encoder.out_channels[0],
            loss_weight=contrastive_loss_weight,
            balance_pos_neg=balance_pos_neg,
            change_classes=change_classes,
            align_corners=self.align_corners
        )
        self.change_classes = tuple(int(c) for c in change_classes)
        self.non_change_classes = tuple(c for c in range(self.num_classes) if c not in self.change_classes)
        self.focal_loss = build_loss(focal_loss) if focal_loss is not None else None
        self.dice_loss = build_loss(dice_loss) if dice_loss is not None else None
        self.separable_loss_weight = float(separable_loss_weight)
        self.hard_negative_ratio = float(hard_negative_ratio)
        self.hard_negative_min_kept = int(hard_negative_min_kept)

    def _resize_logits_and_labels(self, seg_logit, seg_label):
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_label = seg_label.squeeze(1).long()
        return seg_logit, seg_label

    def _get_change_target(self, seg_label):
        target = torch.zeros_like(seg_label, dtype=torch.bool)
        for c in self.change_classes:
            target = target | (seg_label == c)
        return target

    def _separable_change_loss(self, seg_logit, seg_label):
        if self.separable_loss_weight <= 0:
            return seg_logit.new_zeros([])

        valid = (seg_label != self.ignore_index)
        if valid.sum() == 0:
            return seg_logit.new_zeros([])

        change_target = self._get_change_target(seg_label).float()
        pos_logit = torch.logsumexp(seg_logit[:, self.change_classes, ...], dim=1)
        neg_logit = torch.logsumexp(seg_logit[:, self.non_change_classes, ...], dim=1)
        change_logit = pos_logit - neg_logit

        loss_map = F.binary_cross_entropy_with_logits(change_logit, change_target, reduction='none')
        if self.hard_negative_ratio >= 1.0:
            return loss_map[valid].mean() * self.separable_loss_weight

        pos_mask = (change_target > 0.5) & valid
        neg_mask = (change_target < 0.5) & valid
        keep_mask = pos_mask.clone()

        if neg_mask.any():
            neg_losses = loss_map[neg_mask]
            k_ratio = int(self.hard_negative_ratio * neg_losses.numel())
            k = max(self.hard_negative_min_kept, k_ratio)
            k = min(max(k, 0), neg_losses.numel())
            if k > 0:
                topk = torch.topk(neg_losses, k=k, largest=True, sorted=False).values
                threshold = topk.min()
                keep_mask = keep_mask | (neg_mask & (loss_map >= threshold))

        if keep_mask.any():
            loss = loss_map[keep_mask].mean()
        else:
            loss = loss_map[valid].mean()
        return loss * self.separable_loss_weight
        

    def forward_train(
        self,
        inputs,
        img_metas,
        train_cfg,
        gt_semantic_seg,
        gt_semantic_seg_pre,
        gt_semantic_seg_post=None
    ):
        #bc_logit = self.forward(inputs=inputs, gt_semantic_seg_pre=gt_semantic_seg_pre)
        #def forward(self, inputs, gt_semantic_seg_pre):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32; len=3, 1/4,1/8,1/16
        map_features = self.map_encoder(gt_semantic_seg_pre)
        f2_list = []
        bitemporal_features = []
        contrastive_losses = []
        for s, module in enumerate(self.temporal_fusion_modules):
            f2 = x[s]
            m1 = map_features[s]
            if m1.shape[2:] != f2.shape[2:]:
                m1_ = resize(m1, size=f2.shape[2:], mode='bilinear', align_corners=self.align_corners)
            else:
                m1_ = m1

            h = module(features=[f2, m1_])

            if self.extra_branch:
                f_extra = h[:,-self.channels:]
                h = h[:,:-self.channels]

            h_k = h.reshape(
                h.shape[0],
                self.k,
                self.channels,
                h.shape[2],
                h.shape[3]
            ) # (B,K,C,H,W)
            attn_weights = self.attention_weights[s](m1_) # (B,KC, H, W)
            attn_weights = attn_weights.reshape(
                h_k.shape[0], 
                self.k, 
                h_k.shape[2],
                h_k.shape[3],
                h_k.shape[4]).softmax(dim=1) # (B,K,C,H,W)
            f = (h_k * attn_weights).sum(dim=1)  # (B,C,H,W)
            if self.extra_branch:
                f = f + f_extra
            f = resize(input=f, size=x[0].shape[2:], mode='bilinear', align_corners=self.align_corners)
            bitemporal_features.append(f)
            f2_list.append(f2)

        out = self.fusion_conv(torch.cat(bitemporal_features, dim=1))
        bc_logit = self.cls_seg(out)
        losses = self.losses(seg_logit=bc_logit, seg_label=gt_semantic_seg)
        bc_logit_up, bc_label_up = self._resize_logits_and_labels(
            seg_logit=bc_logit, seg_label=gt_semantic_seg)

        if self.focal_loss is not None:
            losses['loss_focal'] = self.focal_loss(
                bc_logit_up, bc_label_up, ignore_index=self.ignore_index)
        if self.dice_loss is not None:
            losses['loss_dice'] = self.dice_loss(bc_logit_up, bc_label_up)
        if self.separable_loss_weight > 0:
            losses['loss_sep'] = self._separable_change_loss(
                seg_logit=bc_logit_up, seg_label=bc_label_up)

        f2_merged = self.contrastive_img_forward(f2_list)
        contrastive_losses = self.contrastive_module(
            bc=gt_semantic_seg, 
            g1=map_features[0], 
            f2=f2_merged, 
            f1=None
        )
        losses.update(contrastive_losses)
        return losses
