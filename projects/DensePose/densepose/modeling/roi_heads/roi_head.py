# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
from typing import Dict, List, Optional
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.tensor import Tensor

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.structures import ImageList, Instances
from densepose.structures import DensePoseList, DensePoseDataRelative
from detectron2.structures.boxes import Boxes


from .. import (
    build_densepose_data_filter,
    build_densepose_embedder,
    build_densepose_head,
    build_densepose_losses,
    build_densepose_predictor,
    densepose_inference,
)
from densepose.modeling.build import build_my_densepose_predictor

class Decoder(nn.Module):
    """
    A semantic segmentation head described in detail in the Panoptic Feature Pyramid Networks paper
    (https://arxiv.org/abs/1901.02446). It takes FPN features as input and merges information from
    all levels of the FPN into single output.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec], in_features):
        super(Decoder, self).__init__()

        # fmt: off
        self.in_features      = in_features
        feature_strides       = {k: v.stride for k, v in input_shape.items()}
        feature_channels      = {k: v.channels for k, v in input_shape.items()}
        num_classes           = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_NUM_CLASSES
        conv_dims             = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_CONV_DIMS
        self.common_stride    = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_COMMON_STRIDE
        norm                  = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_NORM
        # fmt: on

        self.scale_heads = []
        for in_feature in self.in_features:
            head_ops = []
            head_length = max(
                1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride))
            )
            for k in range(head_length):
                conv = Conv2d(
                    feature_channels[in_feature] if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=get_norm(norm, conv_dims),
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if feature_strides[in_feature] != self.common_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            # pyre-fixme[29]: `Union[nn.Module, torch.Tensor]` is not a function.
            self.add_module(in_feature, self.scale_heads[-1])
        self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)

    def forward(self, features: List[torch.Tensor]):
        for i, _ in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[i])
            else:
                x = x + self.scale_heads[i](features[i])
        x = self.predictor(x)
        return x


@ROI_HEADS_REGISTRY.register()
class DensePoseROIHeads(StandardROIHeads):
    """
    A Standard ROIHeads which contains an addition of DensePose head.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self._init_densepose_head(cfg, input_shape)

    def _init_densepose_head(self, cfg, input_shape):
        # fmt: off
        self.densepose_on          = cfg.MODEL.DENSEPOSE_ON
        if not self.densepose_on:
            return
        self.densepose_data_filter = build_densepose_data_filter(cfg)
        dp_pooler_resolution       = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_RESOLUTION
        dp_pooler_sampling_ratio   = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_SAMPLING_RATIO
        dp_pooler_type             = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_TYPE
        self.use_decoder           = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_ON
        # fmt: on
        if self.use_decoder:
            dp_pooler_scales = (1.0 / input_shape[self.in_features[0]].stride,)
        else:
            dp_pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        in_channels = [input_shape[f].channels for f in self.in_features][0]

        if self.use_decoder:
            self.decoder = Decoder(cfg, input_shape, self.in_features)

        self.densepose_pooler = ROIPooler(
            output_size=dp_pooler_resolution,
            scales=dp_pooler_scales,
            sampling_ratio=dp_pooler_sampling_ratio,
            pooler_type=dp_pooler_type,
        )
        self.densepose_head = build_densepose_head(cfg, in_channels)
        self.densepose_predictor = build_densepose_predictor(
            cfg, self.densepose_head.n_out_channels
        )
        self.densepose_losses = build_densepose_losses(cfg)
        self.embedder = build_densepose_embedder(cfg)

    def _forward_densepose(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the densepose prediction branch.

        Args:
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            instances (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains instances for the i-th input image,
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "densepose" and return it.
        """
        if not self.densepose_on:
            return {} if self.training else instances

        features_list = [features[f] for f in self.in_features]
        if self.training:
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            features_list, proposals = self.densepose_data_filter(features_list, proposals)
            if len(proposals) > 0:
                proposal_boxes = [x.proposal_boxes for x in proposals]

                if self.use_decoder:
                    # pyre-fixme[29]: `Union[nn.Module, torch.Tensor]` is not a
                    #  function.
                    features_list = [self.decoder(features_list)]

                features_dp = self.densepose_pooler(features_list, proposal_boxes)
                densepose_head_outputs = self.densepose_head(features_dp)
                densepose_predictor_outputs = self.densepose_predictor(densepose_head_outputs)
                densepose_loss_dict = self.densepose_losses(
                    proposals, densepose_predictor_outputs, embedder=self.embedder
                )
                return densepose_loss_dict
        else:
            pred_boxes = [x.pred_boxes for x in instances]

            if self.use_decoder:
                # pyre-fixme[29]: `Union[nn.Module, torch.Tensor]` is not a function.
                features_list = [self.decoder(features_list)]

            features_dp = self.densepose_pooler(features_list, pred_boxes)
            if len(features_dp) > 0:
                densepose_head_outputs = self.densepose_head(features_dp)
                densepose_predictor_outputs = self.densepose_predictor(densepose_head_outputs)
            else:
                densepose_predictor_outputs = None

            densepose_inference(densepose_predictor_outputs, instances)
            return instances

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ):
        instances, losses = super().forward(images, features, proposals, targets)
        del targets, images

        if self.training:
            losses.update(self._forward_densepose(features, instances))
        return instances, losses

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """

        instances = super().forward_with_given_boxes(features, instances)
        instances = self._forward_densepose(features, instances)
        return instances





# ================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================
# ================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================
@ROI_HEADS_REGISTRY.register()
class MyROIHeads(StandardROIHeads):
    """
    A Standard ROIHeads which contains an addition of DensePose head.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self._init_densepose_head(cfg, input_shape)

    def _init_densepose_head(self, cfg, input_shape):
        # fmt: off
        self.densepose_on          = cfg.MODEL.DENSEPOSE_ON
        if not self.densepose_on:
            return
        self.densepose_data_filter = build_densepose_data_filter(cfg)
        dp_pooler_resolution       = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_RESOLUTION
        dp_pooler_sampling_ratio   = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_SAMPLING_RATIO
        dp_pooler_type             = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_TYPE
        self.use_decoder           = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_ON
        # fmt: on
        if self.use_decoder:
            dp_pooler_scales = (1.0 / input_shape[self.in_features[0]].stride,)
        else:
            dp_pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        in_channels = [input_shape[f].channels for f in self.in_features][0]

        if self.use_decoder:
            self.decoder = Decoder(cfg, input_shape, self.in_features)

        self.densepose_pooler = ROIPooler(
            output_size=dp_pooler_resolution,
            scales=dp_pooler_scales,
            sampling_ratio=dp_pooler_sampling_ratio,
            pooler_type=dp_pooler_type,
        )
        self.densepose_head = build_densepose_head(cfg, in_channels)
        self.densepose_predictor = build_densepose_predictor(
            cfg, self.densepose_head.n_out_channels
        )

        # ---------------------------------
        self.concate_part_densepose_head = ConcatePartDensePoseHead()
        self.my_densepose_predictor = build_my_densepose_predictor(
            cfg, self.densepose_head.n_out_channels
        )
        # ---------------------------------

        self.densepose_losses = build_densepose_losses(cfg)
        self.embedder = build_densepose_embedder(cfg)
        

    def _forward_densepose(self, 
                           features: Dict[str, torch.Tensor], 
                           instances: List[Instances]):
        """
        Forward logic of the densepose prediction branch.

        Args:
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            instances (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains instances for the i-th input image,
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "densepose" and return it.
        """
        if not self.densepose_on:
            return {} if self.training else instances

        total_loss = {}
        features_list = [features[f] for f in self.in_features]
        if self.training:
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            features_list, proposals = self.densepose_data_filter(features_list, proposals)
            if len(proposals) > 0:
                proposal_boxes = [x.proposal_boxes for x in proposals]

                if self.use_decoder:
                    # pyre-fixme[29]: `Union[nn.Module, torch.Tensor]` is not a
                    #  function.
                    features_list = [self.decoder(features_list)]

                features_dp = self.densepose_pooler(features_list, proposal_boxes)
                
                densepose_head_outputs = self.densepose_head(features_dp)
                densepose_predictor_outputs = self.densepose_predictor(densepose_head_outputs)

                densepose_loss_dict = self.densepose_losses(
                    proposals, densepose_predictor_outputs, embedder=self.embedder
                )

                # ============================================================================
                part_bboxs: List[Boxes] = self.get_part_bbox(proposals, features_list[0].shape[2:])
                
                feature_part = self.densepose_pooler(features_list, part_bboxs) # (K*15, 256, 28, 28)

                feature_concate = self.concatePart2All(
                    feature_dp=features_dp, 
                    feature_part=feature_part
                ) # (K*15, 512, 28, 28)

                part_densepose_head_output = self.concate_part_densepose_head(feature_concate) # (K*15, 512, 28, 28)
                # part_densepose_predictor_output = self.partDenseHead(part_densepose_head_output=part_densepose_head_output) 
                part_densepose_predictor_output = self.my_densepose_predictor(
                    part_densepose_head_output,
                    densepose_predictor_outputs.coarse_segm
                )
                part_densepose_loss_dict = self.densepose_losses(
                    proposals, part_densepose_predictor_output, embedder=self.embedder
                )
                new_part_loss = {
                    # 'loss_part_S' : part_densepose_loss_dict['loss_densepose_S'],
                    'loss_part_I' : part_densepose_loss_dict['loss_densepose_I'],
                    'loss_part_U' : part_densepose_loss_dict['loss_densepose_U'],
                    'loss_part_V' : part_densepose_loss_dict['loss_densepose_V'],               
                }
                total_loss.update(densepose_loss_dict)
                total_loss.update(new_part_loss)


                
                # del part_densepose_head_output
                # del feature_concate
                # del feature_part
                # del part_densepose_predictor_output

                # del part_bboxs
                # del part_densepose_predictor_output
                return total_loss
                # ============================================================================
        else:
            pred_boxes = [x.pred_boxes for x in instances]

            if self.use_decoder:
                # pyre-fixme[29]: `Union[nn.Module, torch.Tensor]` is not a function.
                features_list = [self.decoder(features_list)]

            features_dp = self.densepose_pooler(features_list, pred_boxes)
            if len(features_dp) > 0:
                # sum = 0
                # for i in range(len(pred_boxes)):
                #     sum += pred_boxes[i].tensor.shape[0]
                # print(len(pred_boxes),'-'*100, sum)

                densepose_head_outputs = self.densepose_head(features_dp)
                densepose_predictor_outputs = self.densepose_predictor(densepose_head_outputs)

            # ============================================================================
                # part_bboxs: Boxes = self.get_part_bbox(pred_boxes, features_list[0].shape[2:])
                part_bboxs: List[Boxes] = self.inference_get_part_bbox(
                    pred_boxes, 
                    self.maxIndexSegmentations(densepose_predictor_outputs.coarse_segm),
                    [features_list[i].shape[2:] for i in range(len(features_list))],
                    [instances[i].image_size for i in range(len(instances))]
                )
                
                feature_part = self.densepose_pooler(features_list, part_bboxs) # (K*15, 256, 28, 28)

                feature_concate = self.concatePart2All(
                    feature_dp=features_dp, 
                    feature_part=feature_part
                ) # (K*15, 512, 28, 28)

                part_densepose_head_output = self.concate_part_densepose_head(feature_concate) # (K*15, 512, 28, 28)
                # part_densepose_predictor_output = self.partDenseHead(part_densepose_head_output=part_densepose_head_output) 
                part_densepose_predictor_output = self.my_densepose_predictor(
                    part_densepose_head_output,
                    densepose_predictor_outputs.coarse_segm
                )

                ''' update denspose_predictor_outputs with part_densepose_predictor_output '''
                densepose_predictor_outputs.fine_segm = part_densepose_predictor_output.fine_segm
                densepose_predictor_outputs.u = part_densepose_predictor_output.u
                densepose_predictor_outputs.v = part_densepose_predictor_output.v

                # del part_bboxs
                # del pred_boxes
                # del feature_concate
                # del feature_part
                # del part_densepose_head_output
            # ============================================================================
            else:
                densepose_predictor_outputs = None

            densepose_inference(densepose_predictor_outputs, instances)
            return instances

    def maxIndexSegmentations(self, segmentation: Tensor) -> Tensor:
        assert segmentation.shape[1] == 15
        return torch.argmax(segmentation, axis=1)

    def inference_get_part_bbox(self, pred_box_list: List[Boxes], segmentations: Tensor, feature_list_sizes, image_sizes) -> List[Boxes]:
        part_bbox = []      
        
        for i in range(len(pred_box_list)):
            pred_box_tensor: Tensor = pred_box_list[i].tensor   # (K, 4)
            temp_box = self.getHumanPartBox(
                pred_box_tensor, 
                segmentations=[segmentations[i] for i in range(len(segmentations))]
            )
            # assert temp_box.tensor.shape == (15, 4)
            part_bbox.append(self.mapSegmBox2Feature(temp_box, image_sizes[i], feature_list_sizes[i]))

        return part_bbox

    def partDenseHead(self, part_densepose_head_output: Tensor):
        '''
        DensePoseChartPredictorOutput(
            coarse_segm=self.interp2d(self.ann_index_lowres(head_outputs)),
            fine_segm=self.interp2d(self.index_uv_lowres(head_outputs)),
            u=self.interp2d(self.u_lowres(head_outputs)),
            v=self.interp2d(self.v_lowres(head_outputs)),
        )

        self.interp2d = interpolate(
            tensor_nchw, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )
        '''
        pass
    
    def concatePart2All(self, feature_dp: Tensor, feature_part: Tensor) -> Tensor:  # (k*15, 512, 28, 28)
        assert feature_dp.shape[0] * 15 == feature_part.shape[0]
        assert feature_dp.shape[1] == feature_part.shape[1]

        # feature_concate = torch.zeros(feature_part.shape[0], 512, feature_part.shape[2], feature_part.shape[3])
        f_dp = feature_dp[:, None, :, :, :] # (K, 1, 256, 28, 28)
        f_part = feature_part.reshape(f_dp.shape[0], 15, feature_dp.shape[1], feature_dp.shape[2], feature_part.shape[3]) # (K, 15, 256, 28, 28)
        f_dp = f_dp.repeat(1, 15, 1, 1, 1)
        answer = torch.cat([f_dp, f_part], axis=2)
        assert answer.shape[1:] == (15, 512, 28, 28)
        answer = answer.reshape(-1, 512, 28, 28)

        return answer
    
    def get_part_bbox(self, proposals: List[Instances], feature_list_size) -> List[Boxes]:
        ''' get human body part bounding box '''
        '''
            feature_list_size : (H, W)
        '''
        part_bbox = []
        for proposal in proposals:
            H, W = proposal.image_size
            gt_densepose: DensePoseList = proposal.gt_densepose

            densepose_datas: List[DensePoseDataRelative] = gt_densepose.densepose_datas            
            boxes_xyxy_abs: Boxes = gt_densepose.boxes_xyxy_abs
            proposal_boxes: Boxes = proposal.proposal_boxes            

            '''  没有segmentation的情况 '''
            if len(densepose_datas) == 0: 
                temp_tensor = proposal_boxes.tensor.unsqueeze(1).repeat(1, 15, 1)
                temp_tensor = temp_tensor.reshape(-1, 4)    # (K*15, 4)
                part_bbox.append(Boxes(temp_tensor))
                continue

            inter_box: Boxes = self.getInterBox(boxes_xyxy_abs, proposal_boxes)
            inter_segmentation: List[Tensor] = self.getInterSegmentation(
                orignal_box=boxes_xyxy_abs.tensor, 
                input_inter_box=inter_box.tensor, 
                segmentations=torch.cat([segm.segm.unsqueeze(0) for segm in densepose_datas], axis=0)
            )
            temp_box = self.getHumanPartBox(inter_box.tensor, inter_segmentation)
            part_bbox.append(self.mapSegmBox2Feature(temp_box, proposal.image_size, feature_list_size))

        return part_bbox

    def mapSegmBox2Feature(self, bbox: Boxes, image_size, feature_list_size) -> Boxes:
        '''
            image_size : (H1, W1)
            feature_list_size : (H2, W2)
        '''
        bbox_tensor: Tensor = bbox.tensor  # (k*15, 4)
        assert bbox_tensor.shape[1] == 4  

        bbox_tensor[:, 0] = bbox_tensor[:, 0] * feature_list_size[1] / image_size[1]
        bbox_tensor[:, 2] = bbox_tensor[:, 2] * feature_list_size[1] / image_size[1]
        bbox_tensor[:, 1] = bbox_tensor[:, 1] * feature_list_size[0] / image_size[0]
        bbox_tensor[:, 3] = bbox_tensor[:, 3] * feature_list_size[0] / image_size[0]

        return bbox

    def getHumanPartBox(self, bbox: Tensor, segmentations: List[Tensor]) -> Boxes:
        assert bbox.shape == (len(segmentations), 4)
        add_bbox = torch.cat([bbox[:, :2], bbox[:, :2]], axis=1)
        assert add_bbox.shape == bbox.shape

        ans = []
        length = len(segmentations)
        for i in range(length):
            segmentation: Tensor = segmentations[i]
            temp_segm: Tensor = self.getBoxFromSegmentation(segmentation)
            assert temp_segm.shape == (15, 4)
            temp_segm += add_bbox[i]
            ans.append(temp_segm)

        return Boxes(torch.cat(ans, axis=0))  # (k*15, 4)

    def getBoxFromSegmentation(self, segmentation: Tensor) -> Tensor:
        segmentation = segmentation.long()
        H, W = segmentation.shape
        # coordinate = [torch.Tensor([float('inf'), float('inf'), float('-inf'), float('-inf')]) for i in range(15)]
        # coordinate: List[Tensor] = [torch.Tensor([W - 1, H - 1, 0, 0]) for i in range(15)]
        coordinate: Tensor = torch.tensor([W - 1, H - 1, 0, 0], device=segmentation.device).repeat(15, 1)
        assert coordinate.shape == (15, 4)

        # for i in range(segmentation.shape[0]):
        #     for j in range(segmentation.shape[1]):
        #         index = segmentation[i, j]
        #         coordinate[index][0] = min(coordinate[index][0].item(), j)
        #         coordinate[index][1] = min(coordinate[index][1].item(), i)
        #         coordinate[index][2] = max(coordinate[index][2].item(), j)
        #         coordinate[index][3] = max(coordinate[index][3].item(), i)

        # coordinate[0][:2] = 0
        # coordinate[0][2:] = torch.tensor(segmentation.shape)  # set background box coordinate
        for i in range(15):
            cord: Tensor = (segmentation == i).nonzero(as_tuple=False)
            if cord.numel() != 0:
                coordinate[i][:2] = torch.flip(torch.min(cord, axis=0)[0], (0, ))
                coordinate[i][2:] = torch.flip(torch.max(cord, axis=0)[0], (0, ))
            
            if  (coordinate[i][0] >= coordinate[i][2]) or (coordinate[i][1] >= coordinate[i][3]):
                coordinate[i] = torch.Tensor([0, 0, W - 1, H - 1])

        # coordinate = [coordinate[i].unsqueeze(0).long() for i in range(len(coordinate))]
        return coordinate.float()

    def getInterSegmentation(self, orignal_box: Tensor, input_inter_box: Tensor, segmentations: Tensor):
        '''
            orignal_box : (k, 4)
            inter_box : (k, 4) 
            segmentations : (k, 256, 256)
        '''
        assert orignal_box.shape == input_inter_box.shape
        assert orignal_box.shape[0] == segmentations.shape[0]
        assert segmentations.shape[1:] == (256, 256)
        
        inter_box = torch.clone(input_inter_box)
        segm_size = segmentations[0].shape # (256, 256)
        inter_box[:, :2] -= orignal_box[:, :2]
        inter_box[:, 2:] -= orignal_box[:, :2]  # get delta
        width = orignal_box[:, 2] - orignal_box[:, 0]
        height = orignal_box[:, 3] - orignal_box[:, 1]
        inter_box[:, 0] = inter_box[:, 0] * segm_size[0] / width
        inter_box[:, 2] = inter_box[:, 2] * segm_size[0] / width
        inter_box[:, 1] = inter_box[:, 1] * segm_size[1] / height 
        inter_box[:, 3] = inter_box[:, 3] * segm_size[1] / height 

        ans_semg = []
        inter_box = inter_box.long()
        for i in range(inter_box.shape[0]):
            ans_semg.append(segmentations[i, inter_box[i, 1]:inter_box[i, 3], inter_box[i, 0]:inter_box[i, 2]])

        return ans_semg

    def getInterBox(self, a: Boxes, b: Boxes):
        ta: Tensor = a.tensor  # (k, 4)
        tb: Tensor = b.tensor
        assert ta.shape == tb.shape
        tc: Tensor = torch.cat([torch.max(ta[:, :2], tb[:, :2]), torch.min(ta[:, 2:], tb[:, 2:])], axis=1)
        return Boxes(tc)

            # for densepose_data in gt_densepose.densepose_datas:
            #     dp_data: DensePoseDataRelative = densepose_data
            #     segm = dp_data.segm
            #     print(type(segm))
            #     print(segm.shape)
            #     temp_segm = F.interpolate(segm.unsqueeze(0).unsqueeze(0), size=(60, 60))
            #     temp_segm = temp_segm.squeeze().long()
            #     print(temp_segm.shape)

                # ''' show segmentation'''
                # for i in range(temp_segm.shape[0]):
                #     for j in range(temp_segm.shape[1]):
                #         if temp_segm[i][j] != 0:
                #             print(f"{temp_segm[i][j].long().item():>2d}", end=' ')
                #         else: 
                #             print("   ", end='')
                #     print()


    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ):
        instances, losses = super().forward(images, features, proposals, targets)
        # image_size = images.image_sizes # [(704, 1055), (736, 1103)]
        del targets, images

        if self.training:
            losses.update(self._forward_densepose(features, instances))
        return instances, losses

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """

        instances = super().forward_with_given_boxes(features, instances)
        instances = self._forward_densepose(features, instances)
        return instances

class PartDensePoseChartWithPreditor(nn.Module):
    def __init__(self):
        super().__init__()
        map_dict = {
          # 15 -> 25
            0 : [0],
            1 : [1, 2],
            2 : [3],
            3 : [4],
            4 : [6],
            5 : [5],
            6 : [7, 9],
            7 : [8, 10],
            8 : [11, 13],
            9 : [12, 14],
           10 : [15, 17],
           11 : [16, 18],
           12 : [19, 21],
           13 : [20, 22],
           14 : [23, 24],
        }

        self.part2I = [
            nn.ConvTranspose2d(512, len(map_dict[key]), kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)) 
            for key in map_dict
        ]
        

    def forward(self, input):
        pass


'''
(ann_index_lowres): ConvTranspose2d(512, 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
(index_uv_lowres): ConvTranspose2d(512, 25, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
(u_lowres): ConvTranspose2d(512, 25, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
(v_lowres): ConvTranspose2d(512, 25, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
'''

class ConcatePartDensePoseHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
        )

    def forward(self, input):
        return self.model(input)
# ================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================
# ================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================
