# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict, List
import torch
from torch import nn

from detectron2.config import CfgNode
from detectron2.layers import ConvTranspose2d, interpolate

from ...structures import DensePoseChartPredictorOutput
from ..utils import initialize_module_params
from .registry import DENSEPOSE_PREDICTOR_REGISTRY


@DENSEPOSE_PREDICTOR_REGISTRY.register()
class DensePoseChartPredictor(nn.Module):
    """
    Predictor (last layers of a DensePose model) that takes DensePose head outputs as an input
    and produces 4 tensors which represent DensePose results for predefined body parts
    (patches / charts):
     * coarse segmentation, a tensor of shape [N, K, Hout, Wout]
     * fine segmentation, a tensor of shape [N, C, Hout, Wout]
     * U coordinates, a tensor of shape [N, C, Hout, Wout]
     * V coordinates, a tensor of shape [N, C, Hout, Wout]
    where
     - N is the number of instances
     - K is the number of coarse segmentation channels (
         2 = foreground / background,
         15 = one of 14 body parts / background)
     - C is the number of fine segmentation channels (
         24 fine body parts / background)
     - Hout and Wout are height and width of predictions
    """

    def __init__(self, cfg: CfgNode, input_channels: int):
        """
        Initialize predictor using configuration options

        Args:
            cfg (CfgNode): configuration options
            input_channels (int): input tensor size along the channel dimension
        """
        super().__init__()
        dim_in = input_channels
        n_segm_chan = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS
        dim_out_patches = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES + 1
        kernel_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECONV_KERNEL
        # coarse segmentation
        self.ann_index_lowres = ConvTranspose2d(
            dim_in, n_segm_chan, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        # fine segmentation
        self.index_uv_lowres = ConvTranspose2d(
            dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        # U
        self.u_lowres = ConvTranspose2d(
            dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        # V
        self.v_lowres = ConvTranspose2d(
            dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        self.scale_factor = cfg.MODEL.ROI_DENSEPOSE_HEAD.UP_SCALE
        initialize_module_params(self)

    def interp2d(self, tensor_nchw: torch.Tensor):
        """
        Bilinear interpolation method to be used for upscaling

        Args:
            tensor_nchw (tensor): tensor of shape (N, C, H, W)
        Return:
            tensor of shape (N, C, Hout, Wout), where Hout and Wout are computed
                by applying the scale factor to H and W
        """
        return interpolate(
            tensor_nchw, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )

    def forward(self, head_outputs: torch.Tensor):
        """
        Perform forward step on DensePose head outputs

        Args:
            head_outputs (tensor): DensePose head outputs, tensor of shape [N, D, H, W]
        Return:
           An instance of DensePoseChartPredictorOutput
        """
        return DensePoseChartPredictorOutput(
            coarse_segm=self.interp2d(self.ann_index_lowres(head_outputs)),
            fine_segm=self.interp2d(self.index_uv_lowres(head_outputs)),
            u=self.interp2d(self.u_lowres(head_outputs)),
            v=self.interp2d(self.v_lowres(head_outputs)),
        )





#================================================================================================
#================================================================================================

@DENSEPOSE_PREDICTOR_REGISTRY.register()
class MyDensePoseChartPredictor(nn.Module):
    """
    Predictor (last layers of a DensePose model) that takes DensePose head outputs as an input
    and produces 4 tensors which represent DensePose results for predefined body parts
    (patches / charts):
     * coarse segmentation, a tensor of shape [N, K, Hout, Wout]
     * fine segmentation, a tensor of shape [N, C, Hout, Wout]
     * U coordinates, a tensor of shape [N, C, Hout, Wout]
     * V coordinates, a tensor of shape [N, C, Hout, Wout]
    where
     - N is the number of instances
     - K is the number of coarse segmentation channels (
         2 = foreground / background,
         15 = one of 14 body parts / background)
     - C is the number of fine segmentation channels (
         24 fine body parts / background)
     - Hout and Wout are height and width of predictions
    """

    def __init__(self, cfg: CfgNode, input_channels: int):
        super().__init__()
        dim_in = input_channels
        n_segm_chan = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS
        dim_out_patches = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES + 1
        kernel_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECONV_KERNEL
        # # coarse segmentation
        # self.ann_index_lowres = ConvTranspose2d(
        #     dim_in, n_segm_chan, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        # )
        # # fine segmentation
        # self.index_uv_lowres = ConvTranspose2d(
        #     dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        # )
        # # U
        # self.u_lowres = ConvTranspose2d(
        #     dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        # )
        # # V
        # self.v_lowres = ConvTranspose2d(
        #     dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        # )
        
        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------
        self.map_dict = {
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
        
        
        self.part_I = [
            setattr(
                self, 
                'part_I'+str(key), 
                nn.ConvTranspose2d(
                    dim_in, len(self.map_dict[key]), kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
                ) 
            )
            for key in self.map_dict
        ]

        self.part_U = [
            setattr(
                self, 
                'part_U'+str(key), 
                nn.ConvTranspose2d(
                    dim_in, len(self.map_dict[key]), kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
                ) 
            )
            for key in self.map_dict
        ]

        self.part_V = [
            setattr(
                self, 
                'part_V'+str(key), 
                nn.ConvTranspose2d(
                    dim_in, len(self.map_dict[key]), kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
                ) 
            )
            for key in self.map_dict
        ]

        # model.to(torch.device(cfg.MODEL.DEVICE))
        # self.part_I.to(torch.device('cuda'))
        # self.part_U.to(torch.device('cuda'))
        # self.part_V.to(torch.device('cuda'))
        # self.putModelToCuda(self.part_I)
        # self.putModelToCuda(self.part_U)
        # self.putModelToCuda(self.part_V)
        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------

        self.scale_factor = cfg.MODEL.ROI_DENSEPOSE_HEAD.UP_SCALE
        initialize_module_params(self)

    def putModelToCuda(self, model_list: List[torch.Tensor]):
        for model in model_list:
            model.to(torch.device('cuda'))

    def interp2d(self, tensor_nchw: torch.Tensor):
        return interpolate(
            tensor_nchw, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )

    def part2Dense(self, input_tensor: torch.Tensor, model_name: str) -> torch.Tensor:
        assert input_tensor.shape[1:] == (512, 28, 28)

        head_outputs = input_tensor.reshape(-1, 15, 512, 28, 28)    # (K, 15, 512, 28, 28)
        K = head_outputs.shape[0]
        head_outputs = head_outputs.permute(1, 0, 2, 3, 4)          # (15, K, 512, 28, 28)
        head_outputs = head_outputs.to(input_tensor.device)
        answer = [0 for i in range(25)]
        for i in range(15):
            temp_head_output = head_outputs[i]               # (K, 512, 28, 28)
            temp_tensor = getattr(self, model_name+str(i))(temp_head_output)    # (K, 1 or 2, out_dim, out_dim)
            for j in range(len(self.map_dict[i])):
                answer[self.map_dict[i][j]] = temp_tensor[:, j, :, :].unsqueeze(1)   #(K, 1, out_dim, out_dim)

        answer = torch.cat(answer, axis=1)  # (K, 25, out_dim, out_dim)
        assert answer.shape[:2] == (K, 25)
        return answer.to(input_tensor.device)


    def forward(self, head_outputs: torch.Tensor, coarse_segm: torch.Tensor):

        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------
        part2I = self.part2Dense(head_outputs, "part_I")
        part2U = self.part2Dense(head_outputs, "part_U")
        part2V = self.part2Dense(head_outputs, "part_V")

        return DensePoseChartPredictorOutput(
            coarse_segm=coarse_segm,
            fine_segm=self.interp2d(part2I),
            u=self.interp2d(part2U),
            v=self.interp2d(part2V),
        )
        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------

        # return DensePoseChartPredictorOutput(
        #     coarse_segm=self.interp2d(self.ann_index_lowres(head_outputs)),
        #     fine_segm=,
        #     u=self.interp2d(self.u_lowres(head_outputs)),
        #     v=self.interp2d(self.v_lowres(head_outputs)),
        # )
        
        return DensePoseChartPredictorOutput(
            coarse_segm=self.interp2d(self.ann_index_lowres(head_outputs)),
            fine_segm=self.interp2d(self.index_uv_lowres(head_outputs)),
            u=self.interp2d(self.u_lowres(head_outputs)),
            v=self.interp2d(self.v_lowres(head_outputs)),
        )

#================================================================================================
#================================================================================================