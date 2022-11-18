# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.config import configurable
# from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
# from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
import logging
from typing import Dict, Tuple, List, Optional
from collections import OrderedDict
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone, Backbone
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.utils.events import get_event_storage
from detectron2.structures import ImageList

############### Image discriminator ##############
class FCDiscriminator_img(nn.Module):
    def __init__(self, num_classes, ndf1=256, ndf2=128):
        super(FCDiscriminator_img, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ndf1, ndf2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(ndf2, ndf2, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(ndf2, 1, kernel_size=3, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x
#################################

################ Gradient reverse function
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

#######################

@META_ARCH_REGISTRY.register()
class DAobjTwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        dis_type: str,
        # dis_loss_weight: float = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super(GeneralizedRCNN, self).__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        # @yujheli: you may need to build your discriminator here

        self.dis_type = dis_type
        # self.D_img = None
        self.D_img_ins = None
        self.D_img_2 = None
        self.D_img_3 = None
        self.D_img_4 = None
        # self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels['res4']) # Need to know the channel

        # self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels[self.dis_type]) # Need to know the channel
        self.D_img_ins = FCDiscriminator_img(512)
        self.D_img_2 = FCDiscriminator_img(self.backbone._out_feature_channels["vgg2"])
        self.D_img_3 = FCDiscriminator_img(self.backbone._out_feature_channels["vgg3"])
        self.D_img_4 = FCDiscriminator_img(self.backbone._out_feature_channels["vgg4"])
        # self.bceLoss_func = nn.BCEWithLogitsLoss()
    def build_discriminator(self):
        # self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels[self.dis_type]).to(self.device) # Need to know the channel
        self.D_img_ins = FCDiscriminator_img(512)
        self.D_img_2 = FCDiscriminator_img(self.backbone._out_feature_channels["vgg2"]).to(self.device)
        self.D_img_3 = FCDiscriminator_img(self.backbone._out_feature_channels["vgg3"]).to(self.device)
        self.D_img_4 = FCDiscriminator_img(self.backbone._out_feature_channels["vgg4"]).to(self.device)


    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "dis_type": cfg.SEMISUPNET.DIS_TYPE,
            # "dis_loss_ratio": cfg.xxx,
        }

    def preprocess_image_train(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        images_t = [x["image_unlabeled"].to(self.device) for x in batched_inputs]
        images_t = [(x - self.pixel_mean) / self.pixel_std for x in images_t]
        images_t = ImageList.from_tensors(images_t, self.backbone.size_divisibility)

        return images, images_t
    
    def focal_loss(self, x, domain_label, gamma = 1.5):
        BCE = F.binary_cross_entropy_with_logits(x,torch.FloatTensor(x.data.size()).fill_(
                                                                    domain_label).to(self.device))
        p = torch.exp(-BCE)
        
        FLloss = (1-p) ** gamma * BCE# torch.log(domain_label + (-1) ** domain_label * F.sigmoid(x))
        return FLloss
        
    
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if self.D_img_2 == None: # self.D_img_2 == None
            self.build_discriminator()
        if (not self.training) and (not val_mode):  # only conduct when testing mode
            return self.inference(batched_inputs)

        source_label = 0
        target_label = 1

        if branch == "domain":
            # self.D_img.train()
            # source_label = 0
            # target_label = 1
            # images = self.preprocess_image(batched_inputs)
            
            images_s, images_t = self.preprocess_image_train(batched_inputs)

            features = self.backbone(images_s.tensor)
            
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None
            # get the roi box_features
            # 1.Region proposal network
            proposals_rpn_s, _ = self.proposal_generator(
                images_s, features, gt_instances
            )

            # 2.roi_head lower branch
            _, _, box_features_s = self.roi_heads(
                images_s,
                features,
                proposals_rpn_s,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )
            # import pdb
            # pdb.set_trace()

            # features_s = grad_reverse(features[self.dis_type])
            # D_img_out_s = self.D_img(features_s)
            # loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))
            
            box_features_s = grad_reverse(box_features_s)
            features_s_2 = grad_reverse(features["vgg2"])
            features_s_3 = grad_reverse(features["vgg3"])
            features_s_4 = grad_reverse(features["vgg4"])

            D_img_out_s_ins = self.D_img_ins(box_features_s)
            D_img_out_s_2 = self.D_img_2(features_s_2)
            loss_D_img_s_2 = F.binary_cross_entropy_with_logits(D_img_out_s_2,
                                                                torch.FloatTensor(D_img_out_s_2.data.size()).fill_(
                                                                    source_label).to(self.device))
            D_img_out_s_3 = self.D_img_3(features_s_3)
            loss_D_img_s_3 = F.binary_cross_entropy_with_logits(D_img_out_s_3,
                                                                torch.FloatTensor(D_img_out_s_3.data.size()).fill_(
                                                                    source_label).to(self.device))
            D_img_out_s_4 = self.D_img_4(features_s_4)
            # loss_D_img_s_4 = F.binary_cross_entropy_with_logits(D_img_out_s_4,
            #                                                     torch.FloatTensor(D_img_out_s_4.data.size()).fill_(
            #                                                         source_label).to(self.device))
            loss_D_img_s_4 = self.focal_loss(D_img_out_s_4, source_label)
            loss_D_img_s_ins = F.binary_cross_entropy_with_logits(D_img_out_s_ins, torch.FloatTensor(D_img_out_s_ins.data.size()).fill_(source_label).to(self.device))
            # loss_D_img_s_ins = self.focal_loss(D_img_out_s_ins, source_label)
            loss_D_img_s = loss_D_img_s_2 + loss_D_img_s_3 + loss_D_img_s_4#


            features_t = self.backbone(images_t.tensor)
            # get the roi box_features
            # 1.Region proposal network
            proposals_rpn_t, _ = self.proposal_generator(
                images_t, features_t, gt_instances
            )

            # 2.roi_head lower branch
            _, _, box_features_t = self.roi_heads(
                images_t,
                features,
                proposals_rpn_t,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            # features_t = grad_reverse(features_t[self.dis_type])
            # D_img_out_t = self.D_img(features_t)
            # loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).to(self.device))
           
            features_t_2 = grad_reverse(features_t["vgg2"])
            features_t_3 = grad_reverse(features_t["vgg3"])
            features_t_4 = grad_reverse(features_t["vgg4"])

            D_img_out_t_2 = self.D_img_2(features_t_2)
            loss_D_img_t_2 = F.binary_cross_entropy_with_logits(D_img_out_t_2,
                                                                torch.FloatTensor(D_img_out_t_2.data.size()).fill_(
                                                                    target_label).to(self.device))
            D_img_out_t_3 = self.D_img_3(features_t_3)
            loss_D_img_t_3 = F.binary_cross_entropy_with_logits(D_img_out_t_3,
                                                                torch.FloatTensor(D_img_out_t_3.data.size()).fill_(
                                                                    target_label).to(self.device))
            D_img_out_t_4 = self.D_img_4(features_t_4)
            # loss_D_img_t_4 = F.binary_cross_entropy_with_logits(D_img_out_t_4,
            #                                                     torch.FloatTensor(D_img_out_t_4.data.size()).fill_(
            #                                                         target_label).to(self.device))
            loss_D_img_t_4 = self.focal_loss(D_img_out_t_4, target_label)
            # instance level Discriminator
            box_features_t = grad_reverse(box_features_t) #TODO 忘了改box_feature_s！！他妈的！
            D_img_out_t_ins = self.D_img_ins(box_features_t)
            loss_D_img_t_ins = F.binary_cross_entropy_with_logits(D_img_out_t_ins, torch.FloatTensor(D_img_out_t_ins.data.size()).fill_(target_label).to(self.device))
            # loss_D_img_t_ins = self.focal_loss(D_img_out_t_ins, target_label)
            loss_D_img_t = loss_D_img_t_2 + loss_D_img_t_3 + loss_D_img_t_4 #

            # import pdb
            # pdb.set_trace()

            losses = {}
            losses["loss_D_img_s"] = loss_D_img_s + loss_D_img_s_ins
            losses["loss_D_img_t"] = loss_D_img_t + loss_D_img_t_ins
            # return losses, [], [], None
            return losses

        
        # self.D_img.eval()
        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)


        # TODO: remove the usage of if else here. This needs to be re-organized
        if branch.startswith("supervised"):
            
            # features_s = grad_reverse(features[self.dis_type])
            # D_img_out_s = self.D_img(features_s)
            # loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))

            features_s_2 = grad_reverse(features["vgg2"])
            features_s_3 = grad_reverse(features["vgg3"])
            features_s_4 = grad_reverse(features["vgg4"])
    
            D_img_out_s_2 = self.D_img_2(features_s_2)
            loss_D_img_s_2 = F.binary_cross_entropy_with_logits(D_img_out_s_2,
                                                              torch.FloatTensor(D_img_out_s_2.data.size()).fill_(
                                                                  source_label).to(self.device))
            D_img_out_s_3 = self.D_img_3(features_s_3)
            loss_D_img_s_3 = F.binary_cross_entropy_with_logits(D_img_out_s_3,
                                                                torch.FloatTensor(D_img_out_s_3.data.size()).fill_(
                                                                    source_label).to(self.device))
            D_img_out_s_4 = self.D_img_4(features_s_4)
            # loss_D_img_s_4 = F.binary_cross_entropy_with_logits(D_img_out_s_4,
            #                                                     torch.FloatTensor(D_img_out_s_4.data.size()).fill_(
            #                                                         source_label).to(self.device))
            loss_D_img_s_4 = self.focal_loss(D_img_out_s_4, source_label)
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses, box_features = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )
            box_features = grad_reverse(box_features)
            D_img_out_ins = self.D_img_ins(box_features)
            loss_D_img_ins = F.binary_cross_entropy_with_logits(D_img_out_ins, torch.FloatTensor(D_img_out_ins.data.size()).fill_(source_label).to(self.device))
            # loss_D_img_ins = self.focal_loss(D_img_out_ins, source_label)
            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            # losses["loss_D_img_s"] = (loss_D_img_s +loss_D_img_ins)*0.001
            losses["loss_D_img_s"] = (loss_D_img_s_2 + loss_D_img_s_3 + loss_D_img_s_4 + loss_D_img_ins) * 0.001# 
            # return losses, [], [], None  
            return losses

        elif branch.startswith("supervised_target"):

            # features_t = grad_reverse(features_t[self.dis_type])
            # D_img_out_t = self.D_img(features_t)
            # loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).to(self.device))


            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses,_ = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            # losses["loss_D_img_t"] = loss_D_img_t*0.001
            # losses["loss_D_img_s"] = loss_D_img_s*0.001
            # return losses, [], [], None
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            """
            unsupervised weak branch: input image without any ground-truth label; output proposals of rpn and roi-head
            """
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)
            # notice that we do not use any target in ROI head to do inference!
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            # if self.vis_period > 0:
            #     storage = get_event_storage()
            #     if storage.iter % self.vis_period == 0:
            #         self.visualize_training(batched_inputs, proposals_rpn, branch)

            # return {}, proposals_rpn, proposals_roih, ROI_predictions
            return proposals_rpn, proposals_roih, ROI_predictions
        elif branch == "unsup_data_strong":
            raise NotImplementedError()
        elif branch == "val_loss":
            raise NotImplementedError()

    def visualize_training(self, batched_inputs, proposals, branch=""):
        """
        This function different from the original one:
        - it adds "branch" to the `vis_name`.

        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = (
                "Left: GT bounding boxes "
                + branch
                + ";  Right: Predicted proposals "
                + branch
            )
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch



@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses,_ = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses,_ = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None


