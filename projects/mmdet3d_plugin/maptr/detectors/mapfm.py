import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from mmcv.runner import force_fp32, auto_fp16
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import builder
from mmcv.utils import TORCH_VERSION, digit_version
@DETECTORS.register_module()
class MapFM(MVXTwoStageDetector):
    """MapTR.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 modality='vision',
                 lidar_encoder=None,
                 concat_features_after_backbone=False,
                 reverse_feats=False
                 ):

        super(MapFM,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.concat_features_after_backbone = concat_features_after_backbone
        self.reverse_feats = reverse_feats

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }
        self.modality = modality
        if self.modality == 'fusion' and lidar_encoder is not None :
            if lidar_encoder["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**lidar_encoder["voxelize"])
            else:
                voxelize_module = DynamicScatter(**lidar_encoder["voxelize"])
            self.lidar_modal_extractor = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": builder.build_middle_encoder(lidar_encoder["backbone"]),
                }
            )
            self.voxelize_reduce = lidar_encoder.get("voxelize_reduce", True)


    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            # import ipdb; ipdb.set_trace()
            # print(f'img: {img.shape}')
            # print(f'img_feats: {img_feats[0].shape}')
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
            if len(img_feats) > 1 and self.concat_features_after_backbone:
                img_feats = [torch.cat(img_feats, axis=1)]
            if len(img_feats) > 1 and self.reverse_feats:
                img_feats = list(img_feats)
                img_feats.reverse()
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
            
        if len(img_feats) > 1 and self.reverse_feats:
            # import ipdb; ipdb.set_trace()
            img_feats = [img_feats[0]]

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats


    def forward_pts_train(self,
                          pts_feats,
                          lidar_feat,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None,
                          gt_depth=None,
                          gt_seg_mask=None,
                          gt_pv_seg_mask=None,
                          gt_segmap=None,
                          gt_lidar_bev_maps=None
                          ):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(
            pts_feats, lidar_feat, img_metas, prev_bev)

        depth = outs.pop('depth')
        losses = dict()
        # calculate depth loss
        if gt_depth is not None and depth is not None:
            loss_depth = self.pts_bbox_head.transformer.encoder.get_depth_loss(gt_depth, depth)
            if digit_version(TORCH_VERSION) >= digit_version('1.8'):
                loss_depth = torch.nan_to_num(loss_depth)
            losses.update(loss_depth=loss_depth)

        loss_inputs = [gt_bboxes_3d, gt_labels_3d, gt_seg_mask, gt_pv_seg_mask, outs, gt_segmap, gt_lidar_bev_maps]
        losses_pts = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        losses.update(losses_pts)
        # import ipdb;ipdb.set_trace()
        k_one2many = self.pts_bbox_head.k_one2many
        multi_gt_bboxes_3d = copy.deepcopy(gt_bboxes_3d)
        multi_gt_labels_3d = copy.deepcopy(gt_labels_3d)
        for i, (each_gt_bboxes_3d, each_gt_labels_3d) in enumerate(zip(multi_gt_bboxes_3d, multi_gt_labels_3d)):
            each_gt_bboxes_3d.instance_list = each_gt_bboxes_3d.instance_list * k_one2many
            each_gt_bboxes_3d.instance_labels = each_gt_bboxes_3d.instance_labels * k_one2many
            multi_gt_labels_3d[i] = each_gt_labels_3d.repeat(k_one2many)
        # import ipdb;ipdb.set_trace()
        one2many_outs = outs['one2many_outs']
        loss_one2many_inputs = [multi_gt_bboxes_3d, multi_gt_labels_3d, gt_seg_mask, gt_pv_seg_mask, one2many_outs, gt_segmap, gt_lidar_bev_maps]
        loss_dict_one2many = self.pts_bbox_head.loss(*loss_one2many_inputs, img_metas=img_metas)

        lambda_one2many = self.pts_bbox_head.lambda_one2many
        for key, value in loss_dict_one2many.items():
            if key + "_one2many" in losses.keys():
                losses[key + "_one2many"] += value * lambda_one2many
            else:
                losses[key + "_one2many"] = value * lambda_one2many
        # import ipdb;ipdb.set_trace()
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.lidar_modal_extractor["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes
    @auto_fp16(apply_to=('points'), out_fp32=True)
    def extract_lidar_feat(self,points):
        feats, coords, sizes = self.voxelize(points)
        # voxel_features = self.lidar_modal_extractor["voxel_encoder"](feats, sizes, coords)
        batch_size = coords[-1, 0] + 1
        lidar_feat = self.lidar_modal_extractor["backbone"](feats, coords, batch_size, sizes=sizes)
        
        return lidar_feat

    # @auto_fp16(apply_to=('img', 'points'))
    @force_fp32(apply_to=('img','points','prev_bev'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      gt_depth=None,
                      gt_seg_mask=None,
                      gt_pv_seg_mask=None,
                      gt_segmap=None,
                      gt_lidar_bev_maps=None
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        lidar_feat = None
        if self.modality == 'fusion':
            lidar_feat = self.extract_lidar_feat(points)
        
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        # prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        # import pdb;pdb.set_trace()
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas) if len_queue>1 else None

        img_metas = [each[len_queue-1] for each in img_metas]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, lidar_feat, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev, gt_depth,gt_seg_mask,gt_pv_seg_mask, gt_segmap, gt_lidar_bev_maps)

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None,points=None,  **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        points = [points] if points is None else points
        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], points[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return bbox_results

    def pred2result(self, bboxes, scores, labels, pts, attrs=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
            labels (torch.Tensor): Labels with shape of (n, ).
            scores (torch.Tensor): Scores with shape of (n, ).
            attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (torch.Tensor): 3D boxes.
                - scores (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """
        result_dict = dict(
            boxes_3d=bboxes.to('cpu'),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            pts_3d=pts.to('cpu'))

        if attrs is not None:
            result_dict['attrs_3d'] = attrs.cpu()

        return result_dict
    def simple_test_pts(self, x, lidar_feat, img_metas, prev_bev=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, lidar_feat, img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        
        bbox_results = [
            self.pred2result(bboxes, scores, labels, pts)
            for bboxes, scores, labels, pts in bbox_list
        ]
        # import pdb;pdb.set_trace()
        return outs['bev_embed'], bbox_results, outs
    def simple_test(self, img_metas, img=None, points=None, prev_bev=None, rescale=False, **kwargs):
        """Test function without augmentaiton."""
        lidar_feat = None
        if self.modality =='fusion':
            lidar_feat = self.extract_lidar_feat(points)
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts, outs = self.simple_test_pts(
            img_feats, lidar_feat, img_metas, prev_bev, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            
        ################################################################
        # import matplotlib.pyplot as plt
        # # print(img_metas)

        # # Конвертация в numpy
        # bev_seg = torch.sigmoid(outs['seg'][0]).cpu().numpy()

        # # Визуализация
        # plt.imshow(bev_seg[0])
        # plt.title('BEV Segmentation')
        # plt.axis('off')
        # plt.show()

        # batch_idx = 0
        # num_cams = outs['pv_seg'].shape[1]

        # fig, axes = plt.subplots(2, num_cams, figsize=(15, 5))  # 1 строка, num_cams столбцов

        # for cam in range(num_cams):
        #     img_cam = img[batch_idx, cam].cpu().numpy().transpose(1, 2, 0)  # Преобразуем в (H, W, C)
        #     # if img_cam.max() > 1.0:  # Если значения в [0, 255], нормализуем их
        #     #     img_cam = img_cam / 255.0
            
        #     # Если данные в BGR-формате, меняем порядок каналов
        #     img_cam = img_cam[:, :, ::-1]
        #     # Первая строка: показываем изображения камер
        #     ax = axes[0, cam]
        #     ax.imshow(img_cam)
        #     ax.set_title(f'Camera {cam}')
        #     ax.axis('off')
        #     # Извлекаем сегментацию для текущей камеры
        #     pv_seg = torch.sigmoid(outs['pv_seg'][batch_idx, cam]).cpu().numpy()
            
        #     # Отображаем на соответствующем подрисунке
        #     ax = axes[1, cam]
        #     ax.imshow(pv_seg[0], cmap='viridis')  # Используем 'tab10' или любую другую палитру
        #     ax.set_title(f'Camera {cam}')
        #     ax.axis('off')

        # # Настроим общий заголовок для всех камер
        # plt.suptitle('PV Segmentation for All Cameras', fontsize=16)
        # plt.tight_layout()
        # plt.subplots_adjust(top=0.85)  # Добавляем отступ сверху для заголовка
        # plt.show()
        ################################################################

        return new_prev_bev, bbox_list
