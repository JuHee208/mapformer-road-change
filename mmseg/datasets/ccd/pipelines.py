import os.path as osp
from statistics import mean

import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC

from mmseg.datasets.pipelines.formating import DefaultFormatBundle
from mmseg.datasets.pipelines.transforms import PhotoMetricDistortion

from ..pipelines import LoadImagesFromFile, LoadAnnotations, to_tensor
from ..builder import PIPELINES

IMAGENET_MEAN = np.array([123.675, 116.28,103.53]), 
IMAGENET_STD = np.array([58.395,57.12,57.375])
DYNEARTHNET_MEAN = np.array([649.9822,  862.5364,  939.1118])
DYNEARTHNET_STD = np.array([654.9196,  727.9036,  872.8431])

@PIPELINES.register_module()
class LoadMultipleImages(LoadImagesFromFile):
    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2',
                 to_imgnet_scale=True,
                 mean=None,
                 std=None,
                 rgb_only=True):
        super(LoadMultipleImages, self).__init__(to_float32, color_type, file_client_args, imdecode_backend)
        self.to_imgnet_scale = to_imgnet_scale
        if self.to_imgnet_scale:
            assert mean is not None and std is not None
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.rgb_only = rgb_only

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        # image pre
        img1_bytes = self.file_client.get(results['img_info']['filename_pre'])
        img1 = mmcv.imfrombytes(
            img1_bytes, flag=self.color_type, backend=self.imdecode_backend)

        # image post

        img2_bytes = self.file_client.get(results['img_info']['filename'])
        img2 = mmcv.imfrombytes(
            img2_bytes, flag=self.color_type, backend=self.imdecode_backend)

        if self.rgb_only:
            img1 = img1[:,:,:3]
            img2 = img2[:,:,:3]

        if self.to_imgnet_scale:
            img1 = (img1 - self.mean) / self.std * IMAGENET_STD + IMAGENET_MEAN
            img2 = (img2 - self.mean) / self.std * IMAGENET_STD + IMAGENET_MEAN
            img1 = img1.clip(0, 255)
            img2 = img2.clip(0, 255)

        if self.to_float32:
            img1 = img1.astype(np.float32)
            img2 = img2.astype(np.float32)
        else:
            img1 = img1.astype(np.uint8)
            img2 = img2.astype(np.uint8)

        results['filename1'] = results['img_info']['filename_pre']
        results['filename2'] = results['img_info']['filename']
        results['ori_filename'] = results['img_info']['filename']

        results['img'] = np.concatenate((img1, img2), axis=-1)
        results['img_shape'] = img1.shape
        results['ori_shape'] = img1.shape

        # Set initial values for default meta_keys
        results['pad_shape'] = img1.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img1.shape) < 3 else img1.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)

        return results

@PIPELINES.register_module()
class LoadMultipleAnnotations(LoadAnnotations):

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename_post = osp.join(results['seg_prefix'],
                                     results['ann_info']['seg_map'])
            filename_pre  = osp.join(results['seg_prefix'],
                                     results['ann_info']['seg_map_pre'])
            
        else:
            filename_post = results['ann_info']['seg_map']
            filename_pre  = results['ann_info']['seg_map_pre']
        img_bytes_post = self.file_client.get(filename_post)
        img_bytes_pre  = self.file_client.get(filename_pre)
        gt_semantic_seg_post = mmcv.imfrombytes(
            img_bytes_post, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        gt_semantic_seg_pre = mmcv.imfrombytes(
            img_bytes_pre, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        if gt_semantic_seg_post.ndim == 3:
            # go from one-hot encoding to index encoding
            assert gt_semantic_seg_post.ndim == 3
            #assert (gt_semantic_seg_post.sum(axis=2) > 0).mean() > 0.99
            #assert (gt_semantic_seg_pre.sum(axis=2) > 0).mean() > 0.99

            gt_semantic_seg_post = gt_semantic_seg_post.argmax(axis=2)
            gt_semantic_seg_pre = gt_semantic_seg_pre.argmax(axis=2)
        # modify if custom classes
        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg_post[gt_semantic_seg_post == old_id] = new_id
                gt_semantic_seg_pre[gt_semantic_seg_pre == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg_post[gt_semantic_seg_post == 0] = 255
            gt_semantic_seg_post = gt_semantic_seg_post - 1
            gt_semantic_seg_post[gt_semantic_seg_post == 254] = 255
            gt_semantic_seg_pre[gt_semantic_seg_pre == 0] = 255
            gt_semantic_seg_pre = gt_semantic_seg_pre - 1
            gt_semantic_seg_pre[gt_semantic_seg_pre == 254] = 255
        if self.map_255_to_1:
            gt_semantic_seg_post[gt_semantic_seg_post != 0] = 1
            gt_semantic_seg_pre[gt_semantic_seg_pre != 0] = 1
        results['gt_semantic_seg_post'] = gt_semantic_seg_post
        results['gt_semantic_seg_pre'] = gt_semantic_seg_pre
        results['seg_fields'].extend(['gt_semantic_seg_post', 'gt_semantic_seg_pre'])
        return results

@PIPELINES.register_module()
class LoadCCDAnnotations(LoadAnnotations):

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename_post = osp.join(results['seg_prefix'],
                                     results['ann_info']['seg_map_post'])
            filename_pre  = osp.join(results['seg_prefix'],
                                     results['ann_info']['seg_map_pre'])
            filename_bc   = osp.join(results['seg_prefix'],
                                     results['ann_info']['seg_map'])

            
        else:
            filename_post = results['ann_info']['seg_map_post']
            filename_pre  = results['ann_info']['seg_map_pre']
            filename_bc  = results['ann_info']['seg_map']
        img_bytes_post = self.file_client.get(filename_post)
        img_bytes_pre  = self.file_client.get(filename_pre)
        img_bytes_bc  = self.file_client.get(filename_bc)
        gt_semantic_seg_post = mmcv.imfrombytes(
            img_bytes_post, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        gt_semantic_seg_pre = mmcv.imfrombytes(
            img_bytes_pre, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        gt_semantic_seg_bc = mmcv.imfrombytes(
            img_bytes_bc, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        if gt_semantic_seg_post.ndim == 3:
            # go from one-hot encoding to index encoding
            assert gt_semantic_seg_post.ndim == 3
            gt_semantic_seg_post = gt_semantic_seg_post.argmax(axis=2)
            gt_semantic_seg_pre = gt_semantic_seg_pre.argmax(axis=2)
            gt_semantic_seg_bc = gt_semantic_seg_bc.argmax(axis=2)

        # modify if custom classes
        if results.get('label_map', None) is not None:
            # for old_id, new_id in results['label_map'].items():
            #     gt_semantic_seg_post[gt_semantic_seg_post == old_id] = new_id
            #     gt_semantic_seg_pre[gt_semantic_seg_pre == old_id] = new_id
            raise NotImplementedError
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg_post[gt_semantic_seg_post == 0] = 255
            gt_semantic_seg_post = gt_semantic_seg_post - 1
            gt_semantic_seg_post[gt_semantic_seg_post == 254] = 255
            gt_semantic_seg_pre[gt_semantic_seg_pre == 0] = 255
            gt_semantic_seg_pre = gt_semantic_seg_pre - 1
            gt_semantic_seg_pre[gt_semantic_seg_pre == 254] = 255
        if self.map_255_to_1:
            gt_semantic_seg_post[gt_semantic_seg_post != 0] = 1
            gt_semantic_seg_pre[gt_semantic_seg_pre != 0] = 1
        results['gt_semantic_seg_post'] = gt_semantic_seg_post
        results['gt_semantic_seg_pre'] = gt_semantic_seg_pre
        results['gt_semantic_seg'] = gt_semantic_seg_bc
        results['seg_fields'].extend(['gt_semantic_seg_post', 'gt_semantic_seg_pre', 'gt_semantic_seg'])
        return results

@PIPELINES.register_module()
class CreateBinaryChangeMask:
    def __init__(self, ignore_index):
        self.ignore_index = ignore_index

    def __call__(self, results):
        assert 'gt_semantic_seg_pre' in results and 'gt_semantic_seg_post' in results
        gt_pre = results['gt_semantic_seg_pre']
        gt_post = results['gt_semantic_seg_post']

        gt_bc = (gt_pre != gt_post).astype(gt_post.dtype)

        # get rid of ignore class
        gt_bc[gt_pre == self.ignore_index] = self.ignore_index
        gt_bc[gt_post == self.ignore_index] = self.ignore_index

        results['gt_semantic_seg'] = gt_bc
        results['seg_fields'].append('gt_semantic_seg')
        return results


@PIPELINES.register_module()
class RandomDiscreteRotate:
    """Rotate image/labels by one of discrete angles (e.g., 90/180/270)."""

    def __init__(
        self,
        prob=0.5,
        angles=(90, 180, 270),
        pad_val=0,
        seg_pad_val=255,
        center=None,
        auto_bound=False,
    ):
        self.prob = prob
        self.angles = tuple(angles)
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.center = center
        self.auto_bound = auto_bound
        assert 0.0 <= self.prob <= 1.0, "prob must be in [0,1]"
        assert len(self.angles) > 0, "angles must be non-empty"

    def __call__(self, results):
        if np.random.rand() >= self.prob:
            return results
        angle = float(np.random.choice(self.angles))
        results['img'] = mmcv.imrotate(
            results['img'],
            angle=angle,
            border_value=self.pad_val,
            center=self.center,
            auto_bound=self.auto_bound,
        )
        for key in results.get('seg_fields', []):
            results[key] = mmcv.imrotate(
                results[key],
                angle=angle,
                border_value=self.seg_pad_val,
                center=self.center,
                auto_bound=self.auto_bound,
                interpolation='nearest',
            )
        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(prob={self.prob}, "
            f"angles={self.angles}, pad_val={self.pad_val}, "
            f"seg_pad_val={self.seg_pad_val})"
        )


@PIPELINES.register_module()
class ChangeAwareAugment:
    """Apply selected augmentations only when change classes are present."""

    def __init__(
        self,
        change_classes=(1, 2),
        ignore_index=255,
        flip_prob=0.5,
        flip_directions=("horizontal",),
        rotate_prob=0.5,
        rotate_angles=(90, 180, 270),
        photo_prob=1.0,
        rotate_pad_val=0,
        rotate_seg_pad_val=255,
    ):
        self.change_classes = tuple(change_classes)
        self.ignore_index = ignore_index
        self.flip_prob = flip_prob
        self.flip_directions = tuple(flip_directions)
        self.rotate_prob = rotate_prob
        self.rotate_angles = tuple(rotate_angles)
        self.photo_prob = photo_prob
        self.rotate_pad_val = rotate_pad_val
        self.rotate_seg_pad_val = rotate_seg_pad_val
        self.photometric = PhotoMetricDistortion()

    def _has_change(self, gt):
        valid = gt != self.ignore_index
        if not np.any(valid):
            return False
        return np.isin(gt[valid], self.change_classes).any()

    def _flip(self, results):
        if self.flip_prob <= 0 or np.random.rand() >= self.flip_prob:
            return False, None
        direction = np.random.choice(self.flip_directions)
        results['img'] = mmcv.imflip(results['img'], direction=direction)
        for key in results.get('seg_fields', []):
            results[key] = mmcv.imflip(results[key], direction=direction)
        return True, direction

    def _rotate(self, results):
        if self.rotate_prob <= 0 or np.random.rand() >= self.rotate_prob:
            return
        angle = float(np.random.choice(self.rotate_angles))
        results['img'] = mmcv.imrotate(
            results['img'],
            angle=angle,
            border_value=self.rotate_pad_val,
            interpolation='bilinear',
        )
        for key in results.get('seg_fields', []):
            results[key] = mmcv.imrotate(
                results[key],
                angle=angle,
                border_value=self.rotate_seg_pad_val,
                interpolation='nearest',
            )

    def _photometric(self, results):
        if self.photo_prob <= 0 or np.random.rand() >= self.photo_prob:
            return results
        return self.photometric(results)

    def __call__(self, results):
        # Keep meta keys consistent with Collect defaults.
        results.setdefault('flip', False)
        results.setdefault('flip_direction', None)
        gt = results.get('gt_semantic_seg', None)
        if gt is None or not self._has_change(gt):
            return results
        did_flip, direction = self._flip(results)
        if did_flip:
            results['flip'] = True
            results['flip_direction'] = direction
        self._rotate(results)
        results = self._photometric(results)
        return results

@PIPELINES.register_module()
class CustomFormatBundle(DefaultFormatBundle):
    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        if 'gt_semantic_seg' in results:
            # convert to long
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None,
                                                     ...].astype(np.int64)),
                stack=True)
        if 'gt_semantic_seg_pre' in results:
            # convert to long
            results['gt_semantic_seg_pre'] = DC(
                to_tensor(results['gt_semantic_seg_pre'][None,
                                                     ...].astype(np.int64)),
                stack=True)
        if 'gt_semantic_seg_post' in results:
            # convert to long
            results['gt_semantic_seg_post'] = DC(
                to_tensor(results['gt_semantic_seg_post'][None,
                                                     ...].astype(np.int64)),
                stack=True)
        return results
