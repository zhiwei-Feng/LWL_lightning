import pytorch_lightning as pl

import ltr.data.transforms as tfm
from ltr.data import processing, sampler, LTRLoader
from ltr.dataset import YouTubeVOS, Davis


class LWLDataModule(pl.LightningDataModule):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        # Data transform
        self.transform_joint = tfm.Transform(tfm.ToBGR(),
                                             tfm.ToGrayscale(probability=0.05),
                                             tfm.RandomHorizontalFlip(probability=0.5))

        self.transform_train = tfm.Transform(tfm.RandomAffine(p_flip=0.0, max_rotation=15.0,
                                                              max_shear=0.0, max_ar_factor=0.0,
                                                              max_scale=0.2, pad_amount=0),
                                             tfm.ToTensorAndJitter(0.2, normalize=False),
                                             tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

        self.transform_val = tfm.Transform(tfm.ToTensorAndJitter(0.0, normalize=False),
                                           tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

        self.data_processing_train = processing.LWLProcessing(search_area_factor=settings.search_area_factor,
                                                              output_sz=settings.output_sz,
                                                              center_jitter_factor=settings.center_jitter_factor,
                                                              scale_jitter_factor=settings.scale_jitter_factor,
                                                              mode='sequence',
                                                              crop_type=settings.crop_type,
                                                              max_scale_change=settings.max_scale_change,
                                                              transform=self.transform_train,
                                                              joint_transform=self.transform_joint,
                                                              new_roll=True)

        self.data_processing_val = processing.LWLProcessing(search_area_factor=settings.search_area_factor,
                                                            output_sz=settings.output_sz,
                                                            center_jitter_factor=settings.center_jitter_factor,
                                                            scale_jitter_factor=settings.scale_jitter_factor,
                                                            mode='sequence',
                                                            crop_type=settings.crop_type,
                                                            max_scale_change=settings.max_scale_change,
                                                            transform=self.transform_val,
                                                            joint_transform=self.transform_joint,
                                                            new_roll=True)

    def setup(self, stage=None):
        # Datasets
        ytvos_train = YouTubeVOS(version="2018", multiobj=False, split='jjtrain')
        davis_train = Davis(version='2017', multiobj=False, split='train')

        ytvos_val = YouTubeVOS(version="2018", multiobj=False, split='jjvalid')
        # Train sampler and loader
        self.dataset_train = sampler.LWLSampler([ytvos_train, davis_train], [6, 1],
                                                samples_per_epoch=self.settings.batch_size * 1000, max_gap=100,
                                                num_test_frames=3,
                                                num_train_frames=1,
                                                processing=self.data_processing_train)
        self.dataset_val = sampler.LWLSampler([ytvos_val], [1],
                                              samples_per_epoch=self.settings.batch_size * 100, max_gap=100,
                                              num_test_frames=3,
                                              num_train_frames=1,
                                              processing=self.data_processing_val)

    def train_dataloader(self):
        loader_train = LTRLoader('train', self.dataset_train, training=True, num_workers=self.settings.num_workers,
                                 stack_dim=1, batch_size=self.settings.batch_size)
        return loader_train

    def val_dataloader(self):
        loader_val = LTRLoader('val', self.dataset_val, training=False, num_workers=self.settings.num_workers,
                               epoch_interval=5, stack_dim=1, batch_size=self.settings.batch_size)
        return loader_val
