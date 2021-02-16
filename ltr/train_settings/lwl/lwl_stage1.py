import math
import os
from collections import OrderedDict

import torch
import torch.optim as optim
from ltr.data_module.train_dm import LWLDataModule
import pytorch_lightning as pl
import ltr.models.backbone as backbones
import ltr.models.backbone.resnet_mrcnn as mrcnn_backbones
import ltr.models.target_classifier.features as clf_features
import ltr.models.lwl.label_encoder as seg_label_encoder
import ltr.models.lwl.initializer as seg_initializer
import ltr.models.lwl.loss_residual_modules as loss_residual_modules
import ltr.models.meta.steepestdescent as steepestdescent
import ltr.models.lwl.linear_filter as target_clf
import ltr.models.lwl.decoder as lwtl_decoder
from ltr.models.loss.segmentation import LovaszSegLoss
from ltr.models.lwl.lwl import LWTLNet
from pytracking.libs import TensorList
from pytracking.analysis.vos_utils import davis_jaccard_measure


def run(settings):
    settings.description = 'Default train settings with backbone weights fixed. We initialize the backbone ResNet with ' \
                           'pre-trained Mask-RCNN weights. These weights can be obtained from ' \
                           'https://drive.google.com/file/d/12pVHmhqtxaJ151dZrXN1dcgUa7TuAjdA/view?usp=sharing. ' \
                           'Download and save these weights in env_settings.pretrained_networks directory'
    settings.batch_size = 10
    settings.num_workers = 4
    settings.multi_gpu = True
    settings.print_interval = 1
    settings.normalize_mean = [102.9801, 115.9465, 122.7717]
    settings.normalize_std = [1.0, 1.0, 1.0]

    settings.feature_sz = (52, 30)

    # Settings used for generating the image crop input to the network. See documentation of LWTLProcessing class in
    # ltr/data/processing.py for details.
    settings.output_sz = (settings.feature_sz[0] * 16, settings.feature_sz[1] * 16)  # Size of input image crop
    settings.search_area_factor = 5.0
    settings.crop_type = 'inside_major'
    settings.max_scale_change = None

    settings.center_jitter_factor = {'train': 3, 'test': (5.5, 4.5)}
    settings.scale_jitter_factor = {'train': 0.25, 'test': 0.5}

    # 准备好训练的dataloader
    lwl_dm = LWLDataModule(settings)
    model = LitLwlStage1(settings, filter_size=3, num_filters=16, optim_iter=5,
                         backbone_pretrained=True,
                         out_feature_dim=512,
                         frozen_backbone_layers=['conv1', 'bn1', 'layer1', 'layer2', 'layer3',
                                                 'layer4'],
                         label_encoder_dims=(16, 32, 64),
                         use_bn_in_label_enc=False,
                         clf_feat_blocks=0,
                         final_conv=True,
                         backbone_type='mrcnn')
    trainer = pl.Trainer(gpus=1, max_epochs=70, check_val_every_n_epoch=5, accumulate_grad_batches=2)
    trainer.fit(model, lwl_dm)


class LitLwlStage1(pl.LightningModule):
    def __init__(self, settings, filter_size=1, num_filters=1, optim_iter=3, optim_init_reg=0.01,
                 backbone_pretrained=False, clf_feat_blocks=1,
                 clf_feat_norm=True, final_conv=False,
                 out_feature_dim=512,
                 target_model_input_layer='layer3',
                 decoder_input_layers=("layer4", "layer3", "layer2", "layer1",),
                 detach_length=float('Inf'),
                 label_encoder_dims=(1, 1),
                 frozen_backbone_layers=(),
                 decoder_mdim=64, filter_groups=1,
                 use_bn_in_label_enc=True,
                 dilation_factors=None,
                 backbone_type='imagenet'):
        super().__init__()
        self.settings = settings
        ############## BUILD NET ###################
        # backbone feature extractor F
        if backbone_type == 'imagenet':
            backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)
        elif backbone_type == 'mrcnn':
            backbone_net = mrcnn_backbones.resnet50(pretrained=False, frozen_layers=frozen_backbone_layers)
        else:
            raise Exception

        norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))
        layer_channels = backbone_net.out_feature_channels()

        # Extracts features input to the target model
        target_model_feature_extractor = clf_features.residual_basic_block(
            feature_dim=layer_channels[target_model_input_layer],
            num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
            final_conv=final_conv, norm_scale=norm_scale,
            out_dim=out_feature_dim)

        # Few-shot label generator and weight predictor
        label_encoder = seg_label_encoder.ResidualDS16SW(layer_dims=label_encoder_dims + (num_filters,),
                                                         use_bn=use_bn_in_label_enc)

        # Predicts initial target model parameters
        initializer = seg_initializer.FilterInitializerZero(filter_size=filter_size, num_filters=num_filters,
                                                            feature_dim=out_feature_dim, filter_groups=filter_groups)

        # Computes few-shot learning loss
        residual_module = loss_residual_modules.LWTLResidual(init_filter_reg=optim_init_reg,
                                                             filter_dilation_factors=dilation_factors)

        # Iteratively updates the target model parameters by minimizing the few-shot learning loss
        optimizer = steepestdescent.GNSteepestDescent(residual_module=residual_module, num_iter=optim_iter,
                                                      detach_length=detach_length,
                                                      residual_batch_dim=1, compute_losses=True)

        # Target model and Few-shot learner
        target_model = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                               filter_optimizer=optimizer,
                                               feature_extractor=target_model_feature_extractor,
                                               filter_dilation_factors=dilation_factors)

        # Decoder
        decoder_input_layers_channels = {L: layer_channels[L] for L in decoder_input_layers}

        decoder = lwtl_decoder.LWTLDecoder(num_filters, decoder_mdim, decoder_input_layers_channels, use_bn=True)

        # build lwl model
        self.net = LWTLNet(feature_extractor=backbone_net, target_model=target_model, decoder=decoder,
                           label_encoder=label_encoder,
                           target_model_input_layer=target_model_input_layer, decoder_input_layers=decoder_input_layers)
        ############## BUILD NET ###################
        # Load pre-trained maskrcnn weights
        self._load_pretrained_weights()

        # Loss function
        self.objective = {
            'segm': LovaszSegLoss(per_image=False),
        }
        self.loss_weight = {
            'segm': 100.0
        }

        # Optimizer
        self.optimizer = optim.Adam([{'params': self.net.target_model.filter_initializer.parameters(), 'lr': 5e-5},
                                     {'params': self.net.target_model.filter_optimizer.parameters(), 'lr': 1e-4},
                                     {'params': self.net.target_model.feature_extractor.parameters(), 'lr': 2e-5},
                                     {'params': self.net.decoder.parameters(), 'lr': 1e-4},
                                     {'params': self.net.label_encoder.parameters(), 'lr': 2e-4}],
                                    lr=2e-4)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[40, ], gamma=0.2)

        # actor初始化
        self.num_refinement_iter = 2
        self.disable_backbone_bn = False
        self.disable_all_bn = True
        self._update_settings(settings)

    def _train_actor(self, mode=True):
        """ Set whether the network is in train mode.
                args:
                    mode (True) - Bool specifying whether in training mode.
                """
        self.net.train(mode)

        if self.disable_all_bn:
            self.net.eval()
        elif self.disable_backbone_bn:
            for m in self.net.feature_extractor.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()

    def _load_pretrained_weights(self):
        weights_path = os.path.join(self.settings.env.pretrained_networks, 'e2e_mask_rcnn_R_50_FPN_1x_converted.pkl')
        pretrained_weights = torch.load(weights_path)
        self.net.feature_extractor.load_state_dict(pretrained_weights)

    def _update_settings(self, settings=None):
        """Updates the trainer settings. Must be called to update internal settings."""
        if settings is not None:
            self.settings = settings

        if self.settings.env.workspace_dir is not None:
            self.settings.env.workspace_dir = os.path.expanduser(self.settings.env.workspace_dir)
            self._checkpoint_dir = os.path.join(self.settings.env.workspace_dir, 'checkpoints')
            if not os.path.exists(self._checkpoint_dir):
                os.makedirs(self._checkpoint_dir)
        else:
            self._checkpoint_dir = None

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        :param batch:
        :param batch_idx:
        :return:
        """
        data = batch
        # data['epoch'] = self.epoch
        data['settings'] = self.settings
        # loss, stats = self.actor(data)
        segm_pred = self.net(train_imgs=data['train_images'],
                             test_imgs=data['test_images'],
                             train_masks=data['train_masks'],
                             test_masks=data['test_masks'],
                             num_refinement_iter=self.num_refinement_iter)

        acc = 0
        cnt = 0

        segm_pred = segm_pred.view(-1, 1, *segm_pred.shape[-2:])
        gt_segm = data['test_masks']
        gt_segm = gt_segm.view(-1, 1, *gt_segm.shape[-2:])

        loss_segm = self.loss_weight['segm'] * self.objective['segm'](segm_pred, gt_segm)

        acc_l = [davis_jaccard_measure(torch.sigmoid(rm.detach()).cpu().numpy() > 0.5, lb.cpu().numpy()) for
                 rm, lb in zip(segm_pred.view(-1, *segm_pred.shape[-2:]), gt_segm.view(-1, *segm_pred.shape[-2:]))]
        acc += sum(acc_l)
        cnt += len(acc_l)

        loss = loss_segm

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        self.log("train_acc", acc / cnt, on_step=True, prog_bar=True)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch
        segm_pred = self.net(train_imgs=data['train_images'],
                             test_imgs=data['test_images'],
                             train_masks=data['train_masks'],
                             test_masks=data['test_masks'],
                             num_refinement_iter=self.num_refinement_iter)
        acc = 0
        cnt = 0
        segm_pred = segm_pred.view(-1, 1, *segm_pred.shape[-2:])
        gt_segm = data['test_masks']
        gt_segm = gt_segm.view(-1, 1, *gt_segm.shape[-2:])

        acc_l = [davis_jaccard_measure(torch.sigmoid(rm.detach()).cpu().numpy() > 0.5, lb.cpu().numpy()) for
                 rm, lb in zip(segm_pred.view(-1, *segm_pred.shape[-2:]), gt_segm.view(-1, *segm_pred.shape[-2:]))]
        acc += sum(acc_l)
        cnt += len(acc_l)

        self.log('val_acc', acc / cnt, on_step=True, prog_bar=True)

    def configure_optimizers(self):
        return [self.optimizer], [self.lr_scheduler]

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
