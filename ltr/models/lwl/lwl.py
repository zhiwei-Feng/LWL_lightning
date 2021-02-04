from collections import OrderedDict

import torch

from pytracking.libs import TensorList


class LWTLNet(torch.nn.Module):
    def __init__(self, feature_extractor, target_model, decoder, target_model_input_layer, decoder_input_layers,
                 label_encoder=None):
        super().__init__()

        self.feature_extractor = feature_extractor  # Backbone feature extractor F
        self.target_model = target_model  # Target model and the few-shot learner
        self.decoder = decoder  # Segmentation Decoder

        self.label_encoder = label_encoder  # Few-shot label generator and weight predictor

        self.target_model_input_layer = (target_model_input_layer,) if isinstance(target_model_input_layer,
                                                                                  str) else target_model_input_layer
        self.decoder_input_layers = decoder_input_layers
        self.output_layers = sorted(list(set(self.target_model_input_layer + self.decoder_input_layers)))

    def forward(self, train_imgs, test_imgs, train_masks, test_masks, num_refinement_iter=2):
        num_sequences = train_imgs.shape[1]
        num_train_frames = train_imgs.shape[0]
        num_test_frames = test_imgs.shape[0]

        # Extract backbone features
        train_feat_backbone = self.extract_backbone_features(
            train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
        test_feat_backbone = self.extract_backbone_features(
            test_imgs.contiguous().view(-1, test_imgs.shape[-3], test_imgs.shape[-2], test_imgs.shape[-1]))

        # Extract features input to the target model
        train_feat_tm = self.extract_target_model_features(train_feat_backbone)  # seq*frames, channels, height, width
        train_feat_tm = train_feat_tm.view(num_train_frames, num_sequences, *train_feat_tm.shape[-3:])

        train_feat_tm_all = [train_feat_tm, ]

        # Get few-shot learner label and spatial importance weights
        # only use train_masks, no need train_feat_tm
        few_shot_label, few_shot_sw = self.label_encoder(train_masks)

        few_shot_label_all = [few_shot_label, ]
        few_shot_sw_all = None if few_shot_sw is None else [few_shot_sw, ]

        test_feat_tm = self.extract_target_model_features(test_feat_backbone)  # seq*frames, channels, height, width

        # Obtain the target module parameters using the few-shot learner
        filter, filter_iter, _ = self.target_model.get_filter(train_feat_tm, few_shot_label, few_shot_sw)

        mask_predictons_all = []

        # Iterate over the test sequence
        for i in range(num_test_frames):
            # Features for the current frame
            test_feat_tm_it = test_feat_tm.view(num_test_frames, num_sequences, *test_feat_tm.shape[-3:])[i:i + 1, ...]

            # Apply the target model to obtain mask encodings.
            mask_encoding_pred = [self.target_model.apply_target_model(f, test_feat_tm_it) for f in filter_iter]

            test_feat_backbone_it = {k: v.view(num_test_frames, num_sequences, *v.shape[-3:])[i, ...] for k, v in
                                     test_feat_backbone.items()}
            mask_encoding_pred_last_iter = mask_encoding_pred[-1]

            # Run decoder to obtain the segmentation mask
            mask_pred, decoder_feat = self.decoder(mask_encoding_pred_last_iter, test_feat_backbone_it,
                                                   test_imgs.shape[-2:])
            mask_pred = mask_pred.view(1, num_sequences, *mask_pred.shape[-2:])

            mask_predictons_all.append(mask_pred)

            # Convert the segmentation scores to probability
            mask_pred_prob = torch.sigmoid(mask_pred.clone().detach())

            # Obtain label encoding for the predicted mask in the previous frame
            # no need test_feat_tm_it
            few_shot_label, few_shot_sw = self.label_encoder(mask_pred_prob)

            # Extend the training data using the predicted mask
            few_shot_label_all.append(few_shot_label)
            if few_shot_sw_all is not None:
                few_shot_sw_all.append(few_shot_sw)

            train_feat_tm_all.append(test_feat_tm_it)

            # Update the target model using the extended training set
            if (i < (num_test_frames - 1)) and (num_refinement_iter > 0):
                train_feat_tm_it = torch.cat(train_feat_tm_all, dim=0)
                few_shot_label_it = torch.cat(few_shot_label_all, dim=0)

                if few_shot_sw_all is not None:
                    few_shot_sw_it = torch.cat(few_shot_sw_all, dim=0)
                else:
                    few_shot_sw_it = None

                # Run few-shot learner to update the target model
                filter_updated, _, _ = self.target_model.filter_optimizer(TensorList([filter]),
                                                                          feat=train_feat_tm_it,
                                                                          label=few_shot_label_it,
                                                                          sample_weight=few_shot_sw_it,
                                                                          num_iter=num_refinement_iter)

                filter = filter_updated[0]  # filter_updated is a TensorList

        mask_predictons_all = torch.cat(mask_predictons_all, dim=0)
        return mask_predictons_all

    def segment_target(self, target_filter, test_feat_tm, test_feat, obj_id: int = None, image=None, mask=None):
        # Classification features
        # focus: 做evaluation的时候用这个方法来产生target model的输出和最终的mask预测
        assert target_filter.dim() == 5  # seq, filters, ch, h, w
        test_feat_tm = test_feat_tm.view(1, 1, *test_feat_tm.shape[-3:])
        mask_encoding_pred = self.target_model.apply_target_model(target_filter, test_feat_tm)

        mask_pred, decoder_feat = self.decoder(mask_encoding_pred, test_feat,
                                               (test_feat_tm.shape[-2] * 16, test_feat_tm.shape[-1] * 16))

        return mask_pred, mask_encoding_pred

    def get_backbone_target_model_features(self, backbone_feat):
        # Get the backbone feature block which is input to the target model
        feat = OrderedDict({l: backbone_feat[l] for l in self.target_model_input_layer})
        if len(self.target_model_input_layer) == 1:
            return feat[self.target_model_input_layer[0]]
        return feat

    def extract_target_model_features(self, backbone_feat):
        return self.target_model.extract_target_model_features(self.get_backbone_target_model_features(backbone_feat))

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)
