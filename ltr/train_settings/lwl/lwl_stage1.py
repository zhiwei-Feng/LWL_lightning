def run(settings):
    settings.description = 'Default train settings with backbone weights fixed. We initialize the backbone ResNet with ' \
                           'pre-trained Mask-RCNN weights. These weights can be obtained from ' \
                           'https://drive.google.com/file/d/12pVHmhqtxaJ151dZrXN1dcgUa7TuAjdA/view?usp=sharing. ' \
                           'Download and save these weights in env_settings.pretrained_networks directory'
    settings.batch_size = 20
    settings.num_workers = 8
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

