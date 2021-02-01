import os
import ltr.admin.loading as ltr_loading
from pytracking.evaluation.environment import env_settings


def load_network(net_path, **kwargs):
    """ focus: Load network for tracking.
    args:
        net_path - Path to network. If it is not an absolute path, it is relative to the network_path in the local.py.
                   See ltr.admin.loading.load_network for further details.
        **kwargs - Additional key-word arguments that are sent to ltr.admin.loading.load_network.
    """
    kwargs['backbone_pretrained'] = False
    if os.path.isabs(net_path):
        # 如果给的是绝对路径，那么直接加载
        path_full = net_path
        net, _ = ltr_loading.load_network(path_full, **kwargs)
    elif isinstance(env_settings().network_path, (list, tuple)):
        net = None
        for p in env_settings().network_path:
            path_full = os.path.join(p, net_path)
            try:
                net, _ = ltr_loading.load_network(path_full, **kwargs)
                break
            except Exception as e:
                # print(e)
                pass

        assert net is not None, 'Failed to load network'
    else:
        # 只是string的话，使用env_settings()的network_path作为根目录
        path_full = os.path.join(env_settings().network_path, net_path)
        net, _ = ltr_loading.load_network(path_full, **kwargs)

    return net
