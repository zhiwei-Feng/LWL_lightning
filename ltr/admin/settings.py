class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = ''
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks'
        self.davis_dir = ''
        self.youtubevos_dir = ''


class Settings:
    """ Training settings, e.g. the paths to datasets and networks."""

    def __init__(self):
        self.set_default()

    def set_default(self):
        self.env = EnvironmentSettings()
        self.use_gpu = True
