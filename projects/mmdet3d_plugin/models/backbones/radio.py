import torch

from mmdet.models.builder import BACKBONES
from mmcv.runner.base_module import BaseModule

@BACKBONES.register_module()
class RADIO(BaseModule):
    def __init__(self,
                 model_version="radio_v2.5-b",
                 img_size=[480, 800],
                 freeze=None
                 ):
        super(RADIO, self).__init__()
        self.model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=model_version, progress=True, skip_validation=True)
        if "e-radio_v2" in model_version:
            self.model.model.set_optimal_window_size(img_size)
        
        if freeze is None:
            self.model.eval()
        
        #self._freeze_stages()
        
    def forward(self, x):
        x.div_(255.0)
        # import ipdb; ipdb.set_trace()
        summary, spatial_features = self.model(x, feature_fmt='NCHW')
        assert spatial_features.ndim == 4
        return spatial_features