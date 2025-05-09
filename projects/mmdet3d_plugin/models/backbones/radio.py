import torch

from mmdet.models.builder import BACKBONES
from mmcv.runner.base_module import BaseModule

@BACKBONES.register_module()
class RADIO(BaseModule):
    def __init__(self,
                 model_version="radio_v2.5-b",
                 img_size=[480, 800],
                 freeze=None,
                 chunk_size=3,
                 ):
        super(RADIO, self).__init__()
        self.chunk_size = chunk_size
        
        self.model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=model_version, progress=True, skip_validation=True)
        if "e-radio_v2" in model_version:
            self.model.model.set_optimal_window_size(img_size)
        
        if freeze is None:
            self.model.eval()
        
        #self._freeze_stages()
        
    def forward(self, x):
        x = x.div(255.0)  # Без in-place — безопаснее для автоград
        B = x.size(0)
        assert B % self.chunk_size == 0, "Количество входных изображений должно быть кратно 6"

        spatial_outputs = []

        for i in range(0, B, self.chunk_size):
            x_chunk = x[i:i + self.chunk_size]
            _, spatial_features = self.model(x_chunk, feature_fmt='NCHW')
            assert spatial_features.ndim == 4
            spatial_outputs.append(spatial_features)

        return [torch.cat(spatial_outputs, dim=0)]
