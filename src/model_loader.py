import os
import torch
import torchvision
from ultralytics import YOLO



def build_model(nclasses: int = 2, mode: str = None, segment_model: str = None):
    """
    @param[in]  nclasses
    @param[in]  mode  set mode for frame classification or uninformative part mask
    """
    if mode == 'classify':
        #net of Resnet18
        net = torchvision.models.resnet18(num_classes = nclasses)
        net.cuda()
    if mode == 'mask':
        net = YOLO(segment_model)

    return net

def build_LemonFM(nclasses: int = 2, pretrained: bool = True, pretrained_weights = None):


    #net of ConvNext
    net = torchvision.models.convnext_large(weights='DEFAULT')
    input_emdim = net.classifier[2].in_features
    net.classifier[2] = nn.Identity()
    
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        state_dict = state_dict['teacher']

        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith('backbone.')}
        msg = net.load_state_dict(state_dict, strict=False)
        print(msg, input_emdim)

    net.cuda()

    return net


net = build_model(nclasses=num_classes, mode='classify')
model_path = 'Video storyboard classification models'

# Enable multi-GPU support
net = torch.nn.DataParallel(net)
torch.backends.cudnn.benchmark = True
state = torch.load(model_path, map_location=torch.device('cuda'))
net.load_state_dict(state['net'])
net.eval()
