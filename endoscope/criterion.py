import numpy
import torch.nn
import torch.nn as nn
from utils import img2mse
from skimage.metrics import structural_similarity as ssim1

class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, ray_batch, scalars_to_log):
        '''
        training criterion
        '''

        # pred_rgb = outputs['rgb']
        # pred_mask = outputs['mask'].float()
        # gt_rgb = ray_batch['rgb']
        # loss = img2mse(pred_rgb, gt_rgb, pred_mask)

        pred_mask = outputs['mask'].float()
        # print('pred_mask:{}, pred_rgb:{}'.format(outputs['mask'].shape,outputs['rgb'].shape))
        pred_rgb = outputs['rgb']*pred_mask.unsqueeze(-1)
        # pred_mask = outputs['mask'].float()
        gt_rgb = ray_batch['rgb']*pred_mask.unsqueeze(-1)

        lossS = torch.nn.SmoothL1Loss()
        # loss1=lossS(pred_rgb, gt_rgb)
        loss = lossS(pred_rgb, gt_rgb)


        # loss2=ssim1(gt_rgb.cpu().detach().numpy(), pred_rgb.cpu().detach().numpy(), multichannel=True)
        # loss=0.5*loss1+0.5*loss2

        return loss, scalars_to_log

