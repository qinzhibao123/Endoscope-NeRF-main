import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        # self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))   #(640,4,128,128)     temperature=2,q:(640,4,128,4),k:(640,4,128,4)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9) #(640,4,128,128)
            # attn = attn * mask

        attn = F.softmax(attn, dim=-1)   #(640,4,128,128)
        # attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)  #(640,4,128,4)

        return output, attn     #(640,4,128,4)  (640,4,128,128)

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        # x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head  #4
        self.d_k = d_k  #4
        self.d_v = d_v  #4

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)  #linear(16,16,bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        # self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head  # 4,4,4
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)  #640,128, 128, 128

        residual = q  #(640,128,16)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)  #(640,128,4,4)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)  #(640,128,4,4)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)  #(640,128,4,4)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  #(640,4,128,4)

        if mask is not None:
            mask = mask.unsqueeze(1)   #(640,1,128,4) For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)  #(640,4,128,4) to (640,128,16)
        # q = self.dropout(self.fc(q))
        q = self.fc(q)  #(640,128,16)
        q += residual   #(640,128,16)

        q = self.layer_norm(q)  #(640,128,16)

        return q, attn  #(640,128,16)

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

# default tensorflow initialization of linear layers
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)

@torch.jit.script
def fused_mean_variance(x, weight):
    mean = torch.sum(x*weight, dim=2, keepdim=True)
    var = torch.sum(weight * (x - mean)**2, dim=2, keepdim=True)
    return mean, var

class Endoscope_NeRF(nn.Module):
    def __init__(self, args, in_feat_ch=32, n_samples=64, **kwargs):
        super(Endoscope_NeRF, self).__init__()
        self.args = args
        self.anti_alias_pooling = args.anti_alias_pooling
        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        activation_func = nn.ELU(inplace=True)
        self.n_samples = n_samples
        self.ray_dir_fc = nn.Sequential(nn.Linear(4, 16), activation_func, nn.Linear(16, in_feat_ch + 3), activation_func)  #4改为7
        self.base_fc = nn.Sequential(nn.Linear((in_feat_ch+3)*3, 64), activation_func, nn.Linear(64, 32), activation_func)
        self.vis_fc = nn.Sequential(nn.Linear(32, 32), activation_func, nn.Linear(32, 33), activation_func,)
        self.vis_fc2 = nn.Sequential(nn.Linear(32, 32), activation_func, nn.Linear(32, 1), nn.Sigmoid())
        self.geometry_fc = nn.Sequential(nn.Linear(32*2+1, 64), activation_func, nn.Linear(64, 16),activation_func)
        self.ray_attention = MultiHeadAttention(4, 16, 4, 4)
        self.out_geometry_fc = nn.Sequential(nn.Linear(16, 16), activation_func, nn.Linear(16, 1), nn.ReLU())
        self.rgb_fc = nn.Sequential(nn.Linear(32+1+4, 16), activation_func, nn.Linear(16, 8), activation_func, nn.Linear(8, 1))  #32+1+4
        self.pos_encoding = self.posenc(d_hid=16, n_samples=self.n_samples)

        self.base_fc.apply(weights_init)
        self.vis_fc2.apply(weights_init)
        self.vis_fc.apply(weights_init)
        self.geometry_fc.apply(weights_init)
        self.rgb_fc.apply(weights_init)

    def posenc(self, d_hid, n_samples):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table).to("cuda:{}".format(self.args.local_rank)).float().unsqueeze(0)
        return sinusoid_table

    def forward(self, rgb_feat, ray_diff, mask):
        ''':param rgb_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat]
        :param ray_diff: ray direction difference [n_rays, n_samples, n_views, 4], first 3 channels are directions,
        last channel is inner product
        :param mask: mask for whether each projection is valid or not. [n_rays, n_samples, n_views, 1]
        :return: rgb and density output, [n_rays, n_samples, 4]
        ''' #rgb_feat：(465,64,11,35)

        num_views = rgb_feat.shape[2]  #11   rgb_feat：(465,64,11,35)
        direction_feat = self.ray_dir_fc(ray_diff[...,:4])
        rgb_in = rgb_feat[..., :3]  #(465,64,11,3)
        rgb_feat = rgb_feat + direction_feat  #(465,64,11,35)
        if self.anti_alias_pooling:  # anti_alias_pooling=1
            _, dot_prod, dd = torch.split(ray_diff, [3, 1, 1], dim=-1)
            exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1)) #(465,64,11,1)
            dd_prod = torch.exp(torch.abs(self.s) * (dd - 1)) #qin
            weight = (exp_dot_prod - torch.min(exp_dot_prod, dim=2, keepdim=True)[0]) * mask  #(465,64,11,1)
            weight = weight / (torch.sum(weight, dim=2, keepdim=True) + 1e-8)  #(465,64,11,1)
            weight1 = (dd_prod - torch.min(dd_prod, dim=2, keepdim=True)[0]) * mask
            weight1 = weight1 / (torch.sum(weight1, dim=2, keepdim=True) + 1e-8)
        else:
            weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)
            weight1=weight  #q

        # compute mean and variance across different views for each point
        mean, var = fused_mean_variance(rgb_feat, weight)  #(465,64,1,35) [n_rays, n_samples, 1, n_feat]
        globalfeat = torch.cat([mean, var], dim=-1)  #(465,64,1,70) [n_rays, n_samples, 1, 2*n_feat]

        x = torch.cat([globalfeat.expand(-1, -1, num_views, -1), rgb_feat], dim=-1)  #(465,64,11,105) [n_rays, n_samples, n_views, 3*n_feat]
        x = self.base_fc(x)  #(465,64,11,32)

        x_vis = self.vis_fc(x * weight1)  #(465,64,11,33)
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1]-1, 1], dim=-1)
        vis = F.sigmoid(vis) * mask  #(465,64,11,1)
        x = x + x_res  #(465,64,11,32)
        vis = self.vis_fc2(x * vis) * mask   #(465,64,11,1)
        weight1 = vis / (torch.sum(vis, dim=2, keepdim=True) + 1e-8) #(465,64,11,1)

        mean, var = fused_mean_variance(x, weight1)  #(465,64,1,32)  (465,64,1,32)
        globalfeat = torch.cat([mean.squeeze(2), var.squeeze(2), weight1.mean(dim=2)], dim=-1)  #(465,64,65) [n_rays, n_samples, 32*2+1]
        globalfeat = self.geometry_fc(globalfeat)  # (465,64,16) [n_rays, n_samples, 16]
        num_valid_obs = torch.sum(mask, dim=2)  #(465,64,1)
        globalfeat = globalfeat + self.pos_encoding  #(465,64,16)        pos_encoding:(1,64,16)
        globalfeat, _ = self.ray_attention(globalfeat, globalfeat, globalfeat, mask=(num_valid_obs > 1).float())
        sigma = self.out_geometry_fc(globalfeat)  #(465,64,1)
        sigma_out = sigma.masked_fill(num_valid_obs < 1, 0.)  #(465,64,1) set the sigma of invalid point to zero

        # rgb computation  rgb计算************************
        x = torch.cat([x, vis, ray_diff[...,:4]], dim=-1)  #(465,64,11,37)
        x = self.rgb_fc(x)  #(465,64,11,1)
        x = x.masked_fill(mask == 0, -1e9) #(465,64,11,1)
        blending_weights_valid = F.softmax(x, dim=2)  #(465,64,11,1)  color blending
        rgb_out = torch.sum(rgb_in*blending_weights_valid, dim=2)  #(465,64,3)
        out = torch.cat([rgb_out, sigma_out], dim=-1)  #(465,64,4)
        return out

