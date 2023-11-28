import torch
from collections import OrderedDict

########################################################################################################################
# helper functions for nerf ray rendering
########################################################################################################################

def sample_pdf(bins, weights, N_samples, det=False):
    ''':param bins: tensor of shape [N_rays, M+1], M is the number of bins
    :param weights: tensor of shape [N_rays, M]
    :param N_samples: number of samples along each ray
    :param det: if True, will perform deterministic sampling  如果为True，将执行确定性抽样
    :return: [N_rays, N_samples]
    '''

    M = weights.shape[1]  #62
    weights += 1e-5   #（426,62）
    # Get pdf
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)    #(426,62) [N_rays, M]  归一化，得到概率密度
    cdf = torch.cumsum(pdf, dim=-1)  #(426,62)  [N_rays, M]  当前采样点的累积概率=射线上当前采样点的概率和前面累积的概率相加
    cdf = torch.cat([torch.zeros_like(cdf[:, 0:1]), cdf], dim=-1) #(426,63) [N_rays, M+1]

    # Take uniform samples  执行均匀采样
    if det:  #确定性采样？？？  不执行
        u = torch.linspace(0., 1., N_samples, device=bins.device)
        u = u.unsqueeze(0).repeat(bins.shape[0], 1)       # [N_rays, N_samples]
    else:   #均匀采样
        u = torch.rand(bins.shape[0], N_samples, device=bins.device)  #(426,64)   为什么随机采样

    # Invert CDF
    above_inds = torch.zeros_like(u, dtype=torch.long)   #(426,64) [N_rays, N_samples]
    for i in range(M):
        above_inds += (u >= cdf[:, i:i+1]).long()  #（426,64）

    # random sample inside each bin  每一个bin内部随机采样
    below_inds = torch.clamp(above_inds-1, min=0)  #(426,64)  小于0的赋值0  比above的值都小1，除了小于1的元素
    inds_g = torch.stack((below_inds, above_inds), dim=2)     #(426,64,2)  [N_rays, N_samples, 2]

    cdf = cdf.unsqueeze(1).repeat(1, N_samples, 1)  # (426,64,63) [N_rays, N_samples, M+1]
    cdf_g = torch.gather(input=cdf, dim=-1, index=inds_g)  # [N_rays, N_samples, 2]

    bins = bins.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, M+1]
    bins_g = torch.gather(input=bins, dim=-1, index=inds_g)  # [N_rays, N_samples, 2]

    # t = (u-cdf_g[:, :, 0]) / (cdf_g[:, :, 1] - cdf_g[:, :, 0] + TINY_NUMBER)  # [N_rays, N_samples]
    # fix numeric issue  解决数字问题?
    denom = cdf_g[:, :, 1] - cdf_g[:, :, 0]      # [N_rays, N_samples]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[:, :, 0]) / denom

    samples = bins_g[:, :, 0] + t * (bins_g[:, :, 1]-bins_g[:, :, 0])

    return samples  #(426,64)

def sample_along_camera_ray(ray_o, ray_d, depth_range, N_samples, inv_uniform=False, det=False):
    ''':param ray_o: origin of the ray in scene coordinate system; tensor of shape [N_rays, 3]
    :param ray_d: homogeneous ray direction vectors in scene coordinate system; tensor of shape [N_rays, 3]
    :param depth_range: [near_depth, far_depth]
    :param inv_uniform: if True, uniformly sampling inverse depth
    :param det: if True, will perform deterministic sampling
    :return: tensor of shape [N_rays, N_samples, 3]
    '''
    # will sample inside [near_depth, far_depth]
    # assume the nearest possible depth is at least (min_ratio * depth)
    near_depth_value = depth_range[0, 0]
    far_depth_value = depth_range[0, 1]
    assert near_depth_value > 0 and far_depth_value > 0 and far_depth_value > near_depth_value

    near_depth = near_depth_value * torch.ones_like(ray_d[..., 0])

    far_depth = far_depth_value * torch.ones_like(ray_d[..., 0])
    if inv_uniform:
        start = 1. / near_depth     # [N_rays,]
        step = (1. / far_depth - start) / (N_samples-1)
        inv_z_vals = torch.stack([start+i*step for i in range(N_samples)], dim=1)  # [N_rays, N_samples]
        z_vals = 1. / inv_z_vals
    else:
        start = near_depth
        step = (far_depth - near_depth) / (N_samples-1)
        z_vals = torch.stack([start+i*step for i in range(N_samples)], dim=1)  # [N_rays, N_samples]

    if not det:
        # get intervals between samples
        mids = .5 * (z_vals[:, 1:] + z_vals[:, :-1])
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, 0:1], mids], dim=-1)
        # uniform samples in those intervals
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand   # [N_rays, N_samples]

    ray_d = ray_d.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, 3]
    ray_o = ray_o.unsqueeze(1).repeat(1, N_samples, 1)
    pts = z_vals.unsqueeze(2) * ray_d + ray_o       # [N_rays, N_samples, 3]
    return pts, z_vals

########################################################################################################################
# ray rendering of nerf
########################################################################################################################

def raw2outputs(raw, z_vals, mask, white_bkgd=False):
    ''' :param raw: raw network output; tensor of shape [N_rays, N_samples, 4]
    :param z_vals: depth of point samples along rays; tensor of shape [N_rays, N_samples]
    :param ray_d: [N_rays, 3]
    :return: {'rgb': [N_rays, 3], 'depth': [N_rays,], 'weights': [N_rays,], 'depth_std': [N_rays,]}
    '''
    rgb = raw[:, :, :3]     # [N_rays, N_samples, 3]
    sigma = raw[:, :, 3]    # [N_rays, N_samples]

    # note: we did not use the intervals here, because in practice different scenes from COLMAP can have 在这里没有使用间隔，因为来自Clomap的不同场景中有不同的缩放
    # very different scales, and using interval can affect the model's generalization ability.  使用间隔会影响模型的泛化能力
    # Therefore we don't use the intervals for both training and evaluation. 没有将间隔时间用于训练和评估
    sigma2alpha = lambda sigma, dists: 1. - torch.exp(-sigma)

    # point samples are ordered with increasing depth interval between samples 点样本的排序随样本间深度间隔的增加而增加
    #
    dists = z_vals[:, 1:] - z_vals[:, :-1]  #(465,63) 临近采样点的深度间隔
    dists = torch.cat((dists, dists[:, -1:]), dim=-1)  ##(465,64)  [N_rays, N_samples]

    alpha = sigma2alpha(sigma, dists)  #(465,64) [N_rays, N_samples]  alpha表示每个采样点的透明度

    # Eq. (3): T
    T = torch.cumprod(1. - alpha + 1e-10, dim=-1)[:, :-1]   #(465,63) [N_rays, N_samples-1]  累乘不透明度
    T = torch.cat((torch.ones_like(T[:, 0:1]), T), dim=-1)  #(465,64) [N_rays, N_samples]  在0位置添加一个不透明度为1(即最近的采样点透明度为0)

    # maths show weights, and summation of weights along a ray, are always inside [0, 1]  数学显示重量  沿一条射线的权值求和，总是在[0,1]之
    weights = alpha * T     #(465,64)  [N_rays, N_samples]  当前点的权重=当前点的透明度*当前点的不透明度(当前点之前的强度累加)
    rgb_map = torch.sum(weights.unsqueeze(2) * rgb, dim=1)  #(465,3)  [N_rays, 3]  最终渲染图像像素点的rgb=射线上的权重的累加

    if white_bkgd:  #不执行
        rgb_map = rgb_map + (1. - torch.sum(weights, dim=-1, keepdim=True))

    mask = mask.float().sum(dim=1) > 8 #(465,64)变为(465,)  should at least have 8 valid observation on the ray, otherwise don't consider its loss 至少应该有8条有效的射线观测，否则不考虑其损失
    depth_map = torch.sum(weights * z_vals, dim=-1)     #(465,) [N_rays,] 最终渲染深度图上的一个像素点的深度值=射线上采样点的权重*射线上采样点的深度

    ret = OrderedDict([('rgb', rgb_map), ('depth', depth_map), ('weights', weights), ('mask', mask), ('alpha', alpha), ('z_vals', z_vals)])  #**** used for importance sampling of fine samples 用于精细样品的重要性采样
    return ret

def render_rays(ray_batch, model, featmaps, projector, N_samples, inv_uniform=False, N_importance=0, det=False, white_bkgd=False):
    '''
    :param ray_batch: {'ray_o': [N_rays, 3] , 'ray_d': [N_rays, 3], 'view_dir': [N_rays, 2]}
    :param model:  {'net_coarse':  , 'net_fine': }
    :param N_samples: samples along each ray (for both coarse and fine model)
    :param inv_uniform: if True, uniformly sample inverse depth for coarse model
    :param N_importance: additional samples along each ray produced by importance sampling (for fine model)
    :param det: if True, will deterministicly sample depths
    :return: {'outputs_coarse': {}, 'outputs_fine': {}}
    '''

    ret = {'outputs_coarse': None, 'outputs_fine': None}

    # pts: [N_rays, N_samples, 3]  获得采样点的位置和深度
    # z_vals: [N_rays, N_samples]
    pts, z_vals = sample_along_camera_ray(ray_o=ray_batch['ray_o'], ray_d=ray_batch['ray_d'], depth_range=ray_batch['depth_range'], N_samples=N_samples, inv_uniform=inv_uniform, det=det)
    N_rays, N_samples = pts.shape[:2]
    #rgb_feat:(465,64,11,35); ray_diff:(465,64,11,4); mask:(465,64,11,1)
    rgb_feat, ray_diff, mask = projector.compute(pts, ray_batch['camera'], ray_batch['src_rgbs'], ray_batch['src_cameras'], featmaps=featmaps[0])  # [N_rays, N_samples, N_views, x]  #rgb_feat采样点对应的参考图像的rgb和rgb特征的拼接(426,64,12,35)，ray_diff目标射线和源视图射线偏差 (426,64,12,4)，mask：(426,64,12,1)  进入projection.py函数
    pixel_mask = mask[..., 0].sum(dim=2) > 1   # (465,64)  [N_rays, N_samples], should at least have 2 observations  采样点至少在两个源视图中可见，才赋值该采样点的mask为1
    raw_coarse = model.net_coarse(rgb_feat, ray_diff, mask)   #(465,64,4)  经过整个网络后获得的采样点的颜色值和密度值的拼接******* [N_rays, N_samples, 4]
    outputs_coarse = raw2outputs(raw_coarse, z_vals, pixel_mask, white_bkgd=white_bkgd)  #使用粗网络渲染后的颜色，及深度depth、权重、mask、密度、z_vals
    ret['outputs_coarse'] = outputs_coarse

    if N_importance > 0:  #N_importance=64 进行精细模型渲染
        assert model.net_fine is not None
        # detach since we would like to decouple the coarse and fine networks  分离，因为我们想要分离粗网络和细网络
        weights = outputs_coarse['weights'].clone().detach() #(465,64)             # [N_rays, N_samples]
        if inv_uniform:  #不执行
            inv_z_vals = 1. / z_vals
            inv_z_vals_mid = .5 * (inv_z_vals[:, 1:] + inv_z_vals[:, :-1])   # [N_rays, N_samples-1]
            weights = weights[:, 1:-1]      # [N_rays, N_samples-2]
            inv_z_vals = sample_pdf(bins=torch.flip(inv_z_vals_mid, dims=[1]), weights=torch.flip(weights, dims=[1]), N_samples=N_importance, det=det)  # [N_rays, N_importance]
            z_samples = 1. / inv_z_vals
        else:   #开始细采样
            # take mid-points of depth samples  取深度样本的中点  (每两个临近粗采样点的中点)
            z_vals_mid = .5 * (z_vals[:, 1:] + z_vals[:, :-1])   #(465,63) [N_rays, N_samples-1]
            weights = weights[:, 1:-1]      #（465,62） [N_rays, N_samples-2]
            z_samples = sample_pdf(bins=z_vals_mid, weights=weights, N_samples=N_importance, det=det)  #(465,64)   *****   [N_rays, N_importance]

        z_vals = torch.cat((z_vals, z_samples), dim=-1)  #（465,128）将粗采样的标准深度(从近到远)和细采样深度(经过概率密度函数获得的无序深度)拼接 [N_rays, N_samples + N_importance]

        # samples are sorted with increasing depth  采样点随深度的增加而排序
        z_vals, _ = torch.sort(z_vals, dim=-1)  #（465,128） 对所有的采样点进行重新排序
        N_total_samples = N_samples + N_importance  #128

        viewdirs = ray_batch['ray_d'].unsqueeze(1).repeat(1, N_total_samples, 1)
        ray_o = ray_batch['ray_o'].unsqueeze(1).repeat(1, N_total_samples, 1)
        pts = z_vals.unsqueeze(2) * viewdirs + ray_o  #(465,128,3)  混合采样点的位置*******   [N_rays, N_samples + N_importance, 3]
        # rgb_feat_sampled:(465,128,11,35); ray_diff:(465,128,11,4); mask:(465,128,11,1)
        rgb_feat_sampled, ray_diff, mask = projector.compute(pts, ray_batch['camera'], ray_batch['src_rgbs'], ray_batch['src_cameras'], featmaps=featmaps[1])

        pixel_mask = mask[..., 0].sum(dim=2) > 1  # [N_rays, N_samples]. should at least have 2 observations
        raw_fine = model.net_fine(rgb_feat_sampled, ray_diff, mask)   #精细网络输出*****
        outputs_fine = raw2outputs(raw_fine, z_vals, pixel_mask, white_bkgd=white_bkgd)  #精细训练出的颜色和密度，经过rendering渲染出的结果
        ret['outputs_fine'] = outputs_fine

    return ret
