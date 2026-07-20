import os, sys
import numpy as np
from sklearn.decomposition import PCA # PCA: 主成分分析，用于特征降维（如图像向量降维可视化或聚类前处理）
from PIL import Image
from torchvision import transforms as T # PyTorch 图像变换库（如 Resize, Normalize, ToTensor），用于数据预处理
from math import sqrt
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker # StableDiffusionSafetyChecker: Hugging Face diffusers 中用于检测不安全生成图像的组件。
from transformers import AutoFeatureExtractor # 加载 transformer 模型的通用图像特征提取器（比如用于 safety checker）
from PIL import Image
import torch
from einops import rearrange # einops: 用于简洁表达张量重排（如 B C H W ↔ B H W C）
from omegaconf import OmegaConf
import json
from tqdm import tqdm
import glob
from torch import einsum # PyTorch 中爱因斯坦求和符号：用于高效矩阵操作（例如 attention 中 QK 计算）
from sklearn.cluster import KMeans,SpectralClustering # KMeans, SpectralClustering: 两种常用聚类算法，用于图像分割、区域提取等
from skimage.measure import label # 图像区域分析函数：label() 可对二值图像进行连通区域标记
import cv2
# sys.path.append('img2img')
from img2img.run_features_extraction import load_model_from_config

import matplotlib.pyplot as plt
"""
使用主成分分析PCA对特征图数据进行降维，将每个样本的前3个主成分可视化并保存为图像

Args:
    feature_maps_fit_data: 训练数据的特征图，形状为 (N, C, H, W)
    feature_maps_transform_data: 变换数据的特征图，形状为 (B, C, H, W)
    transform_experiments: 变换实验名称列表
    t: 当前时间步
    save_dir: 保存目录
"""
def visualize_and_save_features_pca(feature_maps_fit_data,feature_maps_transform_data, transform_experiments, t, save_dir):
    feature_maps_fit_data = feature_maps_fit_data.cpu().numpy() # 将feature_maps_fit_data从gpu转移到cpu，并转化为Numpy数组
    pca = PCA(n_components=3) 
    pca.fit(feature_maps_fit_data) # 初始化PCA对象，指定降维为3维，并将feature_maps_fit_data拟合PCA模型
    feature_maps_pca = pca.transform(feature_maps_transform_data.cpu().numpy())  # N X 3
    feature_maps_pca = feature_maps_pca.reshape(len(transform_experiments), -1, 3)  # B x (H * W) x 3 将 PCA 结果按实验数（transform_experiments）重新形状，使每个实验对应的特征图有 (H * W) 行，每行有 3 个 PCA 特征（RGB）
    for i, experiment in enumerate(transform_experiments): # 遍历每个实验，将其对应的PCA特征图pca_img提取出来
        pca_img = feature_maps_pca[i]  # (H * W) x 3
        h = w = int(sqrt(pca_img.shape[0])) # 计算重塑之后的图像尺寸(假设PCA特征图维正方形)
        pca_img = pca_img.reshape(h, w, 3)
        pca_img_min = pca_img.min(axis=(0, 1))
        pca_img_max = pca_img.max(axis=(0, 1))
        pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min) # 计算PCA图像的最大值和最小值，并进行归一化处理
        """
        将归一化后的图像转成PIL图像格式，并转换为0-255范围的像素值，
        将图像缩放到512x512大小，保持最近邻插值方式(避免插值影响图像细节)
        将图像保存到指定目录，文件名包含实验名和时间步
        """
        pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
        pca_img = T.Resize(512, interpolation=T.InterpolationMode.NEAREST)(pca_img)
        pca_img.save(os.path.join(save_dir, f"{experiment}_time_{t}.png"))

"""
与visualize_and_save_features_pca()函数类似，但不保存图像，而是将PCA结果保存为Tensor文件
Args:
    feature_maps_fit_data: 训练数据的特征图，形状为 (N, C, H, W)
    feature_maps_transform_data: 变换数据的特征图，形状为 (B, C, H, W)
    save_path: 保存路径
"""
def visualize_and_save_features_pca_custom(feature_maps_fit_data,feature_maps_transform_data, save_path):
    feature_maps_fit_data = feature_maps_fit_data.cpu().numpy()
    pca = PCA(n_components=3)
    pca.fit(feature_maps_fit_data)
    feature_maps_pca = pca.transform(feature_maps_transform_data.cpu().numpy())  # N X 3 将转换数据进行 PCA 映射，输出每个数据点的 3 个主成分
    h = w = int(sqrt(feature_maps_pca.shape[0]))
    feature_maps_pca = feature_maps_pca.reshape(h, w, 3) # 计算图像的形状并将其重塑为图像
    torch.save(feature_maps_pca, save_path)

"""
替换输入图像预设的rick.jpeg图像，若加载失败则返回输入图像
"""
def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3: # 如果你传入的是单张图(H,W,C)，加一个维度变成(1,H,W,C)，统一处理为batch
        images = images[None, ...]
    images = (images * 255).round().astype("uint8") # 加入图像是归一化图像，乘以255并四舍五入到整数
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

# 加载单个特征图文件并转换为列向量特征序列
def load_experiments_features(feature_maps_path):
    feature_maps = []
    feature_map = torch.load(os.path.join(feature_maps_path))[1] # 载入特征图文件（假设 .pt 文件中是 tuple，第 2 个元素才是特征张量）
    feature_map = feature_map.reshape(feature_map.shape[0], -1).t()  # N X C 特征从 (C, H, W) 转为 (C, HW)，再转置为 (HW, C)，即每一行是一个空间位置的特征
    feature_maps.append(feature_map)

    return feature_maps


"""
加载某个 block 的 QK 并计算自注意力图
Args:
    unet_model: UNet 模型实例
    feature_map_path: 特征图路径
    block: block 名称
    t: 时间步
"""
def load_experiments_self_attn_maps(unet_model, feature_map_path, block, t):
    self_attn_maps = []
    block_idx = int(block.split('_')[-1]) # 从block名中解析出索引号，如 "output_block_4" → 4
    # 加载保存的q和k特征图
    self_attn_q = torch.load(os.path.join(feature_map_path, f"{block}_self_attn_q_time_{t}.pt"))
    self_attn_k = torch.load(os.path.join(feature_map_path, f"{block}_self_attn_k_time_{t}.pt"))
    # 根据 block 是 input 还是 output，从模型中读取注意力缩放因子（scale，通常是 1/sqrt(dim)）
    if "output_block" in block:
        scale = unet_model.output_blocks[block_idx][1].transformer_blocks[0].attn1.scale
    else:
        scale = unet_model.input_blocks[block_idx][1].transformer_blocks[0].attn1.scale
    sim = einsum('b i d, b j d -> b i j', self_attn_q, self_attn_k) * scale # 使用爱因斯坦求和计算相似度（QK 转置点积）：生成 (B, N, N) 的注意力图
    self_attn_map = sim.softmax(dim=-1)
    self_attn_map = rearrange(self_attn_map, 'h n m -> n (h m)')
    self_attn_maps.append(self_attn_map)

    return self_attn_maps

def visualize_self_attn_map(attn_map, save_path=None, title=None, size=None):
    """
    可视化 self-attention 映射 (N, M) 为热力图。
    """
    if isinstance(attn_map, torch.Tensor):
        attn_map = attn_map.detach().cpu().numpy()

    if size is None:
        size = int(np.sqrt(attn_map.shape[0]))

    img = attn_map[:, 0]  # 可选：取第一个 query 的 attention 分布
    img = img.reshape(size, size)

    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap='hot')
    if title:
        plt.title(title)
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    plt.close()

def attention_mask(attn_map, threshold=0.9):
    """
    基于平均 attention 权重提取前 threshold 区域掩码。
    """
    if isinstance(attn_map, torch.Tensor):
        attn_map = attn_map.detach().cpu().numpy()

    avg_attention = attn_map.mean(axis=1)
    h = w = int(np.sqrt(avg_attention.shape[0]))
    avg_attention = avg_attention.reshape(h, w)

    flat = avg_attention.flatten()
    sorted_vals = np.sort(flat)
    threshold_val = sorted_vals[int((1 - threshold) * len(flat))]
    binary_mask = (avg_attention >= threshold_val).astype(np.uint8)

    return binary_mask

def analyze_attention_regions(mask):
    """
    使用 skimage.label 对 attention 掩码进行连通区域分析。
    """
    label_map = label(mask, connectivity=2)
    num_regions = label_map.max()
    return label_map, num_regions

def save_binary_mask(mask, save_path):
    """
    保存二值 mask 为灰度图。
    """
    img = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
    img.save(save_path)

def visualize_attn_pipeline(attn_map, out_dir, name_prefix="attn", threshold=0.9):
    os.makedirs(out_dir, exist_ok=True)
    visualize_self_attn_map(attn_map, save_path=os.path.join(out_dir, f"{name_prefix}_heatmap.png"))
    mask = attention_mask(attn_map, threshold=threshold)
    save_binary_mask(mask, os.path.join(out_dir, f"{name_prefix}_mask.png"))
    label_map, region_count = analyze_attention_regions(mask)
    print(f"[{name_prefix}] Attention region count:", region_count)
    return mask, label_map, region_count




def generate_attn_mask(exp_config, model=None):
    exp_path_root = exp_config.twin_extraction.exp_path_root
    model_config = OmegaConf.load(f"{exp_config.twin_extraction.model_config}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if model is None:
        model = load_model_from_config(model_config, exp_config.twin_extraction.ckpt)
        model = model.to(device)
    unet_model = model.model.diffusion_model

    input_folder = os.path.join(exp_path_root, exp_config.twin_extraction.experiment_name, "feature_maps")
    features_all = []
    path = os.path.join(input_folder, 'output_block_11_self_attn_q_time_*.pt')
    maps = glob.glob(path)
    steps = [int(map.split('_')[-1].replace('.pt', '')) for map in maps]
    steps = [step for step in steps if step <= 541]

    for step in steps:
        for i in exp_config.attn_mask.input_blocks:
            block_name = f"input_block_{i}"
            fit_features = load_experiments_self_attn_maps(unet_model, input_folder, block_name, step)
            fit_features = torch.cat(fit_features, dim=0)
            features_all.append(fit_features.unsqueeze(0))
        for i in exp_config.attn_mask.output_blocks:
            block_name = f"output_block_{i}"
            fit_features = load_experiments_self_attn_maps(unet_model, input_folder, block_name, step)
            fit_features = torch.cat(fit_features, dim=0)
            features_all.append(fit_features.unsqueeze(0))

    features_all = torch.cat(features_all, 0)
    features_all = features_all.mean(0)  # 聚合后结果 [N, C]

    res = exp_config.attn_mask.res
    cluster_n = exp_config.attn_mask.n_clusters

    if exp_config.attn_mask.cluster_type == 'kmeans':
        kmeans = KMeans(n_clusters=cluster_n, n_init=5).fit(features_all.cpu().numpy())
        clusters = kmeans.labels_.reshape(res, res)
    elif exp_config.attn_mask.cluster_type == 'spectral':
        feats = features_all.reshape(res**2, 16, res**2).mean(1).cpu().numpy()
        sc = SpectralClustering(cluster_n, affinity='precomputed', n_init=100, assign_labels='kmeans')
        clusters = sc.fit_predict(feats).reshape(res, res)
    else:
        print('Choose one clustering type between kmeans and spectral!')
        return

    clusters = label(clusters)

    # 调用可视化
    attn_map_fake = torch.from_numpy(clusters.astype(np.float32)).flatten()[None, :, None]
    out_dir = os.path.join(exp_path_root, exp_config.twin_extraction.experiment_name, "attn_vis")
    visualize_attn_pipeline(attn_map=attn_map_fake.squeeze(0), out_dir=out_dir, name_prefix="attn_mask_step", threshold=exp_config.attn_mask.threshold)

    mask_path = os.path.join(exp_path_root, exp_config.twin_extraction.experiment_name, 'mask_odise.png')
    mask = np.array(Image.open(mask_path).convert('L')) / 255.

    labels_img = Image.fromarray(clusters.astype(np.int32)).convert('L').resize((512, 512), Image.NEAREST)
    labels_img = np.array(labels_img) + 1

    label_cnt = {}
    labels_cnt_tot = {}
    labels_img_masked = labels_img * mask
    for label_ in np.unique(labels_img):
        label_cnt[label_] = np.sum((labels_img_masked == label_) * 1)
        labels_cnt_tot[label_] = np.sum((labels_img == label_) * 1)

    mask_f = np.zeros_like(mask)
    for label_ in np.unique(labels_img):
        if label_cnt[label_] / labels_cnt_tot[label_] >= exp_config.attn_mask.threshold:
            mask_f[labels_img == label_] = 1.

    # 保存掩码及腐蚀版本
    save_dir = os.path.join(exp_path_root, exp_config.twin_extraction.experiment_name)
    Image.fromarray((mask_f * 255).astype(np.uint8)).convert('L').save(os.path.join(save_dir, 'mask_self_attn.png'))
    kernel = np.ones((exp_config.attn_mask.erosion_ksize, exp_config.attn_mask.erosion_ksize), np.uint8)
    img_erosion = cv2.erode((mask_f * 255).astype(np.uint8), kernel, iterations=exp_config.attn_mask.erosion_iter)
    Image.fromarray(img_erosion).convert('L').save(os.path.join(save_dir, 'mask_self_attn_erosion.png'))
