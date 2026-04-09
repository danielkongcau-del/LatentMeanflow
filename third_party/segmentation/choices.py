def _optional_import(module_name, attr_name):
    try:
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)
    except Exception:
        return None


ScoreNet = _optional_import("Nets.TUnet", "ScoreNet")
windowNet_window7_224 = _optional_import("Nets.WindowNet", "windowNet_window7_224")
SegConvFormer = _optional_import("Nets.SegConvFormer", "SegConvFormer")
ENet = _optional_import("Nets.ENet", "ENet")
SegConvFormerV3 = _optional_import("Nets.SegConvFormerv3", "SegConvFormerV3")
SegConvFormerV4 = _optional_import("Nets.SegConvFormerv4", "SegConvFormerV4")
SampleOrMHSAUNet = _optional_import("Nets.AttentionUnet", "SampleOrMHSAUNet")
CSNet = _optional_import("Nets.CSNet", "CSNet")
AblationCSNet = _optional_import("Nets.AblationCSNet", "AblationCSNet")
segformermodel = _optional_import("Nets.segformer", "segformermodel")
segmentermodel = _optional_import("Nets.segm.segmenter", "segmentermodel")
MUNet = _optional_import("Nets.munet", "MUNet")
CPUNet = _optional_import("Nets.cpunet", "CPUNet")
UNet = _optional_import("Nets.UNet", "UNet")
SegNet = _optional_import("Nets.SegNet", "SegNet")
LaneNet0508 = _optional_import("Nets.LaneNet0508", "LaneNet0508")
DeepLab = _optional_import("Nets.deeplabv3", "DeepLab")
moe_CSNet = _optional_import("Nets.moe_csnet", "moe_CSNet")
moe_CSNet_decoder = _optional_import("Nets.moe_csnet_decoder", "moe_CSNet_decoder")
SegNet_moe = _optional_import("Nets.SegNet_moe", "SegNet_moe")
config_transnuseg = _optional_import("Nets.nuseg_moe", "config_transnuseg")
config_transunet = _optional_import("Nets.transunet", "config_transunet")
config_swinsmt = _optional_import("Nets.swinsmt", "config_swinsmt")
MUNet_moe = _optional_import("Nets.munet_moe", "MUNet_moe")
config_patcher = _optional_import("Nets.patchformer", "config_patcher")
AgriSeg_Lighting = _optional_import("Nets.agriseg_lighting", "AgriSeg_Lighting")
AgriSeg = _optional_import("Nets.agriseg", "AgriSeg")
UNet_moe = _optional_import("Nets.UNet_moe", "UNet_moe")
UNet_Dmoe = _optional_import("Nets.UNet_dmoe", "UNet_Dmoe")
CPUNet_moe = _optional_import("Nets.cpunet_moe", "CPUNet_moe")
SwinMoBA_Seg = _optional_import("Nets.swinmoba", "SwinMoBA_Seg")
Dy_Seg = _optional_import("Nets.Dy_Seg", "Dy_Seg")
Dy_Seg_Global = _optional_import("Nets.Dy_Seg_Global", "Dy_Seg_Global")
CMTFNet = _optional_import("Nets.CMTFNet", "CMTFNet")
SMAFormer = _optional_import("Nets.SMAFormer", "SMAFormer")
BRAUnet = _optional_import("Nets.BRAUnet", "BRAUnet")
ScaleFormer = _optional_import("Nets.ScaleFormer", "ScaleFormer")
BRAUnetHA = _optional_import("Nets.GGNetHA.BRAUnetHA", "BRAUnetHA")
CMTFNetHA = _optional_import("Nets.GGNetHA.CMTFNetHA", "CMTFNetHA")
ScaleFormerHA = _optional_import("Nets.GGNetHA.ScaleFormerHA", "ScaleFormerHA")
config_transunetHA = _optional_import("Nets.GGNetHA.transunetHA", "config_transunetHA")
SwinMoBA_SegHA = _optional_import("Nets.GGNetHA.swinmobaHA", "SwinMoBA_SegHA")
segformermodelHA = _optional_import("Nets.GGNetHA.segformerHA", "segformermodelHA")


import torch
import torch.nn as nn
DeepLab_moe = _optional_import("Nets.deeplabv3_moe", "DeepLab_moe")
summary = _optional_import("torchsummary", "summary")
Segformer_moe = _optional_import("Nets.segformer_moe", "Segformer_moe")
from lr_scheduler import *


def get_lr_scheduler(optimizer, max_iters, sch_name):
    if sch_name == 'warmup_poly':
        return WarmupPolyLR(optimizer, max_iters=max_iters, power=0.9, warmup_factor=float(1.0/3), warmup_iters=0, warmup_method='linear')
    else:
        return None

def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin

def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]

def get_optimizer(model, optim_name):
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)
    if optim_name == 'adam':
        optimizer = torch.optim.Adam(parameters)
    elif optim_name == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=1e-4, momentum=0.9, weight_decay=1e-4)
    elif optim_name == 'adamw':
        optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.05)
        print('optimizer is adamw!!!!!')
    return optimizer


def get_criterion(out_channels, class_weights=None):
    if out_channels == 1:
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()
    return criterion


def choose_net(name, out_channels, img_size=512):
    if name == 'unet':
        if UNet is None:
            raise ImportError("UNet dependencies are unavailable in the current environment.")
        return UNet(n_classes=out_channels, drop_path_rate=0.1)
    elif name == 'segnet':
        return SegNet(label_nbr=out_channels, drop_path_rate=0.1)
    elif name == 'enet':
        return ENet(num_classes=out_channels)
    elif name == 'munet':
        return MUNet(n_classes=out_channels,bilinear=True, drop_path_rate=0.1)
    elif name == 'lanenet0508':
        return LaneNet0508(num_classes=out_channels)
    elif name == 'WindowNet':
        return windowNet_window7_224(num_classes=out_channels)
    elif name == 'cpunet':
        return CPUNet(n_classes=out_channels,bilinear=True, drop_path_rate=0.1)
    elif name == 'SegConvformerv2':
        return ScoreNet(n_classes=out_channels, num_heads=1, drop_path_rate=0.1)

    
    elif name == 'SegFormer':
        return segformermodel(num_classes=out_channels, )
    elif name == 'segmenter':
        return segmentermodel(num_classes=out_channels, use_checkpoint=True)
    elif name == 'deeplabv3-resnet':
        if DeepLab is None:
            raise ImportError("DeepLab v3+ dependencies are unavailable in the current environment.")
        return DeepLab(num_classes=out_channels, backbone='resnet', drop_path_rate=0.1,)    
    elif name == 'csnet':
        if CSNet is None:
            raise ImportError("CSNet dependencies are unavailable in the current environment.")
        return CSNet(n_classes=out_channels, num_heads=2, drop_path_rate=0.1, choice='CNNSimpleAttention',
                                patch_size=16)
    elif name == 'nuseg_moe':
        return config_transnuseg(img_size=512, n_classes=out_channels)

    elif name == 'swinsmt':
        return config_swinsmt(img_size=512, n_classes=out_channels)

    elif name == 'patcher':
        return config_patcher(img_size=512, n_classes=out_channels)

    elif name == 'TransUNet':
        return config_transunet(img_size=img_size, n_classes=out_channels)


    elif name == 'CMTFNet':
        return CMTFNet(num_classes=out_channels)

    elif name == 'BRAUNet':
        return BRAUnet(img_size=512, num_classes=out_channels)

    elif name == 'ScaleFormer':
        return ScaleFormer(num_classes=out_channels)

# ---------------------GG-Net 泛化性实验-------------------------

    elif name == 'CMTFNetHA':
        return CMTFNetHA(num_classes=out_channels)

    elif name == 'BRAUNetHA':
        return BRAUnetHA(img_size=512, num_classes=out_channels)

    elif name == "ScaleFormerHA":
        return ScaleFormerHA(num_classes=out_channels)

    elif name == "TransUNetHA":
        return config_transunetHA(n_classes=out_channels)

    elif name == "SwinUNetHA":
        return SwinMoBA_SegHA(use_shifted_window=True,
            use_relat_position=True,
            use_moe_swinmoba=False,
            topk=1,
            img_size=512,
            window_size=16,
            embed_dim=64,
            n_classes=out_channels,
            drop_path_rate=0.1,
            use_moe_decoder=False)

    elif name == "SegFormerHA":
        return segformermodelHA(num_classes=out_channels)




    elif name == 'CNNSampleAttention':
        return SampleOrMHSAUNet(n_classes=out_channels, num_heads=2, drop_path_rate=0.1,choice='CNNSampleAttention', patch_size=16)
    elif name == 'CNNAttention':
        return SampleOrMHSAUNet(n_classes=out_channels, num_heads=2, drop_path_rate=0.1, choice='CNNAttention', patch_size=16)
    elif name == 'SampleAttention':
        return SampleOrMHSAUNet(n_classes=out_channels, num_heads=2, drop_path_rate=0.1,choice='SampleAttention', patch_size=16)
    elif name == 'Attention':
        return SampleOrMHSAUNet(n_classes=out_channels, num_heads=2, drop_path_rate=0.1,choice='Attention', patch_size=16)


    elif name == "moe_csnet":
        return moe_CSNet(n_classes=out_channels, num_heads=2, drop_path_rate=0.1, choice='CNNSimpleAttention',
                                patch_size=16)
    elif name == "moe_csnet_decoder":
        return moe_CSNet_decoder(n_classes=out_channels, num_heads=2, drop_path_rate=0.1, choice='CNNSimpleAttention',
                                patch_size=16)

    elif name == 'SegConvformer':
        return SegConvFormer(n_classes=out_channels, embeddingdim=64, num_heads=1, drop_path_rate=0.1)
    elif name == 'agriseg_lighting':
        return AgriSeg_Lighting(n_classes=out_channels,
                                drop_path_rate=0.1,
                                multi_k=6,
                                expert_num=2,
                                use_moe_encoder=False, use_moe_decoder=True,
                                choice_moe='dmoe',
                                learnable_vec=1)
    elif name == 'agriseg_lighting_new':
        return AgriSeg_Lighting(n_classes=out_channels,
                                drop_path_rate=0.1,
                                multi_k=6,
                                expert_num=2,
                                use_moe_encoder=False, use_moe_decoder=True,
                                choice_moe='dmoe',
                                learnable_vec=-1)
    elif name == 'agriseg_lighting_new_e3_moe':
        return AgriSeg_Lighting(n_classes=out_channels,
                                drop_path_rate=0.1,
                                multi_k=6,
                                expert_num=3,
                                use_moe_encoder=False, use_moe_decoder=True,
                                choice_moe='moe',
                                learnable_vec=-1)
    elif name == 'agriseg_lighting_e3_dmoe_p(nox)_ox':
        return AgriSeg_Lighting(n_classes=out_channels,
                                drop_path_rate=0.1,
                                multi_k=6,
                                expert_num=3,
                                use_moe_encoder=False, use_moe_decoder=True,
                                choice_moe='dmoe',
                                learnable_vec=-1)
    elif name == 'agriseg_lighting_e2_dmoe_npr(o)_ox':
        return AgriSeg_Lighting(n_classes=out_channels,
                                drop_path_rate=0.1,
                                multi_k=6,
                                expert_num=2,
                                use_moe_encoder=False, use_moe_decoder=True,
                                choice_moe='dmoe',
                                learnable_vec=-1)
# -------------------------------消融lambda（病害数据集）------------------------------
    elif name == 'agriseg_lighting_disease_a=-1':
        return AgriSeg_Lighting(n_classes=out_channels,
                                drop_path_rate=0.1,
                                multi_k=6,
                                expert_num=2,
                                use_moe_encoder=False, use_moe_decoder=True,
                                choice_moe='dmoe',
                                learnable_vec=-1)
    elif name == 'agriseg_lighting_disease_sigmoid(a=-1)':
        return AgriSeg_Lighting(n_classes=out_channels,
                                drop_path_rate=0.1,
                                multi_k=6,
                                expert_num=2,
                                use_moe_encoder=False, use_moe_decoder=True,
                                choice_moe='dmoe',
                                learnable_vec=-1)
    elif name == 'agriseg_lighting_disease_lamda':
        return AgriSeg_Lighting(n_classes=out_channels,
                                drop_path_rate=0.1,
                                multi_k=6,
                                expert_num=2,
                                use_moe_encoder=False, use_moe_decoder=True,
                                choice_moe='dmoe',
                                learnable_vec=-1)
    elif name == 'agriseg_lighting_disease_sigmoid(lamda)':
        return AgriSeg_Lighting(n_classes=out_channels,
                                drop_path_rate=0.1,
                                multi_k=6,
                                expert_num=2,
                                use_moe_encoder=False, use_moe_decoder=True,
                                choice_moe='dmoe',
                                learnable_vec=-1)
    elif name == 'agriseg_lighting_disease_a=-1_norm':
        return AgriSeg_Lighting(n_classes=out_channels,
                                drop_path_rate=0.1,
                                multi_k=6,
                                expert_num=2,
                                use_moe_encoder=False, use_moe_decoder=True,
                                choice_moe='dmoe',
                                learnable_vec=-1)
    elif name == 'agriseg_lighting_disease_a=-1_norm_proj':
        return AgriSeg_Lighting(n_classes=out_channels,
                                drop_path_rate=0.1,
                                multi_k=6,
                                expert_num=2,
                                use_moe_encoder=False, use_moe_decoder=True,
                                choice_moe='dmoe',
                                learnable_vec=-1)
    elif name == 'agriseg_lighting_disease_a=-1_e3':
        return AgriSeg_Lighting(n_classes=out_channels,
                                drop_path_rate=0.1,
                                multi_k=6,
                                expert_num=3,
                                use_moe_encoder=False, use_moe_decoder=True,
                                choice_moe='dmoe',
                                learnable_vec=-1)
    elif name == 'agriseg_lighting_disease_a=-1_e3_moe':
        return AgriSeg_Lighting(n_classes=out_channels,
                                drop_path_rate=0.1,
                                multi_k=6,
                                expert_num=3,
                                use_moe_encoder=False, use_moe_decoder=True,
                                choice_moe='moe',
                                learnable_vec=-1)
    elif name == 'agriseg_lighting_disease_a=-1_e3_dmoe_id':
        return AgriSeg_Lighting(n_classes=out_channels,
                                drop_path_rate=0.1,
                                multi_k=6,
                                expert_num=3,
                                use_moe_encoder=False, use_moe_decoder=True,
                                choice_moe='dmoe',
                                learnable_vec=-1)

    elif name == 'agriseg_lighting_disease_ab3':
        return AgriSeg_Lighting(n_classes=out_channels,
                                drop_path_rate=0.1,
                                multi_k=6,
                                expert_num=2,
                                use_moe_encoder=False, use_moe_decoder=True,
                                choice_moe='dmoe',
                                learnable_vec=0.75)
    elif name == 'agriseg_lighting_disease_ab4':
        return AgriSeg_Lighting(n_classes=out_channels,
                                drop_path_rate=0.1,
                                multi_k=6,
                                expert_num=2,
                                use_moe_encoder=False, use_moe_decoder=True,
                                choice_moe='dmoe',
                                learnable_vec=0.5)
    elif name == 'agriseg_lighting_disease_ab5':
        return AgriSeg_Lighting(n_classes=out_channels,
                                drop_path_rate=0.1,
                                multi_k=6,
                                expert_num=2,
                                use_moe_encoder=False, use_moe_decoder=True,
                                choice_moe='dmoe',
                                learnable_vec=0.25)
# -------------------------------消融lambda（病害数据集）------------------------------
    elif name == 'agriseg_lighting_disease_ab20':
        return AgriSeg_Lighting(n_classes=out_channels,
                                drop_path_rate=0.1,
                                multi_k=6,
                                expert_num=3,
                                use_moe_encoder=False, use_moe_decoder=True,
                                choice_moe='dmoe',
                                learnable_vec=1)
# -------------------------------消融lambda（病害数据集三专家）------------------------------
    elif name == 'agriseg_lighting_disease_ab6':
        return AgriSeg_Lighting(n_classes=out_channels,
                                drop_path_rate=0.1,
                                multi_k=6,
                                expert_num=3,
                                use_moe_encoder=False, use_moe_decoder=True,
                                choice_moe='dmoe',
                                learnable_vec=-1)

    elif name == 'agriseg_lighting_disease_ab7':
        return AgriSeg_Lighting(n_classes=out_channels,
                                drop_path_rate=0.1,
                                multi_k=6,
                                expert_num=3,
                                use_moe_encoder=False, use_moe_decoder=True,
                                choice_moe='dmoe',
                                learnable_vec=1)
    elif name == 'agriseg_lighting_disease_ab8':
        return AgriSeg_Lighting(n_classes=out_channels,
                                drop_path_rate=0.1,
                                multi_k=6,
                                expert_num=3,
                                use_moe_encoder=False, use_moe_decoder=True,
                                choice_moe='dmoe',
                                learnable_vec=0.75)
    elif name == 'agriseg_lighting_disease_ab9':
        return AgriSeg_Lighting(n_classes=out_channels,
                                drop_path_rate=0.1,
                                multi_k=6,
                                expert_num=3,
                                use_moe_encoder=False, use_moe_decoder=True,
                                choice_moe='dmoe',
                                learnable_vec=0.5)
    elif name == 'agriseg_lighting_disease_ab10':
        return AgriSeg_Lighting(n_classes=out_channels,
                                drop_path_rate=0.1,
                                multi_k=6,
                                expert_num=3,
                                use_moe_encoder=False, use_moe_decoder=True,
                                choice_moe='dmoe',
                                learnable_vec=0.25)
    # -------------------------------消融moe（病害数据集三专家，dome—>moe）------------------------------
    elif name == 'agriseg_lighting_disease_ab11':
        return AgriSeg_Lighting(n_classes=out_channels,
                                drop_path_rate=0.1,
                                multi_k=6,
                                expert_num=3,
                                use_moe_encoder=False, use_moe_decoder=True,
                                choice_moe='moe',
                                learnable_vec=1)
    # -------------------------------消融dmoe（杂草数据集二专家，最好的dmoe）------------------------------
    elif name == 'agriseg_lighting_weed':
        return AgriSeg_Lighting(n_classes=out_channels,
                                drop_path_rate=0.1,
                                multi_k=6,
                                expert_num=2,
                                use_moe_encoder=False, use_moe_decoder=True,
                                choice_moe='moe',
                                learnable_vec=1)
    elif name == 'agriseg_lighting_grape':
        return AgriSeg_Lighting(n_classes=out_channels,
                                drop_path_rate=0.1,
                                multi_k=6,
                                expert_num=3,
                                use_moe_encoder=False, use_moe_decoder=True,
                                choice_moe='dmoe',
                                learnable_vec=1)
    # -------------------------------moe泛化性研究------------------------------
    if name == 'unet_moe':
        return UNet_moe(n_classes=out_channels,
                                drop_path_rate=0.1,
                             expert_num=3,
                             use_moe_decoder=True,
                                choice_moe='dmoe',
                                learnable_vec=1)
    if name == 'unet_Dmoe':
        return UNet_Dmoe(n_classes=out_channels, drop_path_rate=0.1)
    if name == 'transunet_moe':
        return config_transunet(img_size=512, n_classes=out_channels)
    elif name == 'CSNet_dmoe':
        return CSNet(n_classes=out_channels, num_heads=2, drop_path_rate=0.1, choice='CNNSimpleAttention',
                                patch_size=16)
    elif name == 'deeplabv3-resnet_moe':
        return DeepLab_moe(num_classes=out_channels, backbone='resnet', drop_path_rate=0.1, use_moe=True, expert_num=2, learnable_vec=0.5)
    elif name == 'SegConvformer_moe':
        return Segformer_moe(num_classes=out_channels,
                     dims=(32, 64, 128, 256),
                     heads=(1, 1, 1, 1),
                     ff_expansion=(4, 4, 4, 4),
                     reduction_ratio=(1, 1, 1, 1),
                     num_layers=4,
                     channels=3,
                     decoder_dim=256,
                     use_checkpoint=True,
                     drop_path_rate=0.1, bilinear=True, use_moe=True, expert_num=2, learnable_vec=0.5)
    elif name == 'cpu-net_moe':
        return CPUNet_moe(n_classes=out_channels, bilinear=True, drop_path_rate=0.1,use_moe=True,expert_num=2,learnable_vec=0.5)
    elif name == 'segnet_moe':
        return SegNet_moe(label_nbr=out_channels, drop_path_rate=0.1,bilinear=True, use_moe=True, expert_num=2, learnable_vec=0.5)
    elif name == 'munet_moe':
        return MUNet_moe(n_classes=out_channels,bilinear=True, drop_path_rate=0.1)

# -----------------------------------test moba------------------------------------------------
    elif name == 'SwinUNet':
        return SwinMoBA_Seg(
            use_shifted_window=True,
            use_relat_position=True,
            use_moe_swinmoba=False,
            topk=1,
            img_size=512,
            window_size=16,
            embed_dim=64,
            n_classes=out_channels,
            drop_path_rate=0.1,
            use_moe_decoder=False,)


    elif name == 'Dy_Seg':
        return Dy_Seg(
            use_shifted_window=True,
            use_relat_position=True,
            query_ratio=[64, 16, 4, 1],
            img_size=512,
            window_size=16,
            embed_dim=64,
            n_classes=out_channels,
            drop_path_rate=0.1,

         )

    elif name == 'Dy_Seg_Global':
        return Dy_Seg_Global(
            poolstep=[8,4,2,1],
            Nq=5,
            img_size=512,
            window_size=16,
            embed_dim=64,
            n_classes=out_channels,
            drop_path_rate=0.1,
            depths=[2, 2, 6, 2],  #depths=[2, 2, 18, 2],
         )



    elif name == 'ab1_swin_moba':   # shifted window/related position embedding/topk=3/window_size 16
        return SwinMoBA_Seg(
            use_shifted_window=True,
            use_relat_position=True,
            use_moe_swinmoba=True,
            topk=3,
            img_size=512,
            window_size=16,
            embed_dim=64,
            n_classes=out_channels,
            drop_path_rate=0.1,
            use_moe_decoder=False,)
    elif name == 'ab2_swin_moba':   # shifted window/related position embedding/topk=3/window_size 16
        return SwinMoBA_Seg(
            use_shifted_window=True,
            use_relat_position=False,
            use_moe_swinmoba=True,
            topk=3,
            img_size=512,
            window_size=16,
            embed_dim=64,
            n_classes=out_channels,
            drop_path_rate=0.1,
            use_moe_decoder=False,)
    # 请根据消融试验1 的结果进行填写
    elif name == 'ab3_swin_moba':   # shifted window/related position embedding/topk=3/window_size 16
        return SwinMoBA_Seg(
            use_shifted_window=False,
            use_relat_position=False,   # 如果ab1好那么True, 如果ab2好那么False
            use_moe_swinmoba=True,
            topk=3,
            img_size=512,
            window_size=16,
            embed_dim=64,
            n_classes=out_channels,
            drop_path_rate=0.1,
            use_moe_decoder=False,)
    # 请根据消融试验1和2 的结果进行填写
    elif name == 'ab4_swin_moba':   # shifted window/related position embedding/topk=3/window_size 16
        return SwinMoBA_Seg(
            use_shifted_window=False,   # 如果ab1 or ab2好，那么选择True， 如果Ab3好那么选择False
            use_relat_position=False,   # 如果ab1好那么True, 如果ab2好那么False
            use_moe_swinmoba=True,
            topk=2,      #
            img_size=512,
            window_size=16,
            embed_dim=64,
            n_classes=out_channels,
            drop_path_rate=0.1,
            use_moe_decoder=False,)
    elif name == 'ab5_swin_moba':   # shifted window/related position embedding/topk=3/window_size 16
        return SwinMoBA_Seg(
            use_shifted_window=False,   # 如果ab1 or ab2好，那么选择True， 如果Ab3好那么选择False
            use_relat_position=False,   # 如果ab1好那么True, 如果ab2好那么False
            use_moe_swinmoba=True,
            topk=4,      #
            img_size=512,
            window_size=16,
            embed_dim=64,
            n_classes=out_channels,
            drop_path_rate=0.1,
            use_moe_decoder=False,)

    elif name == 'ab6_swin_moba':   # shifted window/related position embedding/topk=3/window_size 16
        return SwinMoBA_Seg(
            use_shifted_window=True,   # 如果ab1 or ab2好，那么选择True， 如果Ab3好那么选择False
            use_relat_position=False,   # 如果ab1好那么True, 如果ab2好那么False
            use_moe_swinmoba=True,
            topk=4,      #
            img_size=512,
            window_size=16,
            embed_dim=64,
            n_classes=out_channels,
            drop_path_rate=0.1,
            use_moe_decoder=False,)

    elif name == 'swin_moba_disease_usw':   # shifted window/related position embedding/topk=3/window_size 16
        return SwinMoBA_Seg(
            use_shifted_window=True,   # 如果ab1 or ab2好，那么选择True， 如果Ab3好那么选择False
            use_relat_position=False,   # 如果ab1好那么True, 如果ab2好那么False
            use_moe_swinmoba=True,
            topk=3,      #
            img_size=512,
            window_size=16,
            embed_dim=64,
            n_classes=out_channels,
            drop_path_rate=0.1,
            use_moe_decoder=False,)
    elif name == 'swin_moba_weed_nusw_top4':   # shifted window/related position embedding/topk=3/window_size 16
        return SwinMoBA_Seg(
            use_shifted_window=False,   # 如果ab1 or ab2好，那么选择True， 如果Ab3好那么选择False
            use_relat_position=False,   # 如果ab1好那么True, 如果ab2好那么False
            use_moe_swinmoba=True,
            topk=4,      #
            img_size=512,
            window_size=16,
            embed_dim=64,
            n_classes=out_channels,
            drop_path_rate=0.1,
            use_moe_decoder=False,)

if __name__ == '__main__':
    net_names = [
        # 'enet'，
        # 'lanenet0508'
    ]
    resizes = [
        # (320, 320),
        # (224, 224)
        (528, 960)
    ]

    # batch_cal_comlexity(net_names, resizes, out_channels=2, method=0)
    summary(choose_net(net_names[0], 2).cuda(), (3, resizes[0][0], resizes[0][1]))
