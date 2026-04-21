import math
import warnings
from typing import Tuple, Union
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, FloatTensor

from .maskgit.tokenizer import PretrainedTokenizer as MaskGITVQModel
from .taming.vqmodel import VQModel as TamingVQModel
from .llamagen.vq_model import VQModel as LlamaGenVQModel, ModelArgs as LlamaGenModelArgs


AVAILABLE_MODEL_NAMES = (
    'maskgit-vqgan-imagenet-f16-256',
    'taming/vqgan_imagenet_f16_1024',
    'taming/vqgan_imagenet_f16_16384',
    'amused/amused-256',
    'amused/amused-512',
    'llamagen/vq_ds16_c2i',
)


def make_vqmodel(name: str):
    assert name in AVAILABLE_MODEL_NAMES, f"Model {name} not available"

    if 'maskgit' in name:
        # build model & load weights
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message=r"You are using `torch.load` with `weights_only=False`*")
            vqmodel = MaskGITVQModel(f'ckpts/{name}.bin')
        # wrap model
        model = MaskGITVQModelWrapper(vqmodel)
        model.eval()
        model.requires_grad_(False)
        return model

    elif 'taming' in name:
        from pytorch_lightning.callbacks import ModelCheckpoint
        torch.serialization.add_safe_globals([ModelCheckpoint])
        # load config & build model
        config_path = f'ckpts/{name}.yaml'
        conf = OmegaConf.load(config_path)
        model_params = OmegaConf.to_container(conf.model.params)
        model_params.pop('lossconfig')
        vqmodel = TamingVQModel(**model_params)
        # load weights
        ckpt_path = f'ckpts/{name}.ckpt'
        weights = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        vqmodel.load_state_dict(weights['state_dict'], strict=False)
        del weights
        # wrap model
        model = TamingVQModelWrapper(vqmodel)
        model.eval()
        model.requires_grad_(False)
        return model

    elif 'llamagen' in name:
        # create model args & build model
        args = LlamaGenModelArgs(
            encoder_ch_mult=[1, 1, 2, 2, 4],
            decoder_ch_mult=[1, 1, 2, 2, 4],
            codebook_size=16384,
            codebook_embed_dim=8,
        )
        vqmodel = LlamaGenVQModel(args)
        # load weights
        ckpt_path = f'ckpts/{name}.pt'
        weights = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        vqmodel.load_state_dict(weights['model'], strict=True)
        del weights
        # wrap model
        model = LlamaGenVQModelWrapper(vqmodel)
        model.eval()
        model.requires_grad_(False)
        return model

    elif 'amused' in name:
        from diffusers import VQModel as aMUSEdVQModel
        # build model & load weights
        vq_model = aMUSEdVQModel.from_pretrained(name, subfolder='vqvae')
        # wrap model
        model = aMUSEdVQModelWrapper(vq_model)
        model.eval()
        model.requires_grad_(False)
        return model

    else:
        raise NotImplementedError(f"Model {name} not implemented")


class TamingVQModelWrapper(nn.Module):
    def __init__(self, vqmodel: TamingVQModel):
        super().__init__()
        self.vqmodel = vqmodel

    @property
    def downsample_factor(self):
        return 2 ** (self.vqmodel.encoder.num_resolutions - 1)

    def forward(self, x: Tensor):
        recx = self.vqmodel(x)[0]
        return recx

    def encode(self, x: Tensor):
        h = self.vqmodel.encoder(x)
        h = self.vqmodel.quant_conv(h)
        quant, emb_loss, info = self.vqmodel.quantize(h)
        indices = info[-1]
        return dict(h=h, quant=quant, indices=indices)

    def decode(self, z: Tensor):
        return self.vqmodel.decode(z)

    def decode_indices(self, indices: Tensor, shape: Tuple[int, ...]):
        quant = self.vqmodel.quantize.get_codebook_entry(indices, shape)
        return self.decode(quant)


class MaskGITVQModelWrapper(nn.Module):
    def __init__(self, vqmodel: MaskGITVQModel):
        super().__init__()
        self.vqmodel = vqmodel

    @property
    def downsample_factor(self):
        return 2 ** (len(self.vqmodel.encoder.config.channel_mult) - 1)

    def forward(self, x: Tensor):
        enc = self.encode(x)
        rec = self.decode(enc['quant'])
        return rec

    def encode(self, x: Tensor):
        hidden_states = self.vqmodel.encoder((x + 1) / 2)
        quantized_states, codebook_indices, codebook_loss = self.vqmodel.quantize(hidden_states)
        return dict(h=hidden_states, quant=quantized_states, indices=codebook_indices)

    def decode(self, z: Tensor):
        rec = self.vqmodel.decoder(z)
        return rec * 2 - 1

    def decode_indices(self, indices: Tensor, shape: Tuple[int, ...] = None):  # noqa
        dec = self.vqmodel.decode(indices)
        return dec * 2 - 1

    def decode_code_distribution(self, code_logits: Tensor = None, code_probs: Tensor = None, shape: Tuple[int, ...] = None):
        if (code_logits is None) == (code_probs is None):
            raise ValueError("Pass exactly one of code_logits or code_probs.")

        if code_logits is not None:
            probs = torch.softmax(code_logits.to(dtype=torch.float32), dim=-1)
        else:
            probs = code_probs.to(dtype=torch.float32)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

        if probs.ndim != 3:
            raise ValueError(f"Expected code distribution with shape [B, L, K], got {tuple(probs.shape)}")

        batch_size, num_tokens, codebook_size = probs.shape
        embedding = self.vqmodel.quantize.embedding.weight.to(device=probs.device, dtype=probs.dtype)
        if int(embedding.shape[0]) != int(codebook_size):
            raise ValueError(
                f"Codebook size mismatch: distribution has {codebook_size} channels, "
                f"but tokenizer embedding exposes {int(embedding.shape[0])} entries."
            )

        if shape is None:
            side = int(math.sqrt(num_tokens))
            if side * side != int(num_tokens):
                raise ValueError(
                    "Could not infer square token grid from code distribution. "
                    f"Got num_tokens={num_tokens}."
                )
            height, width = side, side
        else:
            if len(shape) == 4:
                _, height, width, _ = shape
            elif len(shape) == 3:
                _, height, width = shape
            else:
                raise ValueError(f"Unsupported shape for decode_code_distribution: {shape}")

        quantized = torch.einsum("blk,ke->ble", probs, embedding)
        quantized = quantized.view(batch_size, int(height), int(width), -1).permute(0, 3, 1, 2).contiguous()
        return self.decode(quantized)


class LlamaGenVQModelWrapper(nn.Module):
    def __init__(self, vqmodel: LlamaGenVQModel):
        super().__init__()
        self.vqmodel = vqmodel

    @property
    def downsample_factor(self):
        return 2 ** (len(self.vqmodel.config.encoder_ch_mult) - 1)

    def forward(self, x: Tensor):
        recx = self.vqmodel(x)[0]
        return recx

    def encode(self, x: Tensor):
        h = self.vqmodel.encoder(x)
        h = self.vqmodel.quant_conv(h)
        quant, emb_loss, info = self.vqmodel.quantize(h)
        indices = info[-1]
        h = F.normalize(h, p=2, dim=1)
        return dict(h=h, quant=quant, indices=indices)

    def decode(self, z: Tensor):
        return self.vqmodel.decode(z)

    def decode_indices(self, indices: Tensor, shape: Tuple[int, ...]):
        quant = self.vqmodel.quantize.get_codebook_entry(indices, shape, channel_first=False)
        quant = quant.permute(0, 3, 1, 2)
        return self.decode(quant)


class aMUSEdVQModelWrapper(nn.Module):
    def __init__(self, vqmodel):
        super().__init__()
        self.vqmodel = vqmodel

    @property
    def downsample_factor(self):
        return 2 ** (len(self.vqmodel.config['block_out_channels']) - 1)

    def forward(self, x: Union[Tensor, FloatTensor]):
        recx = self.vqmodel(x).sample
        return recx

    def encode(self, x: Union[Tensor, FloatTensor]):
        h = self.vqmodel.encode(x).latents
        quant, emb_loss, info = self.vqmodel.quantize(h)
        indices = info[-1]
        return dict(h=h, quant=quant, indices=indices)

    def decode(self, z: Union[Tensor, FloatTensor]):
        h = self.vqmodel.post_quant_conv(z)
        dec = self.vqmodel.decoder(h)
        return dec

    def decode_indices(self, indices: Tensor, shape: Tuple[int, ...]):
        quant = self.vqmodel.quantize.get_codebook_entry(indices, shape)
        return self.decode(quant)
