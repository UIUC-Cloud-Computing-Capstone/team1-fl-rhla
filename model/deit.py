""" DeiT - Data-efficient Image Transformers

DeiT model defs and weights from https://github.com/facebookresearch/deit, original copyright below

paper: `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

paper: `DeiT III: Revenge of the ViT` - https://arxiv.org/abs/2204.07118

Modifications copyright 2021, Ross Wightman
"""
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from functools import partial
from typing import Optional, Type

import torch
from torch import nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import resample_abs_pos_embed
from timm.models.vision_transformer import VisionTransformer, trunc_normal_, checkpoint_filter_fn
from ._builder import build_model_with_cfg
from ._registry import generate_default_cfgs, register_model, register_model_deprecations

__all__ = ['VisionTransformerDistilled']  # model_registry will add each entrypoint fn to this

def resolve_pretrained_cfg(
        variant: str,
        pretrained_cfg: Optional[Union[str, Dict[str, Any]]] = None,
        pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
) -> PretrainedCfg:
    """Resolve pretrained configuration from various sources."""
    model_with_tag = variant
    pretrained_tag = None
    if pretrained_cfg:
        if isinstance(pretrained_cfg, dict):
            # pretrained_cfg dict passed as arg, validate by converting to PretrainedCfg
            pretrained_cfg = PretrainedCfg(**pretrained_cfg)
        elif isinstance(pretrained_cfg, str):
            pretrained_tag = pretrained_cfg
            pretrained_cfg = None

    # fallback to looking up pretrained cfg in model registry by variant identifier
    if not pretrained_cfg:
        if pretrained_tag:
            model_with_tag = '.'.join([variant, pretrained_tag])
        pretrained_cfg = get_pretrained_cfg(model_with_tag)

    if not pretrained_cfg:
        _logger.warning(
            f"No pretrained configuration specified for {model_with_tag} model. Using a default."
            f" Please add a config to the model pretrained_cfg registry or pass explicitly.")
        pretrained_cfg = PretrainedCfg()  # instance with defaults

    pretrained_cfg_overlay = pretrained_cfg_overlay or {}
    if not pretrained_cfg.architecture:
        pretrained_cfg_overlay.setdefault('architecture', variant)
    pretrained_cfg = dataclasses.replace(pretrained_cfg, **pretrained_cfg_overlay)

    return pretrained_cfg

def _update_default_model_kwargs(pretrained_cfg, kwargs, kwargs_filter) -> None:
    """ Update the default_cfg and kwargs before passing to model

    Args:
        pretrained_cfg: input pretrained cfg (updated in-place)
        kwargs: keyword args passed to model build fn (updated in-place)
        kwargs_filter: keyword arg keys that must be removed before model __init__
    """
    # Set model __init__ args that can be determined by default_cfg (if not already passed as kwargs)
    default_kwarg_names = ('num_classes', 'global_pool', 'in_chans')
    if pretrained_cfg.get('fixed_input_size', False):
        # if fixed_input_size exists and is True, model takes an img_size arg that fixes its input size
        default_kwarg_names += ('img_size',)

    for n in default_kwarg_names:
        # for legacy reasons, model __init__args uses img_size + in_chans as separate args while
        # pretrained_cfg has one input_size=(C, H ,W) entry
        if n == 'img_size':
            input_size = pretrained_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[-2:])
        elif n == 'in_chans':
            input_size = pretrained_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[0])
        elif n == 'num_classes':
            default_val = pretrained_cfg.get(n, None)
            # if default is < 0, don't pass through to model
            if default_val is not None and default_val >= 0:
                kwargs.setdefault(n, pretrained_cfg[n])
        else:
            default_val = pretrained_cfg.get(n, None)
            if default_val is not None:
                kwargs.setdefault(n, pretrained_cfg[n])

    # Filter keyword args for task specific model variants (some 'features only' models, etc.)
    _filter_kwargs(kwargs, names=kwargs_filter)

def _filter_kwargs(kwargs: Dict[str, Any], names: List[str]) -> None:
    if not kwargs or not names:
        return
    for n in names:
        kwargs.pop(n, None)

def build_model_with_cfg(
        model_cls: Union[Type[ModelT], Callable[..., ModelT]],
        variant: str,
        pretrained: bool,
        pretrained_cfg: Optional[Dict] = None,
        pretrained_cfg_overlay: Optional[Dict] = None,
        model_cfg: Optional[Any] = None,
        feature_cfg: Optional[Dict] = None,
        pretrained_strict: bool = True,
        pretrained_filter_fn: Optional[Callable] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        kwargs_filter: Optional[Tuple[str]] = None,
        **kwargs,
) -> ModelT:
    """ Build model with specified default_cfg and optional model_cfg

    This helper fn aids in the construction of a model including:
      * handling default_cfg and associated pretrained weight loading
      * passing through optional model_cfg for models with config based arch spec
      * features_only model adaptation
      * pruning config / model adaptation

    Args:
        model_cls: Model class
        variant: Model variant name
        pretrained: Load the pretrained weights
        pretrained_cfg: Model's pretrained weight/task config
        pretrained_cfg_overlay: Entries that will override those in pretrained_cfg
        model_cfg: Model's architecture config
        feature_cfg: Feature extraction adapter config
        pretrained_strict: Load pretrained weights strictly
        pretrained_filter_fn: Filter callable for pretrained weights
        cache_dir: Override model cache dir for Hugging Face Hub and Torch checkpoints
        kwargs_filter: Kwargs keys to filter (remove) before passing to model
        **kwargs: Model args passed through to model __init__
    """
    pruned = kwargs.pop('pruned', False)
    features = False
    feature_cfg = feature_cfg or {}

    # resolve and update model pretrained config and model kwargs
    pretrained_cfg = resolve_pretrained_cfg(
        variant,
        pretrained_cfg=pretrained_cfg,
        pretrained_cfg_overlay=pretrained_cfg_overlay
    )
    pretrained_cfg = pretrained_cfg.to_dict()

    _update_default_model_kwargs(pretrained_cfg, kwargs, kwargs_filter)

    # Setup for feature extraction wrapper done at end of this fn
    if kwargs.pop('features_only', False):
        features = True
        feature_cfg.setdefault('out_indices', (0, 1, 2, 3, 4))
        if 'out_indices' in kwargs:
            feature_cfg['out_indices'] = kwargs.pop('out_indices')
        if 'feature_cls' in kwargs:
            feature_cfg['feature_cls'] = kwargs.pop('feature_cls')

    # Instantiate the model
    if model_cfg is None:
        model = model_cls(**kwargs)
    else:
        model = model_cls(cfg=model_cfg, **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg  # alias for backwards compat

    # if pruned:
    #     model = adapt_model_from_file(model, variant)

    # For classification models, check class attr, then kwargs, then default to 1k, otherwise 0 for feats
    num_classes_pretrained = 0 if features else getattr(model, 'num_classes', kwargs.get('num_classes', 1000))
    # if pretrained:
    #     load_pretrained(
    #         model,
    #         pretrained_cfg=pretrained_cfg,
    #         num_classes=num_classes_pretrained,
    #         in_chans=kwargs.get('in_chans', 3),
    #         filter_fn=pretrained_filter_fn,
    #         strict=pretrained_strict,
    #         cache_dir=cache_dir,
    #     )

    # Wrap the model in a feature extraction module if enabled
    # if features:
    #     use_getter = False
    #     if 'feature_cls' in feature_cfg:
    #         feature_cls = feature_cfg.pop('feature_cls')
    #         if isinstance(feature_cls, str):
    #             feature_cls = feature_cls.lower()

    #             # flatten_sequential only valid for some feature extractors
    #             if feature_cls not in ('dict', 'list', 'hook'):
    #                 feature_cfg.pop('flatten_sequential', None)

    #             if 'hook' in feature_cls:
    #                 feature_cls = FeatureHookNet
    #             elif feature_cls == 'list':
    #                 feature_cls = FeatureListNet
    #             elif feature_cls == 'dict':
    #                 feature_cls = FeatureDictNet
    #             elif feature_cls == 'fx':
    #                 feature_cls = FeatureGraphNet
    #             elif feature_cls == 'getter':
    #                 use_getter = True
    #                 feature_cls = FeatureGetterNet
    #             else:
    #                 assert False, f'Unknown feature class {feature_cls}'
    #     else:
    #         feature_cls = FeatureListNet

    #     output_fmt = getattr(model, 'output_fmt', None)
    #     if output_fmt is not None and not use_getter:  # don't set default for intermediate feat getter
    #         feature_cfg.setdefault('output_fmt', output_fmt)

    #     model = feature_cls(model, **feature_cfg)
    #     model.pretrained_cfg = pretrained_cfg_for_features(pretrained_cfg)  # add back pretrained cfg
    #     model.default_cfg = model.pretrained_cfg  # alias for rename backwards compat (default_cfg -> pretrained_cfg)

    return model

class VisionTransformerDistilled(VisionTransformer):
    """ Vision Transformer w/ Distillation Token and Head

    Distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, *args, **kwargs):
        weight_init = kwargs.pop('weight_init', '')
        super().__init__(*args, **kwargs, weight_init='skip')
        assert self.global_pool in ('token',)
        dd = {'device': kwargs.get('device', None), 'dtype': kwargs.get('dtype', None)}

        self.num_prefix_tokens = 2
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim, **dd))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + self.num_prefix_tokens, self.embed_dim, **dd))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes, **dd) if self.num_classes > 0 else nn.Identity()
        self.distilled_training = False  # must set this True to train w/ distillation token

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        trunc_normal_(self.dist_token, std=.02)
        super().init_weights(mode=mode)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed|dist_token',
            blocks=[
                (r'^blocks\.(\d+)', None),
                (r'^norm', (99999,))]  # final norm w/ last block
        )

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head, self.head_dist

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def set_distilled_training(self, enable=True):
        self.distilled_training = enable

    def _pos_embed(self, x):
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            prev_grid_size = self.patch_embed.grid_size
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                new_size=(H, W),
                old_size=prev_grid_size,
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            x = torch.cat((
                self.cls_token.expand(x.shape[0], -1, -1),
                self.dist_token.expand(x.shape[0], -1, -1),
                x),
                dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            x = torch.cat((
                self.cls_token.expand(x.shape[0], -1, -1),
                self.dist_token.expand(x.shape[0], -1, -1),
                x),
                dim=1)
            x = x + pos_embed
        return self.pos_drop(x)

    def forward_head(self, x, pre_logits: bool = False) -> torch.Tensor:
        x, x_dist = x[:, 0], x[:, 1]
        if pre_logits:
            return (x + x_dist) / 2
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.distilled_training and self.training and not torch.jit.is_scripting():
            # only return separate classification predictions when training in distilled mode
            return x, x_dist
        else:
            # during standard train / finetune, inference average the classifier predictions
            return (x + x_dist) / 2


def _create_deit(variant, pretrained=False, distilled=False, **kwargs):
    out_indices = kwargs.pop('out_indices', 3)
    model_cls = VisionTransformerDistilled if distilled else VisionTransformer
    model = build_model_with_cfg(
        model_cls,
        variant,
        pretrained,
        pretrained_filter_fn=partial(checkpoint_filter_fn, adapt_layer_scale=True),
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        'license': 'apache-2.0',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    # deit models (FB weights)
    'deit_tiny_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'),
    'deit_small_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'),
    'deit_base_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth'),
    'deit_base_patch16_384.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
        input_size=(3, 384, 384), crop_pct=1.0),

    'deit_tiny_distilled_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth',
        classifier=('head', 'head_dist')),
    'deit_small_distilled_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
        classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth',
        classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_384.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        input_size=(3, 384, 384), crop_pct=1.0,
        classifier=('head', 'head_dist')),

    'deit3_small_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_small_224_1k.pth'),
    'deit3_small_patch16_384.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_small_384_1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit3_medium_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_medium_224_1k.pth'),
    'deit3_base_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_base_224_1k.pth'),
    'deit3_base_patch16_384.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_base_384_1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit3_large_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_large_224_1k.pth'),
    'deit3_large_patch16_384.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_large_384_1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit3_huge_patch14_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_huge_224_1k.pth'),

    'deit3_small_patch16_224.fb_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_small_224_21k.pth',
        crop_pct=1.0),
    'deit3_small_patch16_384.fb_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_small_384_21k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit3_medium_patch16_224.fb_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_medium_224_21k.pth',
        crop_pct=1.0),
    'deit3_base_patch16_224.fb_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_base_224_21k.pth',
        crop_pct=1.0),
    'deit3_base_patch16_384.fb_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_base_384_21k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit3_large_patch16_224.fb_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_large_224_21k.pth',
        crop_pct=1.0),
    'deit3_large_patch16_384.fb_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_large_384_21k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit3_huge_patch14_224.fb_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_huge_224_21k_v1.pth',
        crop_pct=1.0),
})


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = _create_deit('deit_tiny_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_deit('deit_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT base model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_deit('deit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT base model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_deit('deit_base_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs) -> VisionTransformerDistilled:
    """ DeiT-tiny distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = _create_deit(
        'deit_tiny_distilled_patch16_224', pretrained=pretrained, distilled=True, **dict(model_args, **kwargs))
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs) -> VisionTransformerDistilled:
    """ DeiT-small distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_deit(
        'deit_small_distilled_patch16_224', pretrained=pretrained, distilled=True, **dict(model_args, **kwargs))
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs) -> VisionTransformerDistilled:
    """ DeiT-base distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_deit(
        'deit_base_distilled_patch16_224', pretrained=pretrained, distilled=True, **dict(model_args, **kwargs))
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs) -> VisionTransformerDistilled:
    """ DeiT-base distilled model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_deit(
        'deit_base_distilled_patch16_384', pretrained=pretrained, distilled=True, **dict(model_args, **kwargs))
    return model


@register_model
def deit3_small_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-3 small model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, no_embed_class=True, init_values=1e-6)
    model = _create_deit('deit3_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def deit3_small_patch16_384(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-3 small model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, no_embed_class=True, init_values=1e-6)
    model = _create_deit('deit3_small_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def deit3_medium_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-3 medium model @ 224x224 (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=512, depth=12, num_heads=8, no_embed_class=True, init_values=1e-6)
    model = _create_deit('deit3_medium_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def deit3_base_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-3 base model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, no_embed_class=True, init_values=1e-6)
    model = _create_deit('deit3_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def deit3_base_patch16_384(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-3 base model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, no_embed_class=True, init_values=1e-6)
    model = _create_deit('deit3_base_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def deit3_large_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-3 large model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, no_embed_class=True, init_values=1e-6)
    model = _create_deit('deit3_large_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def deit3_large_patch16_384(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-3 large model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, no_embed_class=True, init_values=1e-6)
    model = _create_deit('deit3_large_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def deit3_huge_patch14_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-3 base model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, no_embed_class=True, init_values=1e-6)
    model = _create_deit('deit3_huge_patch14_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


register_model_deprecations(__name__, {
    'deit3_small_patch16_224_in21ft1k': 'deit3_small_patch16_224.fb_in22k_ft_in1k',
    'deit3_small_patch16_384_in21ft1k': 'deit3_small_patch16_384.fb_in22k_ft_in1k',
    'deit3_medium_patch16_224_in21ft1k': 'deit3_medium_patch16_224.fb_in22k_ft_in1k',
    'deit3_base_patch16_224_in21ft1k': 'deit3_base_patch16_224.fb_in22k_ft_in1k',
    'deit3_base_patch16_384_in21ft1k': 'deit3_base_patch16_384.fb_in22k_ft_in1k',
    'deit3_large_patch16_224_in21ft1k': 'deit3_large_patch16_224.fb_in22k_ft_in1k',
    'deit3_large_patch16_384_in21ft1k': 'deit3_large_patch16_384.fb_in22k_ft_in1k',
    'deit3_huge_patch14_224_in21ft1k': 'deit3_huge_patch14_224.fb_in22k_ft_in1k'
})