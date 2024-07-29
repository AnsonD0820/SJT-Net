import logging
import torch
from .feature_extractor import Feature_extractor_ours_mit
from .discriminator import *
from core.models.Classifier_Moudle import classifier_ours


def build_feature_extractor(cfg):
    model_name, backbone_name = cfg.MODEL.NAME.split('_')
    if backbone_name.startswith('mit'):
        backbone = Feature_extractor_ours_mit()
    else:
        raise NotImplementedError
    return backbone


def build_classifier(cfg):
    _, backbone_name = cfg.MODEL.NAME.split('_')
    if backbone_name.startswith('mit'):
        classifier = classifier_ours.Classifier(n_classes=cfg.MODEL.NUM_CLASSES)
    else:
        raise NotImplementedError
    return classifier


def build_adversarial_discriminator(cfg, num_features=None, mid_nc=256):
    _, backbone_name = cfg.MODEL.NAME.split('_')
    if backbone_name.startswith('mit'):
        if num_features is None:
            num_features = 2048
        model_D = PixelDiscriminator(num_features, mid_nc, num_classes=cfg.MODEL.NUM_CLASSES)
    else:
        raise NotImplementedError
    return model_D