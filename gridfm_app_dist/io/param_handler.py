import torch
from gridfm_app_dist.training.loss import MixedLoss
from gridfm_app_dist.io.registries import (
    NORMALIZERS_REGISTRY,
    MODELS_REGISTRY,
    LOSS_REGISTRY,
    TRANSFORM_REGISTRY,
    TASK_REGISTRY,
    PHYSICS_DECODER_REGISTRY,
)

import argparse
from torch_geometric.transforms import Compose
from gridfm_app_dist.tasks.base_task import BaseTask


class NestedNamespace(argparse.Namespace):
    """
    A namespace object that supports nested structures, allowing for
    easy access and manipulation of hierarchical configurations.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, NestedNamespace(**value))
            elif isinstance(value, list):
                list_of_namespaces = []
                for element in value:
                    if isinstance(element, dict):
                        list_of_namespaces.append(NestedNamespace(**element))
                    else:
                        list_of_namespaces.append(element)
                setattr(self, key, list_of_namespaces)
            else:
                setattr(self, key, value)

    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, NestedNamespace):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def flatten(self, parent_key="", sep="."):
        items = []
        for key, value in self.__dict__.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, NestedNamespace):
                items.extend(value.flatten(new_key, sep=sep).items())
            else:
                items.append((new_key, value))
        return dict(items)


def load_normalizer(args):
    """Load the appropriate data normalization method."""
    method = args.data.normalization
    try:
        return NORMALIZERS_REGISTRY.create(method, args)
    except KeyError:
        raise ValueError(f"Unknown transformation: {method}")


def get_loss_function(args):
    """Build a MixedLoss from the configured loss names / args / weights.
    For the VoltageLossDetection task, configure a single loss, e.g.:
        training:
          losses: ["VLDTopologyLoss"]
          loss_args:
            - input_state_threshold: 0.5
              prediction_threshold: 0.5
              topology_weight: 1.0
              target_anchor_weight: 0.25
              off_vm_weight: 1.0
              on_vm_weight: 0.5
              unreachable_l1_weight: 1.0
              topology_confidence_gamma: 10.0
          loss_weights: [1.0]
    """
    loss_functions = []
    for loss_name, loss_args in zip(args.training.losses, args.training.loss_args):
        try:
            loss_functions.append(LOSS_REGISTRY.create(loss_name, loss_args, args))
        except KeyError:
            raise ValueError(f"Unknown loss: {loss_name}")

    return MixedLoss(loss_functions=loss_functions, weights=args.training.loss_weights)


def load_model(args) -> torch.nn.Module:
    """Load the appropriate model."""
    model_type = args.model.type
    try:
        return MODELS_REGISTRY.create(model_type, args)
    except KeyError:
        raise ValueError(f"Unknown model type: {model_type}")


def get_task_transforms(args) -> Compose:
    """Load the task-specific transforms."""
    task_transforms = args.task.task_name
    try:
        return TRANSFORM_REGISTRY.create(task_transforms, args)
    except KeyError:
        raise ValueError(f"Unknown task: {task_transforms}")


def get_task(args, data_normalizers) -> BaseTask:
    """Load the task module."""
    task = args.task.task_name
    try:
        return TASK_REGISTRY.create(task, args, data_normalizers)
    except KeyError:
        raise ValueError(f"Unknown task: {task}")


def get_physics_decoder(args) -> torch.nn.Module:
    """Load the physics decoder associated with the task."""
    task = args.task.task_name
    try:
        return PHYSICS_DECODER_REGISTRY.create(task)
    except KeyError:
        raise ValueError(f"No physics decoder associate to {task} task")
