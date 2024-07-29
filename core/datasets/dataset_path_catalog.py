import os
from core.datasets.vaihingen import vaihingenDataSet
from core.datasets.vaihingen_selftraining import vaihingenSelfTrainingDataSet
from core.datasets.potsdam_selftraining import potsdamSelfTrainingDataSet
from core.datasets.potsdam import potsdamDataSet


class DatasetCatalog(object):
    DATASET_DIR = "datasets"
    DATASETS = {
        "vaihingen_train": {
            "data_dir": "vaihingen",
            "data_list": "vaihingen_train_list.txt"
        },
        "vaihingen_self_train": {
            "data_dir": "vaihingen",
            "data_list": "vaihingen_train_list.txt",
            "label_dir": "vaihingen/soft_labels/inference/vaihingen_train"
        },
        "vaihingen_val": {
            "data_dir": "vaihingen",
            "data_list": "vaihingen_val_list.txt"
        },
        "potsdam_train": {
            "data_dir": "potsdam",
            "data_list": "potsdam_train_list.txt"
        },
        "potsdam_self_train": {
            "data_dir": "potsdam",
            "data_list": "potsdam_train_list.txt",
            "label_dir": "potsdam/soft_labels/inference/potsdam_train"
        },
        "potsdam_val": {
            "data_dir": "potsdam",
            "data_list": "potsdam_val_list.txt"
        },
    }

    @staticmethod
    def get(name, mode, num_classes, max_iters=None, transform=None):
        if "potsdam" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            if 'self' in name:
                args['label_dir'] = os.path.join(data_dir, attrs["label_dir"])
                return potsdamSelfTrainingDataSet(args["root"], args["data_list"], args['label_dir'], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
            return potsdamDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
        elif "vaihingen" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            if 'self' in name:
                args['label_dir'] = os.path.join(data_dir, attrs["label_dir"])
                return vaihingenSelfTrainingDataSet(args["root"], args["data_list"], args['label_dir'], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
            return vaihingenDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)

        raise RuntimeError("Dataset not available: {}".format(name))