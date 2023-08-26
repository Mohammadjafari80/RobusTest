import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import faiss
import torchvision.models as models
import torch.nn.functional as F
from PIL import ImageFilter
import random
from torchvision.transforms import InterpolationMode

BICUBIC = InterpolationMode.BICUBIC
from torchvision.datasets import CIFAR10, MNIST, SVHN, FashionMNIST, CIFAR100
import os
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
from robustness import model_utils
from torchvision import models
from robustness.datasets import ImageNet
import requests
import subprocess


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


transform_color = transforms.Compose(
    [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
)

transform_resnet18 = transforms.Compose(
    [
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)


transform_color_bw = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]
)

transform_resnet18_bw = transforms.Compose(
    [
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]
)


moco_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

moco_transform_bw = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

mu = torch.tensor(mean).view(3, 1, 1).cuda()
std = torch.tensor(std).view(3, 1, 1).cuda()


class Transform:
    def __init__(self, bw=False):
        self.moco_transform = moco_transform_bw if bw else moco_transform

    def __call__(self, x):
        x_1 = self.moco_transform(x)
        x_2 = self.moco_transform(x)
        return x_1, x_2


class Model(torch.nn.Module):
    def __init__(self, backbone="18", path="./pretrained_models/"):
        super().__init__()
        self.norm = lambda x: (x - mu) / std
        if backbone == "152":
            self.backbone = models.resnet152(pretrained=True)
        elif backbone == "50":
            self.backbone = models.resnet50(pretrained=True)
        elif backbone == "18":
            self.backbone = models.resnet18(pretrained=True)
        else:
            self.backbone = RobustModel(path=path, arch=backbone)

        self.backbone.fc = torch.nn.Identity()
        freeze_parameters(self.backbone, backbone, train_fc=False)

    def forward(self, x):
        x = self.norm(x)
        z1 = self.backbone(x)
        z_n = F.normalize(z1, dim=-1)
        return z_n


robust_urls = {
    "resnet18_linf_eps0.5": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet18_linf_eps0.5.ckpt",
    "resnet18_linf_eps1.0": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet18_linf_eps1.0.ckpt",
    "resnet18_linf_eps2.0": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet18_linf_eps2.0.ckpt",
    "resnet18_linf_eps4.0": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet18_linf_eps4.0.ckpt",
    "resnet18_linf_eps8.0": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet18_linf_eps8.0.ckpt",
    "resnet50_linf_eps0.5": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_linf_eps0.5.ckpt",
    "resnet50_linf_eps1.0": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_linf_eps1.0.ckpt",
    "resnet50_linf_eps2.0": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_linf_eps2.0.ckpt",
    "resnet50_linf_eps4.0": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_linf_eps4.0.ckpt",
    "resnet50_linf_eps8.0": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_linf_eps8.0.ckpt",
    "wide_resnet50_2_linf_eps0.5": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/wide_resnet50_2_linf_eps0.5.ckpt",
    "wide_resnet50_2_linf_eps1.0": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/wide_resnet50_2_linf_eps1.0.ckpt",
    "wide_resnet50_2_linf_eps2.0": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/wide_resnet50_2_linf_eps2.0.ckpt",
    "wide_resnet50_2_linf_eps4.0": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/wide_resnet50_2_linf_eps4.0.ckpt",
    "wide_resnet50_2_linf_eps8.0": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/wide_resnet50_2_linf_eps8.0.ckpt",
}

class RobustModel(torch.nn.Module):
    def __init__(self, arch="resnet50_linf_eps2.0", path="./pretrained_models/"):
        super().__init__()
        path = download_and_load_backnone(robust_urls[arch], arch, path)
        self.model, _ = resume_finetuning_from_checkpoint(
            path, "_".join(arch.split("_")[:-2])
        )
        self.model = self.model.model

    def forward(self, x):
        return self.model(x)


def resume_finetuning_from_checkpoint(finetuned_model_path, arch):
    """Given arguments, dataset object and a finetuned model_path, returns a model
    with loaded weights and returns the checkpoint necessary for resuming training.
    """
    print("[Resuming finetuning from a checkpoint...]")
    model, checkpoint = model_utils.make_and_restore_model(
        arch=arch, dataset=ImageNet("/imagenet/"), resume_path=finetuned_model_path
    )
    return model, checkpoint


def download_and_load_backnone(url, model_name, path):
    arch = "_".join(model_name.split("_")[:-2])
    print(f"{arch}, {model_name}")
    os.makedirs(path, exist_ok=True)
    ckpt_path = os.path.join(path, f"{model_name}.ckpt")

    # Check if checkpoint file already exists
    if os.path.exists(ckpt_path):
        print(f"{model_name} checkpoint file already exists.")
        return ckpt_path

    r = requests.get(url, allow_redirects=True)  # to get content after redirection
    ckpt_url = r.url
    with open(ckpt_path, "wb") as f:
        f.write(r.content)

    return ckpt_path


def freeze_parameters(model, backbone, train_fc=False):
    if not train_fc:
        for p in model.fc.parameters():
            p.requires_grad = False
    if backbone == 152:
        for p in model.conv1.parameters():
            p.requires_grad = False
        for p in model.bn1.parameters():
            p.requires_grad = False
        for p in model.layer1.parameters():
            p.requires_grad = False
        for p in model.layer2.parameters():
            p.requires_grad = False


def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)


def get_loaders(
    source_dataset,
    target_datset,
    label_class,
    batch_size,
    source_path,
    target_path,
    backbone,
    test_type="ad",
):
    trainset, trainset_msad, normal_labels = get_train_dataset(
        source_dataset, test_type, label_class, source_path, backbone
    )

    print(
        f"Train Dataset: {source_dataset}, Normal Classes: {normal_labels}, length Trainset: {len(trainset)}"
    )

    testsets = []

    if test_type == "ad" or test_type == "osr":
        testsets.append(
            get_test_dataset(source_dataset, normal_labels, source_path, backbone)
        )
    else:
        testsets.append(
            get_test_dataset(
                source_dataset, [i for i in range(20)], source_path, backbone
            )
        )
        testsets.append(get_test_dataset(target_datset, [], target_path, backbone))

    testset = torch.utils.data.ConcatDataset(testsets)

    print(f"length Testset: {len(testset)}")

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    train_loader_msad = torch.utils.data.DataLoader(
        trainset_msad, batch_size=batch_size, shuffle=True, num_workers=2
    )

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader, train_loader_msad


def get_train_dataset(dataset, test_type, label_class, path, backbone):
    if dataset == "cifar10":
        return get_CIFAR10_train(test_type, label_class, path, backbone)
    elif dataset == "cifar100":
        return get_CIFAR100_train(test_type, label_class, path, backbone)
    elif dataset == "mnist":
        return get_MNIST_train(test_type, label_class, path, backbone)
    elif dataset == "fashion":
        return get_FASHION_MNIST_train(test_type, label_class, path, backbone)
    elif dataset == "svhn":
        return get_SVHN_train(test_type, label_class, path, backbone)
    elif dataset == "mvtec":
        return get_MVTEC_train(label_class, path, backbone)
    elif dataset == "mri":
        return get_BrainMRI_train()
    else:
        raise Exception("Source Dataset is not supported yet. ")
        exit()


def get_test_dataset(dataset, normal_labels, path, backbone):
    if dataset == "cifar10":
        return get_CIFAR10_test(normal_labels, path, backbone)
    elif dataset == "cifar100":
        return get_CIFAR100_test(normal_labels, path, backbone)
    elif dataset == "mnist":
        return get_MNIST_test(normal_labels, path, backbone)
    elif dataset == "fashion":
        return get_FASHION_MNIST_test(normal_labels, path, backbone)
    elif dataset == "svhn":
        return get_SVHN_test(normal_labels, path, backbone)
    elif dataset == "mvtec":
        return get_MVTEC_test(normal_labels, path, backbone)
    elif dataset == "mri":
        return get_BrainMRI_test()
    else:
        raise Exception("Target Dataset is not supported yet. ")
        exit()


mvtec_labels = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def get_CIFAR10_train(test_type, normal_class_indx, path, backbone):
    transform = transform_color if backbone == 152 else transform_resnet18

    trainset = CIFAR10(root=path, train=True, download=True, transform=transform)

    normal_class_labels = []

    if test_type == "ad":
        normal_class_labels = [normal_class_indx]
    elif test_type == "osr":
        unique_labels = np.unique(trainset.targets)
        np.random.shuffle(unique_labels)
        n_normal = 6
        normal_class_labels = unique_labels[:n_normal]
    else:
        unique_labels = np.unique(trainset.targets)
        normal_class_labels = unique_labels

    normal_mask = np.isin(trainset.targets, normal_class_labels)

    trainset.data = trainset.data[normal_mask]
    trainset.targets = [0 for t in trainset.targets]

    trainset_msad = CIFAR10(root=path, train=True, download=True, transform=Transform())
    trainset_msad.data = trainset_msad.data[normal_mask]
    trainset_msad.targets = [0 for t in trainset_msad.targets]

    return trainset, trainset_msad, normal_class_labels


def get_CIFAR10_test(normal_class_labels, path, backbone):
    transform = transform_color if backbone == 152 else transform_resnet18

    testset = CIFAR10(root=path, train=False, download=True, transform=transform)
    test_mask = np.isin(testset.targets, normal_class_labels)

    testset.targets = np.array(testset.targets)
    testset.targets[test_mask] = 0
    testset.targets[~test_mask] = 1

    return testset


def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array(
        [
            4,
            1,
            14,
            8,
            0,
            6,
            7,
            7,
            18,
            3,
            3,
            14,
            9,
            18,
            7,
            11,
            3,
            9,
            7,
            11,
            6,
            11,
            5,
            10,
            7,
            6,
            13,
            15,
            3,
            15,
            0,
            11,
            1,
            10,
            12,
            14,
            16,
            9,
            11,
            5,
            5,
            19,
            8,
            8,
            15,
            13,
            14,
            17,
            18,
            10,
            16,
            4,
            17,
            4,
            2,
            0,
            17,
            4,
            18,
            17,
            10,
            3,
            2,
            12,
            12,
            16,
            12,
            1,
            9,
            19,
            2,
            10,
            0,
            1,
            16,
            12,
            9,
            13,
            15,
            13,
            16,
            19,
            2,
            4,
            6,
            19,
            5,
            5,
            8,
            19,
            18,
            1,
            2,
            15,
            6,
            0,
            17,
            8,
            14,
            13,
        ]
    )
    return coarse_labels[targets]


def get_CIFAR100_train(test_type, normal_class_indx, path, backbone):
    transform = transform_color if backbone == 152 else transform_resnet18

    trainset = CIFAR100(root=path, train=True, download=True, transform=transform)
    trainset.targets = sparse2coarse(trainset.targets)

    normal_class_labels = []

    if test_type == "ad":
        normal_class_labels = [normal_class_indx]
    elif test_type == "osr":
        unique_labels = np.unique(trainset.targets)
        np.random.shuffle(unique_labels)
        n_normal = 12
        normal_class_labels = unique_labels[:n_normal]
    else:
        unique_labels = np.unique(trainset.targets)
        normal_class_labels = unique_labels

    normal_mask = np.isin(trainset.targets, normal_class_labels)

    trainset.data = trainset.data[normal_mask]
    trainset.targets = [0 for t in trainset.targets]

    trainset_msad = CIFAR100(
        root=path, train=True, download=True, transform=Transform()
    )
    trainset_msad.targets = sparse2coarse(trainset_msad.targets)
    trainset_msad.data = trainset_msad.data[normal_mask]
    trainset_msad.targets = [0 for t in trainset_msad.targets]

    return trainset, trainset_msad, normal_class_labels


def get_CIFAR100_test(normal_class_labels, path, backbone):
    transform = transform_color if backbone == 152 else transform_resnet18

    testset = CIFAR100(root=path, train=False, download=True, transform=transform)
    testset.targets = sparse2coarse(testset.targets)

    test_mask = np.isin(testset.targets, normal_class_labels)

    testset.targets = np.array(testset.targets)
    testset.targets[test_mask] = 0
    testset.targets[~test_mask] = 1

    return testset


def get_MNIST_train(test_type, normal_class_indx, path, backbone):
    transform = transform_color_bw if backbone == 152 else transform_resnet18_bw

    trainset = MNIST(root=path, train=True, download=True, transform=transform)
    normal_class_labels = []

    if test_type == "ad":
        normal_class_labels = [normal_class_indx]
    elif test_type == "osr":
        unique_labels = np.unique(trainset.targets)
        np.random.shuffle(unique_labels)
        n_normal = 6
        normal_class_labels = unique_labels[:n_normal]
    else:
        unique_labels = np.unique(trainset.targets)
        normal_class_labels = unique_labels

    normal_mask = np.isin(trainset.targets, normal_class_labels)

    trainset.data = trainset.data[normal_mask]
    trainset.targets = [0 for t in trainset.targets]

    trainset_msad = MNIST(
        root=path, train=True, download=True, transform=Transform(bw=True)
    )
    trainset_msad.data = trainset_msad.data[normal_mask]
    trainset_msad.targets = [0 for t in trainset_msad.targets]

    return trainset, trainset_msad, normal_class_labels


def get_MNIST_test(normal_class_labels, path, backbone):
    transform = transform_color_bw if backbone == 152 else transform_resnet18_bw

    testset = MNIST(root=path, train=False, download=True, transform=transform)
    test_mask = np.isin(testset.targets, normal_class_labels)

    testset.targets = np.array(testset.targets)
    testset.targets[test_mask] = 0
    testset.targets[~test_mask] = 1

    return testset


def get_FASHION_MNIST_train(test_type, normal_class_indx, path, backbone):
    transform = transform_color_bw if backbone == 152 else transform_resnet18_bw

    trainset = FashionMNIST(root=path, train=True, download=True, transform=transform)

    normal_class_labels = []

    if test_type == "ad":
        normal_class_labels = [normal_class_indx]
    elif test_type == "osr":
        unique_labels = np.unique(trainset.targets)
        np.random.shuffle(unique_labels)
        n_normal = 6
        normal_class_labels = unique_labels[:n_normal]
    else:
        unique_labels = np.unique(trainset.targets)
        normal_class_labels = unique_labels

    normal_mask = np.isin(trainset.targets, normal_class_labels)

    trainset.data = trainset.data[normal_mask]
    trainset.targets = [0 for t in trainset.targets]

    trainset_msad = MNIST(
        root=path, train=True, download=True, transform=Transform(bw=True)
    )
    trainset_msad.data = trainset_msad.data[normal_mask]
    trainset_msad.targets = [0 for t in trainset_msad.targets]

    return trainset, trainset_msad, normal_class_labels


def get_FASHION_MNIST_test(normal_class_labels, path, backbone):
    transform = transform_color_bw if backbone == 152 else transform_resnet18_bw

    testset = FashionMNIST(root=path, train=False, download=True, transform=transform)

    test_mask = np.isin(testset.targets, normal_class_labels)

    testset.targets = np.array(testset.targets)
    testset.targets[test_mask] = 0
    testset.targets[~test_mask] = 1

    return testset


def get_SVHN_train(test_type, normal_class_indx, path, backbone):
    transform = transform_color if backbone == 152 else transform_resnet18

    trainset = SVHN(root=path, split="train", download=True, transform=transform)

    normal_class_labels = []

    if test_type == "ad":
        normal_class_labels = [normal_class_indx]
    elif test_type == "osr":
        unique_labels = np.unique(trainset.labels)
        np.random.shuffle(unique_labels)
        n_normal = 6
        normal_class_labels = unique_labels[:n_normal]
    else:
        unique_labels = np.unique(trainset.labels)
        normal_class_labels = unique_labels

    normal_mask = np.isin(trainset.labels, normal_class_labels)

    trainset.data = trainset.data[normal_mask]
    trainset.labels = [0 for t in trainset.labels]

    trainset_msad = SVHN(root=path, split="train", download=True, transform=Transform())
    trainset_msad.data = trainset_msad.data[normal_mask]
    trainset_msad.labels = [0 for t in trainset_msad.labels]

    return trainset, trainset_msad, normal_class_labels


def get_SVHN_test(normal_class_labels, path, backbone):
    transform = transform_color if backbone == 152 else transform_resnet18

    testset = SVHN(root=path, split="test", download=True, transform=transform)
    test_mask = np.isin(testset.labels, normal_class_labels)

    testset.labels = np.array(testset.labels)
    testset.labels[test_mask] = 0
    testset.labels[~test_mask] = 1

    return testset


from tqdm import tqdm


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        category,
        transform=None,
        target_transform=None,
        train=True,
        normal=True,
        download=False,
    ):
        self.transform = transform

        # Check if dataset directory exists
        dataset_dir = os.path.join(root, "mvtec_anomaly_detection")
        if not os.path.exists(dataset_dir):
            if download:
                self.download_dataset(root)
            else:
                raise ValueError(
                    "Dataset not found. Please set download=True to download the dataset."
                )

        if train:
            self.data = glob(
                os.path.join(dataset_dir, category, "train", "good", "*.png")
            )

        else:
            image_files = glob(
                os.path.join(dataset_dir, category, "test", "*", "*.png")
            )
            normal_image_files = glob(
                os.path.join(dataset_dir, category, "test", "good", "*.png")
            )
            anomaly_image_files = list(set(image_files) - set(normal_image_files))
            self.data = image_files

        self.data.sort(key=lambda y: y.lower())
        self.train = train

    def __getitem__(self, index):
        image_file = self.data[index]
        image = Image.open(image_file).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if os.path.dirname(image_file).endswith("good"):
            target = 0
        else:
            target = 1

        return image, target

    def __len__(self):
        return len(self.data)

    def download_dataset(self, root):
        url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"
        dataset_dir = os.path.join(root, "mvtec_anomaly_detection")

        # Create directory for dataset
        os.makedirs(dataset_dir, exist_ok=True)

        # Download and extract dataset
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024

        desc = "\033[33mDownloading MVTEC...\033[0m"
        progress_bar = tqdm(
            total=total_size_in_bytes,
            unit="iB",
            unit_scale=True,
            desc=desc,
            position=0,
            leave=True,
        )

        with open(os.path.join(root, "mvtec_anomaly_detection.tar.xz"), "wb") as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)

        progress_bar.close()

        tar_command = [
            "tar",
            "-xf",
            os.path.join(root, "mvtec_anomaly_detection.tar.xz"),
            "-C",
            dataset_dir,
        ]
        subprocess.run(tar_command)


def get_MVTEC_train(normal_class_indx, path, backbone):
    normal_class = mvtec_labels[normal_class_indx]
    transform = transform_color if backbone == 152 else transform_resnet18

    trainset = MVTecDataset(path, normal_class, transform, train=True, download=True)

    trainset_msad = MVTecDataset(path, normal_class, Transform(), train=True)

    return trainset, trainset_msad, normal_class_indx


def get_MVTEC_test(normal_class_indx, path, backbone):
    normal_class = mvtec_labels[normal_class_indx]
    transform = transform_color if backbone == 152 else transform_resnet18

    testset = MVTecDataset(path, normal_class, transform, train=False)

    return testset


import zipfile
import shutil


class BrainMRI(torch.utils.data.Dataset):
    def __init__(
        self,
        transform=None,
        target_transform=None,
        train=True,
        normal=True,
        normal_only=False,
    ):
        self._download_and_extract()
        self.transform = transform
        if train:
            self.image_files = glob(
                os.path.join("./MRI", "./Training", "notumor", "*.jpg")
            )
        else:
            image_files = glob(os.path.join("./MRI", "./Testing", "*", "*.jpg"))
            normal_image_files = glob(
                os.path.join("./MRI", "./Testing", "notumor", "*.jpg")
            )
            anomaly_image_files = list(set(image_files) - set(normal_image_files))
            self.image_files = image_files

        self.image_files.sort(key=lambda y: y.lower())
        self.train = train

    def _download_and_extract(self):
        google_id = "1AOPOfQ05aSrr2RkILipGmEkgLDrZCKz_"
        file_path = os.path.join("./MRI", "Training")

        if os.path.exists(file_path):
            return

        if not os.path.exists("./MRI"):
            os.makedirs("./MRI")

        if not os.path.exists(file_path):
            subprocess.run(["gdown", google_id, "-O", "./MRI/archive(3).zip"])

        with zipfile.ZipFile("./MRI/archive(3).zip", "r") as zip_ref:
            zip_ref.extractall("./MRI/")

        os.rename("./MRI/Training/glioma", "./MRI/Training/glioma_tr")
        os.rename("./MRI/Training/meningioma", "./MRI/Training/meningioma_tr")
        os.rename("./MRI/Training/pituitary", "./MRI/Training/pituitary_tr")

        shutil.move("./MRI/Training/glioma_tr", "./MRI/Testing")
        shutil.move("./MRI/Training/meningioma_tr", "./MRI/Testing")
        shutil.move("./MRI/Training/pituitary_tr", "./MRI/Testing")

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if os.path.dirname(image_file).endswith("notumor"):
            target = 0
        else:
            target = 1

        return image, target

    def __len__(self):
        return len(self.image_files)


def get_BrainMRI_train():
    transform_aug_mri = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
        ]
    )

    trainset = BrainMRI(transform_aug_mri, train=True)

    trainset_msad = BrainMRI(Transform(), train=True)

    return trainset, trainset_msad, None


def get_BrainMRI_test():
    transform_mri = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=3),
        ]
    )

    testset = BrainMRI(transform_mri, train=False)

    return testset
