# Brain tumor 3D segmentation with MONAI The brain tumor dataset can be downloaded from
# https://www.kaggle.com/andrewmvd/brain-tumor-segmentation-in-mri-brats-2015/.
#
# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import logging
import os

import monai
import numpy as np
import torch
from ignite.metrics import Accuracy
from monai.config import print_config
from monai.data import CacheDataset, DataLoader
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.handlers import LrScheduleHandler, StatsHandler, CheckpointSaver, TensorBoardStatsHandler, \
    TensorBoardImageHandler, MeanDice, ValidationHandler
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.losses import DiceLoss
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (AddChanneld, Compose, CropForegroundd, LoadImaged,
                              Orientationd, RandCropByPosNegLabeld, Spacingd, ToTensord,
                              ScaleIntensityd, MapTransform, Activationsd, AsDiscreted, KeepLargestConnectedComponentd)
from monai.utils import set_determinism

logging.basicConfig(level=logging.INFO)

print_config()

# Setup data directory
root_dir = os.path.dirname(__file__)
data_dir = os.path.join(root_dir, "data")
train_dirs = [os.path.join(data_dir, "train", "HGG"), os.path.join(data_dir, "train", "LGG")]
test_dir = os.path.join(data_dir, "test", "HGG_LGG")
res_dir = os.path.join(root_dir, "results")
log_dir = os.path.join(res_dir, "runs")
os.makedirs(res_dir, exist_ok=True)

# Set dataset path
data_dicts = []
for train_dir in train_dirs:
    test_folders = os.listdir(train_dir)

    for test_folder in test_folders:
        test_folder_path = os.path.join(train_dir, test_folder)
        data_files = os.listdir(test_folder_path)
        mri_files = [f for f in data_files if "more" not in f]
        if len(mri_files) == len(data_files):
            continue

        label_file = [f for f in data_files if "more" in f][0]

        data_dicts.extend(
            [{"image": os.path.join(test_folder_path, mri_file), "label": os.path.join(test_folder_path, label_file)}
             for mri_file in mri_files])

print("Number of training data : ", len(data_dicts))
valid_split = 0.3
valid_n = int(valid_split * len(data_dicts))
train_files, val_files = data_dicts[:-valid_n], data_dicts[-valid_n:]

# Set deterministic training for reproducibility
set_determinism(seed=0)


# Setup transforms for training and validation
class FixLabelAffineAndReduceClassesToOne(MapTransform):
    """
    Custom transform to fixe saved label affine transform if necessary (value may not be aligned with the input image)
    """

    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        image_affine = d[f"image_meta_dict"]["affine"]

        try:
            meta_data = d[f"label_meta_dict"]
            label_affine = meta_data["affine"]
            if np.array_equal(label_affine, np.identity(4)):
                meta_data["affine"] = image_affine

        except KeyError:
            pass

        # Saturate label values greater than 1 to 1
        label_values = d["label"]
        d["label"] = (label_values > 0).astype(label_values.dtype)
        return d


train_transforms = Compose([LoadImaged(keys=["image", "label"]),  #
                            FixLabelAffineAndReduceClassesToOne(keys=["image", "label"]),  #
                            AddChanneld(keys=["image", "label"]),  #
                            Spacingd(keys=["image", "label"], pixdim=(1, 1, 1), mode=("bilinear", "nearest")),  #
                            Orientationd(keys=["image", "label"], axcodes="RAS"),  #
                            ScaleIntensityd(keys=["image"]),  #
                            CropForegroundd(keys=["image", "label"], source_key="image"),  #
                            RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",  #
                                                   spatial_size=(48, 48, 48), pos=1, neg=1, num_samples=4,  #
                                                   image_key="image", image_threshold=0, ),  #
                            ToTensord(keys=["image", "label"]),  #
                            ])

val_transforms = Compose([LoadImaged(keys=["image", "label"]),  #
                          FixLabelAffineAndReduceClassesToOne(keys=["image", "label"]),  #
                          AddChanneld(keys=["image", "label"]),  #
                          Spacingd(keys=["image", "label"], pixdim=(1, 1, 1), mode=("bilinear", "nearest")),  #
                          Orientationd(keys=["image", "label"], axcodes="RAS"),  #
                          ScaleIntensityd(keys=["image"]),  #
                          CropForegroundd(keys=["image", "label"], source_key="image"),  #
                          ToTensord(keys=["image", "label"]),  #
                          ])

# DataLoader for training and validation
# use batch_size=2 to load images and use RandCropByPosNegLabeld
# to generate 2 x 4 images for network training
train_ds = CacheDataset(data=train_files, transform=train_transforms, num_workers=0)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

val_ds = CacheDataset(data=val_files, transform=val_transforms, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=1)

# Create Model, Loss, Optimizer
# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
device = torch.device("cuda:0")
model = UNet(dimensions=3,  #
             in_channels=1,  #
             out_channels=1,  #
             channels=(16, 32, 64, 128, 256),  #
             strides=(2, 2, 2, 2),  #
             num_res_units=2, norm=Norm.BATCH, ).to(device)  #

loss_function = DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-3)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

val_post_transforms = Compose(
    [
        Activationsd(keys="pred", sigmoid=True),  #
        AsDiscreted(keys="pred", threshold_values=True),  #
        KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),  #
    ]
)

val_handlers = [
    StatsHandler(output_transform=lambda x: None),  #
    TensorBoardStatsHandler(log_dir=log_dir, output_transform=lambda x: None),  #
    TensorBoardImageHandler(  #
        log_dir=log_dir,  #
        batch_transform=lambda x: (x["image"], x["label"]),  #
        output_transform=lambda x: x["pred"],  #
    ),  #
    CheckpointSaver(save_dir=log_dir, save_dict={"net": model}, save_key_metric=True),  #
]

evaluator = SupervisedEvaluator(
    device=device,  #
    val_data_loader=val_loader,  #
    network=model,  #
    inferer=SlidingWindowInferer(roi_size=(48, 48, 48), sw_batch_size=4, overlap=0.5),  #
    post_transform=val_post_transforms,  #
    key_val_metric={  #
        "val_mean_dice": MeanDice(include_background=True, output_transform=lambda x: (x["pred"], x["label"]))  #
    },  #
    additional_metrics={"val_acc": Accuracy(output_transform=lambda x: (x["pred"], x["label"]))},  #
    val_handlers=val_handlers,  #
    # if no FP16 support in GPU or PyTorch version < 1.6, will not enable AMP evaluation
    amp=True if monai.utils.get_torch_version_tuple() >= (1, 6) else False,  #
)

train_post_transforms = Compose(
    [Activationsd(keys="pred", sigmoid=True),  #
     AsDiscreted(keys="pred", threshold_values=True),  #
     KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),  #
     ])

train_handlers = [LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),  #
                  ValidationHandler(validator=evaluator, interval=2, epoch_level=True),  #
                  StatsHandler(tag_name="train_loss", output_transform=lambda x: x["loss"]),  #
                  TensorBoardStatsHandler(log_dir=log_dir, output_transform=lambda x: None),
                  CheckpointSaver(save_dir=log_dir, save_dict={"net": model, "opt": optimizer}, save_interval=50,
                                  epoch_level=True),  #
                  ]

# Execute a training process using MONAI supervised trainer
trainer = SupervisedTrainer(device=device,  #
                            max_epochs=600,  #
                            train_data_loader=train_loader,  #
                            network=model,  #
                            optimizer=optimizer,  #
                            loss_function=loss_function,  #
                            inferer=SimpleInferer(),  #
                            # if no FP16 support in GPU or PyTorch version < 1.6, will not enable AMP evaluation
                            amp=True if monai.utils.get_torch_version_tuple() >= (1, 6) else False,  #
                            post_transform=train_post_transforms,  #
                            key_train_metric={"train_acc": Accuracy(output_transform=lambda x: (x["pred"], x["label"]),
                                                                    device=device)},  #
                            train_handlers=train_handlers,  #
                            )
trainer.run()
