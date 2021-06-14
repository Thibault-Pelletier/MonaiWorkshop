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
from monai.apps import download_url, extractall
from monai.utils import first, set_determinism
from monai.transforms import (AsDiscrete, AddChanneld, Compose, CropForegroundd, LoadImaged,
                              Orientationd, RandCropByPosNegLabeld, Spacingd, ToTensord,
                              ScaleIntensityd, MapTransform)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import compute_meandice
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset
from monai.config import print_config
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report, confusion_matrix, )

print_config()

# Setup data directory
root_dir = "./"
data_dir = os.path.join(root_dir, "data")
res_dir = os.path.join(root_dir, "results")
os.makedirs(res_dir, exist_ok=True)
trained_model_path = os.path.join(res_dir, "net_key_metric=0.7314.pt")

if not os.path.exists(trained_model_path):
    resource = "https://github.com/Thibault-Pelletier/MonaiWorkshop/raw/e29cdaa46e0097db909478e95c001d303ae963ab/results/net_key_metric%3D0.7314.pt"
    download_url(url=resource, filepath=trained_model_path)

# Download data if necessary
resource = "https://drive.google.com/uc?id=1aMc9eW_fGCphGBjAKDedxu8-aJVcSczd"  # Full 2.9 GB dataset
# resource = "https://drive.google.com/uc?id=1rZwPR3CFlFmYTev2YkxTJDfmLrbCiy63"  # Small 200 MB dataset subset

compressed_file = os.path.join(root_dir, "brats2015 - data.zip")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    download_url(url=resource, filepath=compressed_file)
    extractall(filepath=compressed_file, output_dir=data_dir)

# Set dataset path
train_dirs = [os.path.join(data_dir, "train", "HGG"), os.path.join(data_dir, "train", "LGG")]
test_dir = os.path.join(data_dir, "test", "HGG_LGG")

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
# train_files, val_files = data_dicts[:10], data_dicts[-10:]

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
                                                   spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=4,  #
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

# Check transforms in DataLoader
check_ds = Dataset(data=val_files, transform=val_transforms)
check_loader = DataLoader(check_ds, batch_size=1)
check_data = first(check_loader)
image, label = (check_data["image"][0][0], check_data["label"][0][0])
print(f"image shape: {image.shape}, label shape: {label.shape}")

# plot the slice [:, :, 50]
i_slice = 50
plt.figure("check", (12, 6))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(image[:, :, i_slice], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("label")
plt.imshow(label[:, :, i_slice])
plt.show()

# DataLoader for training and validation+
# use batch_size=2 to load images and use RandCropByPosNegLabeld
# to generate 2 x 4 images for network training
# train_ds = CacheDataset(data=train_files, transform=train_transforms, num_workers=0)
train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

# val_ds = CacheDataset(data=val_files, transform=val_transforms, num_workers=0)
val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1)

# Create Model, Loss, Optimizer
# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = UNet(dimensions=3,  #
             in_channels=1,  #
             out_channels=1,  #
             channels=(16, 32, 64, 128, 256),  #
             strides=(2, 2, 2, 2),  #
             num_res_units=2, norm=Norm.BATCH, ).to(device)  #

loss_function = DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-3)

# Execute a typical PyTorch training process

max_epochs = 50
# max_epochs = 300
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
post_pred = AsDiscrete(threshold_values=True, n_classes=1)
post_label = AsDiscrete(n_classes=1)

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = (batch_data["image"].to(device), batch_data["label"].to(device),)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"{step}/{len(train_ds) // train_loader.batch_size}, "
              f"train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            metric_sum = 0.0
            metric_count = 0
            for val_data in val_loader:
                val_inputs, val_labels = (val_data["image"].to(device), val_data["label"].to(device),)
                roi_size = (96, 96, 96)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                val_outputs = post_pred(val_outputs)
                val_labels = post_label(val_labels)
                value = compute_meandice(y_pred=val_outputs, y=val_labels, include_background=False, )
                metric_count += len(value)
                metric_sum += value.sum().item()
            metric = metric_sum / metric_count
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(res_dir, "best_metric_model.pth"))
                print("saved new best metric model")
            print(f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                  f"\nbest mean dice: {best_metric:.4f} "
                  f"at epoch: {best_metric_epoch}")

print(f"train completed, best_metric: {best_metric:.4f} "
      f"at epoch: {best_metric_epoch}")

# Plot the loss and metric
plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.show()

# Check best model output with the input image and label for some of the validation data
# model.load_state_dict(torch.load(os.path.join(res_dir, "best_metric_model.pth")))
model.load_state_dict(torch.load(trained_model_path, map_location=device))
model.eval()

with torch.no_grad():
    for i, val_data in enumerate(val_loader):
        if i >= 10:
            break

        roi_size = (96, 96, 96)
        sw_batch_size = 4
        val_outputs = post_pred(sliding_window_inference(val_data["image"].to(device), roi_size, sw_batch_size, model))

        # plot the slice [:, :, 50]
        i_slice = 50
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title(f"image {i}")
        plt.imshow(val_data["image"][0, 0, :, :, i_slice], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title(f"label {i}")
        plt.imshow(val_data["label"][0, 0, :, :, i_slice])
        plt.subplot(1, 3, 3)
        plt.title(f"output {i}")
        plt.imshow(val_outputs.cpu().numpy()[0, 0, :, :, i_slice])
        plt.show()

# Check the confusion matrix on the validation data
cpu_device = torch.device("cpu")
y_pred = torch.tensor([], dtype=torch.float32, device=cpu_device)
y = torch.tensor([], dtype=torch.long, device=cpu_device)

with torch.no_grad():
    for i, val_data in enumerate(val_loader):
        if i >= 10:
            break

        roi_size = (96, 96, 96)

        sw_batch_size = 4
        outputs = sliding_window_inference(val_data["image"].to(device), roi_size, sw_batch_size, model)
        outputs = post_pred(outputs).cpu()
        labels = val_data["label"].cpu()

        y_pred = torch.cat([y_pred, outputs.flatten()], dim=0)
        y = torch.cat([y, labels.flatten()], dim=0)

print(classification_report(y.numpy(), y_pred.numpy(), target_names=["non-tumor", "tumor"]))

cm = confusion_matrix(y.numpy(), y_pred.numpy(), normalize="true", )
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["non-tumor", "tumor"], )
disp.plot(ax=plt.subplots(1, 1, facecolor="white")[1])
plt.show()
