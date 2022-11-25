import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import ImageFile
# Image.File.LOAD_TRUNCATED_IMAGES = True


class WaterDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # self.list_files = os.listdir(self.root_dir)
        # Load Data
        train_path_watermarked_images = self.root_dir + '/train/watermark/'
        train_path_nonwatermarked_images = self.root_dir + '/train/no-watermark/'

        tp_watermarked = np.array([])  # array with watermarked image names
        # array with nonwatermarked image names
        tp_nonwatermarked = np.array([])

        # data length = 12510
        for root, dirs, files in os.walk(train_path_watermarked_images, topdown=True):
            for file in files:
                # append just the name of the file into array
                tp_watermarked = np.append(
                    tp_watermarked, WaterDataset.takeFileName(file))

        # data length = 12477
        for root, dirs, files in os.walk(train_path_nonwatermarked_images, topdown=True):
            for file in files:
                tp_nonwatermarked = np.append(tp_nonwatermarked, WaterDataset.takeFileName(
                    file))  # append just the name of the file into array

        self.tp_watermarked_sorted, self.tp_nonwatermarked_sorted = WaterDataset.matchFileNames(
            tp_watermarked, tp_nonwatermarked, train_path_watermarked_images, train_path_nonwatermarked_images)

        valid_path_watermarked_images = self.root_dir + '/valid/watermark/'
        valid_path_nonwatermarked_images = self.root_dir + '/valid/no-watermark/'

        vp_watermarked = np.array([])  # array with watermarked image names
        # array with nonwatermarked image names
        vp_nonwatermarked = np.array([])

        # data length = 3299
        for root, dirs, files in os.walk(valid_path_watermarked_images, topdown=True):
            for file in files:
                # append just the name of the file into array
                vp_watermarked = np.append(
                    vp_watermarked, WaterDataset.takeFileName(file))

        # data length = 3289
        for root, dirs, files in os.walk(valid_path_nonwatermarked_images, topdown=True):
            for file in files:
                vp_nonwatermarked = np.append(vp_nonwatermarked, WaterDataset.takeFileName(
                    file))  # append just the name of the file into array

        self.vp_watermarked_sorted, self.vp_nonwatermarked_sorted = WaterDataset.matchFileNames(
            vp_watermarked, vp_nonwatermarked, valid_path_watermarked_images, valid_path_nonwatermarked_images)

    def __len__(self):
        return len(self.tp_watermarked_sorted)

    def __getitem__(self, index):
        input_img_file = self.tp_watermarked_sorted[index]
        input_img_path = os.path.join(self.root_dir, input_img_file)
        input_image = np.array(Image.open(input_img_path).convert("RGB"))

        target_img_file = self.tp_nonwatermarked_sorted[index]
        target_img_path = os.path.join(self.root_dir, target_img_file)
        target_image = np.array(Image.open(target_img_path).convert("RGB"))

        # do augment
        augmentations = config.both_transform(
            image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image

    def takeFileName(filedir):  # remove just file name from directory and return
        # filename = np.array(filedir.split('/'))[-1].split('.')[0] # take out the name, isolate the jpeg, then return the name
        # take out the name, then return the name
        filename = np.array(filedir.split('/'))[-1]
        # print(filename)
        return filename

    def matchFileNames(watermarkedarr, nonwatermarkedarr, dname_wm, dname_nwm):
        sortedwmarr = np.array([])
        sortednwmarr = np.array([])

        wmarr = list(watermarkedarr)
        nwmarr = list(nonwatermarkedarr)

        length = len(watermarkedarr) if len(watermarkedarr) >= len(
            nonwatermarkedarr) else len(nonwatermarkedarr)

        for pos in range(length):
            try:
                if length == len(watermarkedarr):  # more images in watermarked array
                    exist_nwm = nwmarr.index(wmarr[pos])
                    # this is the iterable
                    sortedwmarr = np.append(
                        sortedwmarr, dname_wm + watermarkedarr[pos])
                    sortednwmarr = np.append(
                        sortednwmarr, dname_nwm + nonwatermarkedarr[exist_nwm])  # this is the match
                # more images in nonwatermarked array
                elif length == len(nonwatermarkedarr):
                    exist_wm = wmarr.index(nwmarr[pos])
                    sortedwmarr = np.append(
                        sortedwmarr, dname_wm + watermarkedarr[exist_wm])  # this is the match
                    # this is the iterable
                    sortednwmarr = np.append(
                        sortednwmarr, dname_nwm + nonwatermarkedarr[pos])
            except ValueError:
                continue
        return sortedwmarr, sortednwmarr


def test():
    model = WaterDataset('/Users/ma107/Downloads/wm-nowm')
    print(model.__getitem__(3))


if __name__ == "__main__":
    test()
