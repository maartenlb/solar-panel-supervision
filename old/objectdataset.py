import torch
from torch.utils.data import Dataset
import json
from PIL import Image


class SolarObject(Dataset):
    """
    Torchloader dataset of solar panels which has bounding boxes
    """

    def __init__(self, data_folder, data_file):
        super().__init__()
        with open(data_file) as f:
            data_dict = json.load(f)

        self.data_file = data_file
        self.data_folder = data_folder

        self.images = []
        self.objects = []
        self.labels = []

        for image in data_dict:
            self.images.append(image)

            bbox = data_dict[image]["bounding_boxes"]

            if data_dict[image]["solar_panel"]:
                new_list = []
                for i in range(data_dict[image]["solar_panel_count"]):
                    new_list.append(1)
                self.labels.append([new_list])
            else:
                self.labels.append([0])

            self.objects.append(bbox)

        assert len(self.images) == len(self.objects) == len(self.labels)

    def __getitem__(self, index):
        image = Image.open(self.data_folder + self.images[index])

        bounding_boxes = torch.FloatTensor(self.objects[index])
        labels = torch.LongTensor(self.labels[index])
        print(labels.shape)

        # TODO: transformation images

        return image, bounding_boxes, labels

    def __len__(self):
        return len(self.images)


s = SolarObject("data/processed_imgs/", "data/image_polygons.json")
i = s.__getitem__(2)
print(i[2].item())
