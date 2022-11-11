import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.colour_dictionary import get_colour_list
from utils.pedestrain_dataset import pedestrain_dataset


warnings.filterwarnings("ignore", category=UserWarning) # to discard upsample deprecation


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("----No checkpoints at given path----")
        return
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")))
    print("----checkpoints loaded from path: {}----".format(checkpoint_path))
    return model


def name_colours(model,
                 images_list,
                 n_colours=2,
                 same_person=False,
                 max_images=50,
                 batch_size=1,
                 num_workers=0,
                 device="cuda",
                 show_image=False):
    """Get a list of colour names from a pedestrian image / or multiple images from the same person.

    The function uses a U2-netS model trained on the PCN dataset to segment the image into 11 colours
    + the background. The first n_colours in the mask (based on extension) are selected. If run on
    multiple images form the same person, the most frequently detected colours are selected.

    :param model:       pre-loaded U2-netS (also set to device and in eval mode)
    :param images_list: list of image paths (full path)
    :param n_colours:   maximum number of colours to get
    :param same_person: (bool) if images are from the same person (in this case,
                        the most frequent colours are chosen)
    :param max_images:  (int) max number of images to consider in case same_person=True.
                        If > 0, images_list is shuffled before picking the first max_images images.
    :param batch_size:  (int) batch size for the model, to speedup computation
    :param num_workers: (int) n workers for preprocessing the images
    :param device:      device name ("cuda", "cpu", ...)
    :param show_image:  (bool) show the output colour mask with the original image.

    :return: colour_names:   (list, str) list of detected colour names
    :return: colour_numbers: (list, int) list of detected colour indices

    2022 / Almaviva / d.mascali.esterno@almaviva.it
    """

    color_list, color_names = get_colour_list()
    color_list = torch.tensor(color_list)

    if not same_person:
        max_images = None

    dataset = pedestrain_dataset(images_list, show_mode=show_image, max_images=max_images)
    loader = DataLoader(dataset,
                        batch_size=batch_size, num_workers=num_workers,
                        shuffle=False, drop_last=False)

    colour_names = []
    colour_numbers = []

    print("Naming pedestrian cloths. Same person mode: " + str(same_person))

    for batch, img_size in loader:
        output_tensor = model(batch.to(device))
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = output_tensor.cpu().detach().squeeze(dim=1)

        for counter, img in enumerate(output_tensor):
            img_1d = img.view(-1)  # it's just one instance
            bins = torch.bincount(img_1d, minlength=12)
            non_zero = len(bins[bins > 0])  # for excluding indices of empty colours
            sorted_colour_numbers = torch.argsort(bins, descending=True)[0:non_zero]
            sorted_colour_numbers = sorted_colour_numbers[sorted_colour_numbers != 0]  # for excluding the background
            sorted_colour_numbers = sorted_colour_numbers[0:n_colours]
            colour_numbers.append(sorted_colour_numbers.tolist())
            if not same_person:
                # convert colour numbers to colour names
                colour_names.append([color_names[i] for i in sorted_colour_numbers.tolist()])
                if show_image:
                    # restore mean and std
                    orig_image = (batch[counter] + 1) * 0.5
                    orig_image = orig_image.permute((1, 2, 0)).numpy()
                    # take care of output mask
                    output_to_show = color_list[img, :]
                    output_to_show = output_to_show.numpy() / 255
                    to_show = np.hstack([orig_image, output_to_show])
                    plt.imshow(to_show)
                    plt.title(colour_names[-1])
                    plt.axis('off')
                    plt.show()

    if same_person:
        all_indices = np.array([j for i in colour_numbers for j in i])
        uniques, counts = np.unique(all_indices, return_counts=True)
        colour_numbers = uniques[np.argsort(counts)[::-1]][0:n_colours]
        colour_names = [color_names[i] for i in colour_numbers.tolist()]
        if show_image:
            # use the last one as example:
            # restore mean and std
            orig_image = (batch[counter] + 1) * 0.5
            orig_image = orig_image.permute((1, 2, 0)).numpy()
            plt.imshow(orig_image)
            plt.title(colour_names)
            plt.axis('off')
            plt.show()

    return colour_names, colour_numbers

