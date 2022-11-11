from name_ped_clothes_colours import *
from networks.u2net import U2NETS


def main():
    # --------------------Set options------------------------------------------
    device = 'cuda'
    batch_size = 10
    num_workers = 2
    n_colours = 2  # maximum number of colours to name
    same_person = False  # if images are all from the same pedestrian
    max_images = 100   # Only if images are from the same pedestrian
    show_image = True  # show the output for each image
    checkpoint_path = "checkpoints/u2netS_on_PCN_itr_11200.pth"
    # define image list:
    image_dir = 'example_images_independent'
    images_list = sorted(os.listdir(image_dir))
    images_list = [os.path.join(image_dir, i) for i in images_list]
    # ---------------------------------------------------------------------------

    # load model:
    n_classes = 12
    net = U2NETS(in_ch=3, out_ch=n_classes)
    net = load_checkpoint(net, checkpoint_path)
    net = net.to(device).eval()

    output_colors = name_colours(net, images_list,
                                 n_colours=n_colours,
                                 same_person=same_person,
                                 max_images=max_images,
                                 batch_size=batch_size, num_workers=num_workers, device=device,
                                 show_image=show_image)


if __name__ == "__main__":
    main()