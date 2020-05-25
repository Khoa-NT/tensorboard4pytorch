# https://pytorch.org/docs/stable/tensorboard.html

# https://discuss.pytorch.org/t/tensorboard-image-quality-lower-than-matplotlib/58233

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch

from tensorboard import notebook
from PIL import Image
# from utils.img_aug_func import array2rgb

class Logger(object):

    def __init__(self, log_dir=None, comment='', max_queue=50):
        """
        Create a summary writer logging to log_dir
        :param log_dir: Place for saving tensorboard log. If log_dir=None:
                        Writer will output to ./runs/ directory by default
        :param comment: Only work if log_dir=None. Add comment in the name of tensorboard file.
        :param max_queue: How many item show in the tensorboard
        """
        self.writer = SummaryWriter(log_dir=log_dir, comment=comment, max_queue=max_queue)

    def scalar_summary(self, tag, value, step):
        """
        Log a scalar variable
        :param tag: Name of the scalar tab in tensorboard
        :param value: Input scalar value
        :param step: Add scalr value at step
        :return:
        """
        self.writer.add_scalar(tag,value,step)
        self.writer.flush()

    def image_summary(self, tag, images, step, dataformats='HWC'):
        """
        Log a images
        :param tag: Name of the image tab in tensorboard
        :param images: Input image
        :param step:
        :param dataformats: "CHW", "HWC", "HW"
        :return:
        """
        if isinstance(images, torch.Tensor):
            self.writer.add_image(tag, images.type(torch.uint8), step, dataformats=dataformats)
        else:
            self.writer.add_image(tag, images.astype(np.uint8), step, dataformats=dataformats)
        self.writer.flush()

    def batch_image_summary(self, tag, input_list_batch_imgs, step, dataformats='NHWC'):
        """
        Log a list batch images .
        :param tag:
        :param input_list_batch_imgs: list of image has shape follows dataformats
        :param step:
        :param dataformats: NCHW or NHWC
        :return:
        """
        if not isinstance(input_list_batch_imgs, list):
            input_list_batch_imgs = [input_list_batch_imgs]
        list_images=[]

        for i in range(len(input_list_batch_imgs)):
            img = input_list_batch_imgs[i]

            # Convert item in input_list_batch_imgs to numpy array shape = [N, H, W, C]
            if isinstance(img, torch.Tensor):
                if dataformats=="NCHW":
                    if (img.is_cuda):
                        img = img.cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
                    else:
                        img = img.permute(0, 2, 3, 1).numpy().astype(np.uint8)
                elif dataformats=="NHWC":
                    if (img.is_cuda):
                        img = img.cpu().numpy().astype(np.uint8)
                    else:
                        img = img.numpy().astype(np.uint8)
                else:
                    raise ValueError("Input not match dataformats: NCHW or NHWC")
            elif isinstance(img, np.ndarray):
                if dataformats == "NCHW":
                    img = np.transpose(img,(0,2,3,1)).astype(np.uint8)
                elif dataformats == "NHWC":
                    img = img.astype(np.uint8)
                else:
                    raise ValueError("Input not match dataformats: NCHW or NHWC")
            else:
                raise ValueError("Input not match datatype: torch.Tensor or np.ndarray")

            img = array2rgb(img, dataformats="NHWC") # return img.shape = [N,H,W,3]

            list_images.append(img)

        concated_imgs = np.concatenate(list_images, axis=2).astype (np.uint8)
        log_imgs = [concated_imgs[i] for i in range(len(concated_imgs))] # HWC
        log_imgs = np.concatenate(log_imgs, axis=0).astype (np.uint8)
        self.image_summary(tag, log_imgs, step, dataformats='HWC')


def array2rgb(img, dataformats='HW'):
    """
    Convert an array to rbg image for visualization
    :param img: Input array image
    :param dataformats:
    :return:
    """
    if dataformats == 'HW':
        assert (len(img.shape) == 2), "Input must has shape [H, W]"
        if np.max(img) <= 1:
            img = img * 255
        ret = np.repeat(np.expand_dims(img, -1), 3, -1)
        ret = ret.astype(np.uint8)
    elif dataformats == 'HWC':
        assert (len(img.shape) == 3), "Input must has shape HWC"
        if img.shape[-1] == 1:
            ret = np.repeat(img, 3, -1)
            ret = ret.astype(np.uint8)
        elif img.shape[-1] == 3:  # Do nothing with RBG input
            ret = img.astype(np.uint8)
    elif dataformats == 'NHWC':
        assert (len(img.shape) == 4), "Input must has shape NHWC"
        if img.shape[-1] == 1:
            if np.max(img) <= 1:
                img = img * 255
            ret = np.repeat(img, 3, -1)
            ret = ret.astype(np.uint8)
        elif img.shape[-1] == 3:  # Do nothing with RBG input
            ret = img.astype(np.uint8)
        else:
            raise ValueError("Input doesn't have correct channel size")
    else:
        raise ValueError("dataformats is not correct: HW or HWC or NHWC")
    return ret