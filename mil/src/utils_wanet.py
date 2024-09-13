import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

def get_dataset_denormalization(mean, std):
   
    if mean.__len__() == 1:
        mean = - mean
    else:  # len > 1
        mean = [-i for i in mean]

    if std.__len__() == 1:
        std = 1 / std
    else:  # len > 1
        std = [1 / i for i in std]

    # copy from answer in
    # https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
    # user: https://discuss.pytorch.org/u/svd3

    invTrans = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.],
                             std=std),
        transforms.Normalize(mean=mean,
                             std=[1., 1., 1.]),
    ])

    # return A.Compose(
    #         [
    #             A.Normalize(mean = [0., 0., 0.], std=std, max_pixel_value=255.0, always_apply=True),
    #             A.Normalize(mean = mean, std=[1., 1., 1.], max_pixel_value=255.0, always_apply=True),
    #         ]
    #     )

    return invTrans