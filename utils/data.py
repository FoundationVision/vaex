import PIL.Image as PImage
from PIL import ImageFile
from torchvision.transforms import InterpolationMode, transforms

PImage.MAX_IMAGE_PIXELS = (1024 * 1024 * 1024 // 4 // 3) * 5
ImageFile.LOAD_TRUNCATED_IMAGES = False


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def pil_load(path: str, proposal_size):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f)
        w: int = img.width
        h: int = img.height
        sh: int = min(h, w)
        if sh > proposal_size:
            ratio: float = proposal_size / sh
            w = round(ratio * w)
            h = round(ratio * h)
        img.draft('RGB', (w, h))
        img = img.convert('RGB')
    return img


def build_dataset(
    datasets_str: str, subset_ratio: float, final_reso: int, mid_reso=1.125, hflip=False,
):
    # build augmentations
    mid_reso = round(min(mid_reso, 2) * final_reso)  # first resize to mid_reso, then crop to final_reso
    train_aug, val_aug = [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),  # transforms.Resize: resize the shorter edge to mid_reso
        transforms.RandomCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ], [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),  # transforms.Resize: resize the shorter edge to mid_reso
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ]
    if hflip: train_aug.insert(0, transforms.RandomHorizontalFlip())
    train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)
    
    train_set = UnlabeledImageFolders(datasets_str=datasets_str, subset_ratio=subset_ratio, transform=train_aug)  # todo: junfeng; only `train_set` required, no need to create a 'validation_set'
    
    # log dataset
    print(f'[Dataset] {len(train_set)=}')
    print_aug(train_aug, '[train]')
    print_aug(val_aug, '[val]')
    return train_set, val_aug


def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


def no_transform(x): return x


def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')
