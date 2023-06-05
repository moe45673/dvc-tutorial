from joblib import Parallel, delayed
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.restoration import denoise_tv_chambolle
from pathlib import Path
from tqdm import tqdm
from dvc.api import params_show
import numpy as np

DATA_DIR = Path("data")
train_dir = DATA_DIR / "raw" / "train"
test_dir = DATA_DIR / "raw" / "test"


def save_image(image_array, image_path):
    """
    Save an image given as NumPy array using its current path
    to create its target_path
    """
    # Create the path to the image in the prepared directory
    target_path = str(image_path).replace("raw", "prepared")
    Path(target_path).parent.mkdir(parents=True, exist_ok=True)

    # Scale image data and convert to uint8
    image_array = (image_array * 255).astype(np.uint8)

    imsave(target_path, image_array)


def resize_image(image_path, target_size):
    """
    Resize image to target size.
    """
    image = imread(image_path) / 255.0
    image = resize(image, target_size, anti_aliasing=True)

    save_image(image, image_path)


def denoise_image(image_path, weight):
    """
    Denoise image using total variation filter.
    """
    image = imread(image_path) / 255.0
    image = denoise_tv_chambolle(image, weight=weight, multichannel=True)

    save_image(image, image_path)


if __name__ == "__main__":
    image_paths = []
    params = params_show()['preprocess']
    denoise_weight = params["denoise_weight"]

    for directory in list(train_dir.iterdir()) + \
                     list(test_dir.iterdir()):
        image_paths.extend(list(directory.glob("*.png")))

    Parallel(n_jobs=10, backend="multiprocessing")(
        delayed(denoise_image)(path, denoise_weight) for path in tqdm(image_paths)
    )
