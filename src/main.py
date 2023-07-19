from src.config import config
from src.core import prune_images_generic

if __name__ == "__main__":
    prune_images_generic(
        config["base_path"],
        config["w"],
        config["h"],
        config["score_threshold"],
        config["image_mean_threshold"],
    )
