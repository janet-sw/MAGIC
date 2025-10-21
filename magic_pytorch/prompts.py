from importlib import resources
import os
import functools
import random
import inflect
import json

IE = inflect.engine()
ASSETS_PATH = resources.files("magic_pytorch.assets")

@functools.cache
def _load_lines(path):
    """
    Load lines from a file. First tries to load from `path` directly, and if that doesn't exist, searches the
    `magic_pytorch/assets` directory for a file named `path`.
    """
    if not os.path.exists(path):
        newpath = ASSETS_PATH.joinpath(path)
    if not os.path.exists(newpath):
        raise FileNotFoundError(f"Could not find {path} or ddpo_pytorch.assets/{path}")
    path = newpath
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def from_file(path, low=None, high=None):
    prompts = _load_lines(path)[low:high]
    return random.choice(prompts), {}


def imagenet_all():
    return from_file("imagenet_classes.txt")


def imagenet_animals():
    return from_file("imagenet_classes.txt", 0, 398)


def imagenet_dogs():
    return from_file("imagenet_classes.txt", 151, 269)


def simple_animals():
    return from_file("simple_animals.txt")

def anything_prompt():
    return from_file("anything_prompt.txt")

def unsafe_prompt():
    return from_file("unsafe_prompt.txt")

def lesion_body_part():
    return from_file("lesion_prompt.txt")


def from_json_file(path, low=None, high=None):
    with open(path, "r") as f:
        entries = json.load(f)
    
    return random.choice(entries), {}

def load_image_prompt_pair():
    return from_json_file("/media/janet/DermDPO/magic_pytorch/assets/all_prompts.json")