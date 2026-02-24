#!/usr/bin/env python3

import argparse
import pathlib
import numpy as np
import PIL.Image
import imgviz
import tritonclient.http as httpclient
from loguru import logger
from osam._models.yoloworld.clip import tokenize
from tritonclient.utils import np_to_triton_dtype


TRITON_URL = "localhost:8881"
MODEL_NAME = "sam3_pipeline"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=pathlib.Path, required=True)
    parser.add_argument("--text-prompt", type=str)
    parser.add_argument("--box-prompt", type=str)
    parser.add_argument("--output", type=pathlib.Path, default="output.jpg")
    return parser.parse_args()


def main():
    args = parse_args()

    image = PIL.Image.open(args.image).convert("RGB")
    original_width, original_height = image.size

    resized = image.resize((1008, 1008))
    image_np = np.asarray(resized).transpose(2, 0, 1).astype(np.uint8)

    text_prompt = args.text_prompt if args.text_prompt else "visual"
    tokens = tokenize(texts=[text_prompt], context_length=32)

    if args.box_prompt:
        box = np.array([float(x) for x in args.box_prompt.split(",")], dtype=np.float32)
    else:
        box = np.array([0, 0, 0, 0], dtype=np.float32)

    box_coords = box.reshape(1, 1, 4)
    box_labels = np.array([[1]], dtype=np.int64)
    box_masks = np.array([[False if args.box_prompt else True]], dtype=np.bool_)

    original_height_np = np.array([original_height], dtype=np.int64)
    original_width_np = np.array([original_width], dtype=np.int64)

    client = httpclient.InferenceServerClient(TRITON_URL)

    inputs = []

    def add_input(name, data):
        inp = httpclient.InferInput(name, data.shape, np_to_triton_dtype(data.dtype))
        inp.set_data_from_numpy(data)
        inputs.append(inp)

    add_input("image", image_np)
    add_input("tokens", tokens)
    add_input("original_height", original_height_np)
    add_input("original_width", original_width_np)
    add_input("box_coords", box_coords)
    add_input("box_labels", box_labels)
    add_input("box_masks", box_masks)

    logger.info("Running Triton inference...")
    results = client.infer(MODEL_NAME, inputs)

    boxes = results.as_numpy("boxes")
    scores = results.as_numpy("scores")
    masks = results.as_numpy("masks")

    logger.info(
        f"Output shapes: boxes={boxes.shape}, scores={scores.shape}, masks={masks.shape}"
    )

    # ===== Visualization =====
    viz = imgviz.instances2rgb(
        image=np.asarray(image),
        masks=masks[:, 0, :, :],
        bboxes=boxes[:, [1, 0, 3, 2]],
        labels=np.arange(len(masks)) + 1,
        captions=[f"{text_prompt}: {s:.0%}" for s in scores],
        font_size=max(1, min(image.size) // 40),
    )

    output_image = PIL.Image.fromarray(viz)
    output_image.save(args.output)

    logger.success(f"Saved result to {args.output}")


if __name__ == "__main__":
    main()