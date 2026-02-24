#!/usr/bin/env python3

import pathlib
import typing

import imgviz
import numpy as np
import onnxruntime
import PIL.Image
import torch
from loguru import logger
from numpy.typing import NDArray
from osam._models.yoloworld.clip import tokenize
from torchvision.transforms import v2

from infer_torch import get_replace_freqs_cis
from sam3.model.sam3_image import Sam3Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


# =========================
# Image Encoder
# =========================
class _ImageEncoder(torch.nn.Module):
    def __init__(self, processor: Sam3Processor) -> None:
        super().__init__()
        self._processor = processor
        self._transform = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5]*3, std=[0.5]*3),
            ]
        )

    def forward(self, image: torch.Tensor):
        image = self._transform(image).unsqueeze(0)

        backbone_out = self._processor.model.backbone._forward_image_no_act_ckpt(image)

        del backbone_out["vision_features"]
        del backbone_out["sam2_backbone_out"]

        return (
            *backbone_out["vision_pos_enc"],
            *backbone_out["backbone_fpn"],
        )


def _export_image_encoder(processor: Sam3Processor, image: PIL.Image.Image):
    image = image.resize((1008, 1008), resample=PIL.Image.BILINEAR)
    onnx_file = pathlib.Path("models/sam3_image_encoder.onnx")

    if not onnx_file.exists():
        encoder = _ImageEncoder(processor).eval().cuda()
        input_image = v2.functional.to_image(image).cuda()

        logger.debug("Exporting image encoder...")

        torch.onnx.export(
            encoder,
            args=(input_image,),
            f=onnx_file,
            input_names=["image"],
            output_names=[
                "vision_pos_enc_0",
                "vision_pos_enc_1",
                "vision_pos_enc_2",
                "backbone_fpn_0",
                "backbone_fpn_1",
                "backbone_fpn_2",
            ],
            opset_version=18,
            verify=False,
        )

    session = onnxruntime.InferenceSession(str(onnx_file))
    output = session.run(None, {"image": np.asarray(image).transpose(2, 0, 1)})

    return output[:3], output[3:]


# =========================
# Language Encoder
# =========================
class _LanguageEncoder(torch.nn.Module):
    def __init__(self, processor: Sam3Processor):
        super().__init__()
        self._processor = processor

    def forward(self, tokens: torch.Tensor):
        model = self._processor.model
        text_attention_mask = (tokens != 0).bool()

        inputs_embeds = model.backbone.language_backbone.encoder.token_embedding(tokens)
        _, text_memory = model.backbone.language_backbone.encoder(tokens)

        text_attention_mask = text_attention_mask.ne(1)
        text_memory = text_memory.transpose(0, 1)
        text_memory_resized = model.backbone.language_backbone.resizer(text_memory)

        return (
            text_attention_mask,
            text_memory_resized,
            inputs_embeds.transpose(0, 1),
        )


def _export_language_encoder(processor: Sam3Processor):
    tokens = tokenize(texts=["person"], context_length=32)
    onnx_file = pathlib.Path("models/sam3_language_encoder.onnx")

    if not onnx_file.exists():
        encoder = _LanguageEncoder(processor).eval().cuda()
        tokens_input = torch.from_numpy(tokens).cuda()

        logger.debug("Exporting language encoder...")

        torch.onnx.export(
            encoder,
            args=(tokens_input,),
            f=onnx_file,
            input_names=["tokens"],
            output_names=["text_attention_mask", "text_memory", "text_embeds"],
            opset_version=18,
            verify=False,
        )

    session = onnxruntime.InferenceSession(str(onnx_file))
    return session.run(None, {"tokens": tokens})


# =========================
# Decoder
# =========================
class _Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._model = build_sam3_image_model()
        self._processor = Sam3Processor(self._model)

    def forward(
        self,
        original_height,
        original_width,
        vision_pos_enc_0,
        vision_pos_enc_1,
        vision_pos_enc_2,
        backbone_fpn_0,
        backbone_fpn_1,
        backbone_fpn_2,
        language_mask,
        language_features,
        language_embeds,
        box_coords,
        box_labels,
        box_masks,
    ):
        geometric_prompt = self._processor.model._get_dummy_prompt()
        geometric_prompt.box_embeddings = box_coords
        geometric_prompt.box_labels = box_labels
        geometric_prompt.box_mask = box_masks

        state = {
            "original_height": original_height,
            "original_width": original_width,
            "backbone_out": {
                "vision_pos_enc": [
                    vision_pos_enc_0,
                    vision_pos_enc_1,
                    vision_pos_enc_2,
                ],
                "backbone_fpn": [
                    backbone_fpn_0,
                    backbone_fpn_1,
                    backbone_fpn_2,
                ],
                "language_mask": language_mask,
                "language_features": language_features,
                "language_embeds": language_embeds,
            },
            "geometric_prompt": geometric_prompt,
        }

        result = self._processor._forward_grounding(state)
        return result["boxes"], result["scores"], result["masks"]


def main():
    model = build_sam3_image_model()
    get_replace_freqs_cis(model)
    processor = Sam3Processor(model)

    image = PIL.Image.open("images/bus.jpg")

    vision_pos_enc, backbone_fpn = _export_image_encoder(processor, image)
    language_mask, language_features, language_embeds = _export_language_encoder(processor)

    decoder = _Decoder().eval().cuda()

    torch.onnx.export(
        decoder,
        args=(
            torch.tensor(image.height).cuda(),
            torch.tensor(image.width).cuda(),
            *[torch.tensor(x).cuda() for x in vision_pos_enc],
            *[torch.tensor(x).cuda() for x in backbone_fpn],
            torch.tensor(language_mask).cuda(),
            torch.tensor(language_features).cuda(),
            torch.tensor(language_embeds).cuda(),
            torch.tensor([[[0.16, 0.40, 0.06, 0.02]]], dtype=torch.float32).cuda(),
            torch.tensor([[1]], dtype=torch.int64).cuda(),
            torch.tensor([[True]]).cuda(),
        ),
        f="models/sam3_decoder.onnx",
        input_names=[
            "original_height",
            "original_width",
            "vision_pos_enc_0",
            "vision_pos_enc_1",
            "vision_pos_enc_2",
            "backbone_fpn_0",
            "backbone_fpn_1",
            "backbone_fpn_2",
            "language_mask",
            "language_features",
            "language_embeds",
            "box_coords",
            "box_labels",
            "box_masks",
        ],
        output_names=["boxes", "scores", "masks"],
        opset_version=18,
        verify=False,
    )

    print("Export DONE ✔")


if __name__ == "__main__":
    main()