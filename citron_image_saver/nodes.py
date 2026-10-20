import os
from datetime import datetime
from dataclasses import dataclass
from typing import Any
import json
import numpy as np
import re

from PIL import Image
import torch

import folder_paths
from nodes import MAX_RESOLUTION

from .saver.saver import save_image
from .utils import sanitize_filename, get_sha256, full_checkpoint_path_for
from .prompt_metadata_extractor import PromptMetadataExtractor

MAX_HASH_LENGTH = 16
MAX_MANUAL_HASHES = 30
DEFAULT_TIME_FORMAT = "%Y-%m-%d-%H%M%S"

# Matches 'anything' or 'anything:anything' with trimmed white space
RE_MANUAL_HASH = re.compile(r'^\s*([^:]+?)(?:\s*:\s*([^\s:][^:]*?))?\s*$')

def parse_checkpoint_name(ckpt_name: str) -> str:
    return os.path.basename(ckpt_name)

def strip_model_extension(filename: str) -> str:
    """Remove the extension only if it's a known model file extension, preserving dots in model names."""
    name_without_ext, ext = os.path.splitext(filename)
    if ext.lower() in folder_paths.supported_pt_extensions | {".gguf"}:
        return name_without_ext
    return filename

def parse_checkpoint_name_without_extension(ckpt_name: str) -> str:
    return strip_model_extension(parse_checkpoint_name(ckpt_name))

def get_timestamp(time_format: str) -> str:
    now = datetime.now()
    try:
        return now.strftime(time_format)
    except ValueError:
        return now.strftime(DEFAULT_TIME_FORMAT)

def save_json(image_info: dict[str, Any] | None, filename: str) -> None:
    workflow = (image_info or {}).get('workflow')
    if workflow is None:
        print('citron_image_saver: No workflow found, skipping saving of JSON')
        return
    try:
        with open(f'{filename}.json', 'w') as workflow_file:
            json.dump(workflow, workflow_file)
        print(f'citron_image_saver: Saved workflow to {filename}.json')
    except OSError as e:
        print(f'citron_image_saver: Failed to save workflow as json due to: {e}')

@dataclass
class Metadata:
    modelname: str = ''
    width: int = 512
    height: int = 512
    seed: int = 0
    steps: int = 20
    cfg: float = 7.0
    sampler_name: str = ''
    scheduler_name: str = 'normal'
    denoise: float = 1.0
    clip_skip: int = 0
    custom: str = ''
    a111_params: str = ''
    final_hashes: str = ''

def make_pathname(pattern: str, metadata: Metadata, counter: int, time_format: str) -> str:
    replacements = {
        "%date": get_timestamp("%Y-%m-%d"),
        "%time": get_timestamp(time_format),
        "%model": parse_checkpoint_name(metadata.modelname),
        "%width": str(metadata.width),
        "%height": str(metadata.height),
        "%seed": str(metadata.seed),
        "%counter": str(counter),
        "%sampler_name": metadata.sampler_name,
        "%steps": str(metadata.steps),
        "%cfg": str(metadata.cfg),
        "%scheduler_name": metadata.scheduler_name,
        "%basemodelname": parse_checkpoint_name_without_extension(metadata.modelname),
        "%denoise": str(metadata.denoise),
        "%clip_skip": str(metadata.clip_skip),
        "%custom": metadata.custom,
    }
    for key, value in replacements.items():
        pattern = pattern.replace(key, value)

    directory, basename = os.path.split(pattern)
    return os.path.join(directory, sanitize_filename(basename))

def make_filename(pattern: str, metadata: Metadata, counter: int, time_format: str) -> str:
    filename = make_pathname(pattern, metadata, counter, time_format)
    return get_timestamp(time_format) if filename == "" else filename

def get_multiple_models(modelname: str, additional_hashes: str) -> tuple[str, str]:
    """Use the first model as the primary one and move the rest into additional_hashes."""
    model_names = [m.strip() for m in modelname.split(',')]

    for additional_model in model_names[1:]:
        additional_ckpt_path = full_checkpoint_path_for(additional_model)
        if additional_ckpt_path:
            if additional_hashes:
                additional_hashes += ","
            additional_hashes += f"{additional_model}:{get_sha256(additional_ckpt_path)[:10]}"
    return model_names[0], additional_hashes

def parse_manual_hashes(additional_hashes: str, existing_hashes: set[str]) -> dict[str, tuple[str | None, float | None, str]]:
    """Normalize additional_hashes (comma or newline separated, optionally 'name:hash') into resource entries."""
    manual_entries: dict[str, tuple[str | None, float | None, str]] = {}
    unnamed_count = 0

    for entry in (additional_hashes.replace("\n", ",").split(",") if additional_hashes else []):
        match = RE_MANUAL_HASH.search(entry)
        if match is None:
            print(f"citron_image_saver: Invalid additional hash string: '{entry}'")
            continue

        groups = tuple(group for group in match.groups() if group)
        name, hash = groups if len(groups) > 1 else (None, groups[0])

        if len(hash) > MAX_HASH_LENGTH:
            print(f"citron_image_saver: Skipping hash. Length exceeds maximum of {MAX_HASH_LENGTH} characters: {hash}")
            continue

        if any(hash.lower() == existing_hash.lower() for _, _, existing_hash in manual_entries.values()):
            print(f"citron_image_saver: Skipping duplicate hash: {hash}")
            continue

        if hash.lower() in existing_hashes:
            print(f"citron_image_saver: Skipping manual hash already present in resources: {hash}")
            continue

        if name is None:
            unnamed_count += 1
            name = f"manual{unnamed_count}"
        elif name in manual_entries:
            print(f"citron_image_saver: Duplicate manual hash name '{name}' is being overwritten.")

        manual_entries[name] = (None, None, hash)

        if len(manual_entries) >= MAX_MANUAL_HASHES:
            print(f"citron_image_saver: Reached maximum limit of {MAX_MANUAL_HASHES} manual hashes. Skipping the rest.")
            break

    return manual_entries

def get_unique_filename(output_path: str, filename_prefix: str, extension: str) -> str:
    existing_files = [f for f in os.listdir(output_path) if f.startswith(filename_prefix) and f.endswith(extension)]

    suffixes: list[int] = []
    for f in existing_files:
        parts = os.path.splitext(f)[0].split('_')
        if parts[-1].isdigit():
            suffixes.append(int(parts[-1]))

    if not existing_files:
        return filename_prefix
    return f"{filename_prefix}_{max(suffixes, default=0) + 1:02d}"

def make_metadata(modelname: str, positive: str, negative: str, width: int, height: int, seed_value: int, steps: int, cfg: float, sampler_name: str, scheduler_name: str, denoise: float, clip_skip: int, custom: str, additional_hashes: str) -> Metadata:
    modelname, additional_hashes = get_multiple_models(modelname, additional_hashes)

    ckpt_path = full_checkpoint_path_for(modelname)
    modelhash = get_sha256(ckpt_path)[:10] if ckpt_path else ""

    metadata_extractor = PromptMetadataExtractor([positive, negative])
    embeddings = metadata_extractor.get_embeddings()
    loras = metadata_extractor.get_loras()

    existing_hashes = {modelhash.lower()} | {t[2].lower() for t in loras.values()} | {t[2].lower() for t in embeddings.values()}
    manual_entries = parse_manual_hashes(additional_hashes, existing_hashes)

    hashes = {key: hash_value.upper() for key, (_, _, hash_value) in (embeddings | loras | manual_entries).items()}
    if modelhash:
        hashes["model"] = modelhash.upper()

    clip_skip_str = f", Clip skip: {abs(clip_skip)}" if clip_skip != 0 else ""
    custom_str = f", {custom}" if custom else ""
    model_hash_str = f", Model hash: {modelhash.upper()}" if modelhash else ""
    hashes_str = f", Hashes: {json.dumps(hashes, separators=(',', ':'))}" if hashes else ""

    a111_params = (
        f"{positive.strip()}\nNegative prompt: {negative.strip()}\n"
        f"Steps: {steps}, Sampler: {sampler_name}, CFG scale: {cfg}, Seed: {seed_value}, "
        f"Size: {width}x{height}{clip_skip_str}{custom_str}{model_hash_str}, Model: {parse_checkpoint_name_without_extension(modelname)}{hashes_str}, Version: ComfyUI"
    )

    all_resources = {modelname: (ckpt_path, None, modelhash)} | loras | embeddings | manual_entries
    final_hashes = ",".join(
        f"{strip_model_extension(name.split(':')[-1])}:{hash_value}" if name else hash_value
        for name, (_, _, hash_value) in all_resources.items()
        if hash_value
    )

    return Metadata(modelname, width, height, seed_value, steps, cfg, sampler_name, scheduler_name, denoise, clip_skip, custom, a111_params, final_hashes)

def save_images(
    images: list[torch.Tensor],
    filename_pattern: str,
    extension: str,
    path: str,
    quality_jpeg_or_webp: int,
    lossless_webp: bool,
    optimize_png: bool,
    prompt: dict[str, Any] | None,
    extra_pnginfo: dict[str, Any] | None,
    save_workflow_as_json: bool,
    embed_workflow: bool,
    counter: int,
    time_format: str,
    show_preview: bool,
    metadata: Metadata,
) -> dict[str, Any]:
    """Shared save path of the saver nodes. Returns the node result with the optional UI preview."""
    path = make_pathname(path, metadata, counter, time_format)
    filename_prefix = make_filename(filename_pattern, metadata, counter, time_format)

    output_path = os.path.join(folder_paths.output_directory, path)
    if not os.path.exists(output_path.strip()):
        print(f'citron_image_saver: The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
        os.makedirs(output_path, exist_ok=True)

    filenames: list[str] = []
    for image in images:
        img = Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))

        unique_prefix = get_unique_filename(output_path, filename_prefix, extension)
        final_filename = f"{unique_prefix}.{extension}"

        save_image(img, os.path.join(output_path, final_filename), extension, quality_jpeg_or_webp, lossless_webp, optimize_png, metadata.a111_params, prompt, extra_pnginfo, embed_workflow)

        if save_workflow_as_json:
            save_json(extra_pnginfo, os.path.join(output_path, unique_prefix))

        filenames.append(final_filename)

    result: dict[str, Any] = {"result": (metadata.final_hashes, metadata.a111_params)}
    if show_preview:
        subfolder = os.path.normpath(path)
        result["ui"] = {"images": [{"filename": f, "subfolder": subfolder if subfolder != '.' else '', "type": 'output'} for f in filenames]}
    return result

HASHES_TOOLTIP = "Comma-separated list of the hashes to chain with other Image Saver additional_hashes"
A1111_TOOLTIP = "Written parameters to the image metadata"

# Shared input definitions; each node picks the ones it needs, in its own order.
INPUT_SPECS: dict[str, tuple] = {
    "images":                ("IMAGE",    {"tooltip": "image(s) to save"}),
    "filename":              ("STRING",   {"default": '%time_%basemodelname_%seed', "multiline": False, "tooltip": "filename (available variables: %date, %time, %model, %width, %height, %seed, %counter, %sampler_name, %steps, %cfg, %scheduler_name, %basemodelname, %denoise, %clip_skip)"}),
    "path":                  ("STRING",   {"default": '', "multiline": False, "tooltip": "path to save the images (under Comfy's save directory)"}),
    "extension":             (['png', 'jpeg', 'jpg', 'webp'], {"tooltip": "file extension/type to save image as"}),
    "steps":                 ("INT",      {"default": 20, "min": 1, "max": 10000, "tooltip": "number of steps"}),
    "cfg":                   ("FLOAT",    {"default": 7.0, "min": 0.0, "max": 100.0, "tooltip": "CFG value"}),
    "modelname":             ("STRING",   {"default": '', "multiline": False, "tooltip": "model name (can be multiple, separated by commas)"}),
    "sampler_name":          ("STRING",   {"default": '', "multiline": False, "tooltip": "sampler name (as string)"}),
    "scheduler_name":        ("STRING",   {"default": 'normal', "multiline": False, "tooltip": "scheduler name (as string)"}),
    "positive":              ("STRING",   {"default": 'unknown', "multiline": True, "tooltip": "positive prompt"}),
    "negative":              ("STRING",   {"default": 'unknown', "multiline": True, "tooltip": "negative prompt"}),
    "seed_value":            ("INT",      {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "seed"}),
    "width":                 ("INT",      {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "image width"}),
    "height":                ("INT",      {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "image height"}),
    "lossless_webp":         ("BOOLEAN",  {"default": True, "tooltip": "if True, saved WEBP files will be lossless"}),
    "quality_jpeg_or_webp":  ("INT",      {"default": 100, "min": 1, "max": 100, "tooltip": "quality setting of JPEG/WEBP"}),
    "optimize_png":          ("BOOLEAN",  {"default": False, "tooltip": "if True, saved PNG files will be optimized (can reduce file size but is slower)"}),
    "counter":               ("INT",      {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "counter"}),
    "denoise":               ("FLOAT",    {"default": 1.0, "min": 0.0, "max": 1.0, "tooltip": "denoise value"}),
    "clip_skip":             ("INT",      {"default": 0, "min": -24, "max": 24, "tooltip": "skip last CLIP layers (positive or negative value, 0 for no skip)"}),
    "time_format":           ("STRING",   {"default": DEFAULT_TIME_FORMAT, "multiline": False, "tooltip": "timestamp format"}),
    "save_workflow_as_json": ("BOOLEAN",  {"default": False, "tooltip": "if True, also saves the workflow as a separate JSON file"}),
    "embed_workflow":        ("BOOLEAN",  {"default": True, "tooltip": "if True, embeds the workflow in the saved image files.\nStable for PNG, experimental for WEBP.\nJPEG experimental and only if metadata size is below 65535 bytes"}),
    "additional_hashes":     ("STRING",   {"default": "", "multiline": False, "tooltip": "hashes separated by commas, optionally with names. 'Name:HASH' (e.g., 'MyLoRA:FF735FF83F98')"}),
    "show_preview":          ("BOOLEAN",  {"default": True, "tooltip": "if True, displays saved images in the UI preview"}),
    "custom":                ("STRING",   {"default": "", "multiline": False, "tooltip": "custom string to add to the metadata, inserted into the a111 string between clip skip and model hash"}),
    "metadata":              ("METADATA", {"default": None, "tooltip": "metadata to embed in the image"}),
}

def pick_inputs(*names: str) -> dict[str, tuple]:
    return {name: INPUT_SPECS[name] for name in names}

HIDDEN_INPUTS = {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}

class ImageSaverMetadata:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "optional": pick_inputs("modelname", "positive", "negative", "width", "height", "seed_value", "steps", "cfg", "sampler_name", "scheduler_name", "denoise", "clip_skip", "additional_hashes", "custom"),
        }

    RETURN_TYPES = ("METADATA","STRING","STRING")
    RETURN_NAMES = ("metadata","hashes","a1111_params")
    OUTPUT_TOOLTIPS = ("metadata for Citron Image Saver Simple", HASHES_TOOLTIP, A1111_TOOLTIP)
    FUNCTION = "get_metadata"
    CATEGORY = "Citron_ImageSaver"
    DESCRIPTION = "Prepare metadata for Citron Image Saver Simple"

    def get_metadata(
        self,
        modelname: str = "",
        positive: str = "unknown",
        negative: str = "unknown",
        width: int = 512,
        height: int = 512,
        seed_value: int = 0,
        steps: int = 20,
        cfg: float = 7.0,
        sampler_name: str = "",
        scheduler_name: str = "",
        denoise: float = 1.0,
        clip_skip: int = 0,
        custom: str = "",
        additional_hashes: str = "",
    ) -> tuple[Metadata, str, str]:
        metadata = make_metadata(modelname, positive, negative, width, height, seed_value, steps, cfg, sampler_name, scheduler_name, denoise, clip_skip, custom, additional_hashes)
        return (metadata, metadata.final_hashes, metadata.a111_params)

class ImageSaverSimple:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": pick_inputs("images", "filename", "path", "extension", "lossless_webp", "quality_jpeg_or_webp", "optimize_png", "embed_workflow", "save_workflow_as_json"),
            "optional": pick_inputs("metadata", "counter", "time_format", "show_preview"),
            "hidden": HIDDEN_INPUTS,
        }

    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("hashes","a1111_params")
    OUTPUT_TOOLTIPS = (HASHES_TOOLTIP, A1111_TOOLTIP)
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "Citron_ImageSaver"
    DESCRIPTION = "Save images with generation metadata"

    def save_images(self,
        images: list[torch.Tensor],
        filename: str,
        path: str,
        extension: str,
        lossless_webp: bool,
        quality_jpeg_or_webp: int,
        optimize_png: bool,
        embed_workflow: bool = True,
        save_workflow_as_json: bool = False,
        show_preview: bool = True,
        metadata: Metadata | None = None,
        counter: int = 0,
        time_format: str = DEFAULT_TIME_FORMAT,
        prompt: dict[str, Any] | None = None,
        extra_pnginfo: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if metadata is None:
            metadata = Metadata()

        return save_images(images, filename, extension, path, quality_jpeg_or_webp, lossless_webp, optimize_png, prompt, extra_pnginfo, save_workflow_as_json, embed_workflow, counter, time_format, show_preview, metadata)

class ImageSaver:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": pick_inputs("images", "filename", "path", "extension"),
            "optional": pick_inputs("steps", "cfg", "modelname", "sampler_name", "scheduler_name", "positive", "negative", "seed_value", "width", "height", "lossless_webp", "quality_jpeg_or_webp", "optimize_png", "counter", "denoise", "clip_skip", "time_format", "save_workflow_as_json", "embed_workflow", "additional_hashes", "show_preview", "custom"),
            "hidden": HIDDEN_INPUTS,
        }

    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("hashes","a1111_params")
    OUTPUT_TOOLTIPS = (HASHES_TOOLTIP, A1111_TOOLTIP)
    FUNCTION = "save_files"

    OUTPUT_NODE = True

    CATEGORY = "Citron_ImageSaver"
    DESCRIPTION = "Save images with generation metadata"

    def save_files(
        self,
        images: list[torch.Tensor],
        filename: str,
        path: str,
        extension: str,
        steps: int = 20,
        cfg: float = 7.0,
        modelname: str = "",
        sampler_name: str = "",
        scheduler_name: str = "normal",
        positive: str = "unknown",
        negative: str = "unknown",
        seed_value: int = 0,
        width: int = 512,
        height: int = 512,
        lossless_webp: bool = True,
        quality_jpeg_or_webp: int = 100,
        optimize_png: bool = False,
        counter: int = 0,
        denoise: float = 1.0,
        clip_skip: int = 0,
        time_format: str = DEFAULT_TIME_FORMAT,
        save_workflow_as_json: bool = False,
        embed_workflow: bool = True,
        additional_hashes: str = "",
        show_preview: bool = True,
        custom: str = "",
        prompt: dict[str, Any] | None = None,
        extra_pnginfo: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = make_metadata(modelname, positive, negative, width, height, seed_value, steps, cfg, sampler_name, scheduler_name, denoise, clip_skip, custom, additional_hashes)

        return save_images(images, filename, extension, path, quality_jpeg_or_webp, lossless_webp, optimize_png, prompt, extra_pnginfo, save_workflow_as_json, embed_workflow, counter, time_format, show_preview, metadata)
