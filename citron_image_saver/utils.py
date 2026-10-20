import hashlib
import os
from typing import Optional
from collections.abc import Collection, Iterator
from pathlib import Path
from tqdm import tqdm
import folder_paths
import re

def sanitize_filename(filename: str) -> str:
    """Remove characters that are unsafe for filenames."""
    # Remove characters that are generally unsafe across file systems
    unsafe_chars = r'[<>:"|?*\x00-\x1f]'
    sanitized = re.sub(unsafe_chars, '', filename)

    # Remove trailing periods and spaces (problematic on Windows)
    sanitized = sanitized.rstrip('. ')
    return sanitized

def get_sha256(file_path: str) -> str:
    """
    Given the file path, finds a matching sha256 file, or creates one
    based on the headers in the source file
    """
    file_no_ext = os.path.splitext(file_path)[0]
    hash_file = file_no_ext + ".sha256"

    if os.path.exists(hash_file):
        try:
            with open(hash_file, "r") as f:
                return f.read().strip()
        except OSError as e:
            print(f"citron_image_saver: Error reading existing hash file: {e}")

    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        file_size = os.fstat(f.fileno()).st_size
        block_size = 1048576 # 1 MB

        print(f"citron_image_saver: Calculating sha256 for '{Path(file_path).stem}'")
        with tqdm(None, None, file_size, unit="B", unit_scale=True, unit_divisor=1024) as progress_bar:
            for byte_block in iter(lambda: f.read(block_size), b""):
                progress_bar.update(len(byte_block))
                sha256_hash.update(byte_block)

    try:
        with open(hash_file, "w") as f:
            f.write(sha256_hash.hexdigest())
    except OSError as e:
        print(f"citron_image_saver: Error writing hash to {hash_file}: {e}")

    return sha256_hash.hexdigest()

def full_path_for(folder_name: str, name: str) -> Optional[str]:
    """
    Based on a lora/embedding name, e.g., 'epi_noise_offset2', finds the path as known in comfy, including extension.
    """
    matching_file = get_file_path_match(folder_name, name)
    if matching_file is None:
        print(f'citron_image_saver: could not find full path to {folder_name} entry "{name}"')
        return None
    return folder_paths.get_full_path(folder_name, matching_file)

def full_embedding_path_for(embedding: str) -> Optional[str]:
    return full_path_for("embeddings", embedding)

def full_lora_path_for(lora: str) -> Optional[str]:
    return full_path_for("loras", lora)

def full_checkpoint_path_for(model_name: str) -> str:
    if not model_name:
        return ''

    supported_extensions = set(folder_paths.supported_pt_extensions) | {".gguf"}

    matching_checkpoint = get_file_path_match("checkpoints", model_name, supported_extensions)
    if matching_checkpoint is not None:
        return folder_paths.get_full_path("checkpoints", matching_checkpoint)

    matching_model = get_file_path_match("diffusion_models", model_name, supported_extensions)
    if matching_model:
        return folder_paths.get_full_path("diffusion_models", matching_model)

    print(f'Could not find full path to checkpoint "{model_name}"')
    return ''

def get_file_path_iterator(folder_name: str, supported_extensions: Optional[Collection[str]] = None) -> Iterator[Path]:
    """
    Returns an iterator over valid file paths for the specified model folder.
    """
    if supported_extensions is None:
        return (Path(x) for x in folder_paths.get_filename_list(folder_name))
    else:
        return custom_file_path_generator(folder_name, supported_extensions)

def custom_file_path_generator(folder_name: str, supported_extensions: Collection[str]) -> Iterator[Path]:
    """
    Generator function for file paths, allowing for a customized extension check.
    """
    model_paths = folder_paths.folder_names_and_paths.get(folder_name, [[], set()])[0]
    for path in model_paths:
        if os.path.exists(path):
            base_path = Path(path)
            for root, _, files in os.walk(path):
                root_path = Path(root).relative_to(base_path)
                for file in files:
                    file_path = root_path / file
                    if file_path.suffix.lower() in supported_extensions:
                        yield file_path

def get_file_path_match(folder_name: str, file_name: str, supported_extensions: Optional[Collection[str]] = None) -> Optional[str]:
    file_path = Path(file_name)
    paths = list(get_file_path_iterator(folder_name, supported_extensions))

    # first try full path match, then fallback to just name match, matching the extension if appropriate
    if file_path.suffix.lower() not in (supported_extensions if supported_extensions is not None else folder_paths.supported_pt_extensions):
        matchers = (lambda p: p.with_suffix('') == file_path, lambda p: p.stem == file_path.name)
    else:
        matchers = (lambda p: p == file_path, lambda p: p.name == file_path.name)

    for matches in matchers:
        match = next((p for p in paths if matches(p)), None)
        if match is not None:
            return str(match)
    return None
