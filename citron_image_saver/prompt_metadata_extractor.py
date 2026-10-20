import re
from typing import List, Dict, Tuple
from comfy.sd1_clip import escape_important, unescape_important, token_weights

from .utils import full_embedding_path_for, full_lora_path_for, get_sha256

"""
Extracts Embeddings and Lora's from the given prompts
and allows asking for their sha's
"""
class PromptMetadataExtractor:
    # Anything that follows embedding:<characters except , or whitespace
    EMBEDDING: str = r'embedding:([^,\s\(\)\:]+)'
    # Anything that follows <lora:NAME> with allowance for :weight, :weight.fractal or LBW
    LORA: str = r'<lora:([^>:]+)(?::([^>]+))?>'

    def __init__(self, prompts: List[str]) -> None:
        self.__embeddings: Dict[str, Tuple[str, float, str]] = {}
        self.__loras: Dict[str, Tuple[str, float, str]] = {}
        self.__perform(prompts)

    def get_embeddings(self) -> Dict[str, Tuple[str, float, str]]:
        """
        Returns the embeddings used in the given prompts
        Example output: {"EasyNegative": ("path/to/embedding", 1.0, "66a7279a88")}
        """
        return self.__embeddings

    def get_loras(self) -> Dict[str, Tuple[str, float, str]]:
        """
        Returns the lora's used in the given prompts
        Example output: {"epi_noiseoffset2": ("path/to/lora", 1.0, "81680c064e")}
        """
        return self.__loras

    # Private API
    def __perform(self, prompts: List[str]) -> None:
        for prompt in prompts:
            # Use ComfyUI's built-in attention parser to get accurate weights for embeddings
            parsed = ((unescape_important(value), weight) for value, weight in token_weights(escape_important(prompt), 1.0))
            for text, weight in parsed:
                embeddings = re.findall(self.EMBEDDING, text, re.IGNORECASE | re.MULTILINE)
                for embedding in embeddings:
                    self.__extract_embedding_information(embedding, weight)
            loras = re.findall(self.LORA, prompt, re.IGNORECASE | re.MULTILINE)
            for lora in loras:
                self.__extract_lora_information(lora)

    def __extract_embedding_information(self, embedding: str, weight: float) -> None:
        embedding_name = f"embed:{embedding}"
        embedding_path = full_embedding_path_for(embedding)
        if embedding_path == None:
            return
        sha = self.__get_shortened_sha(embedding_path)
        self.__embeddings[embedding_name] = (embedding_path, weight, sha)

    def __extract_lora_information(self, lora: Tuple[str, str]) -> None:
        lora_name = f"LORA:{lora[0]}"
        lora_path = full_lora_path_for(lora[0])
        if lora_path == None:
            return
        try:
            lora_weight = float(lora[1].split(':')[0])
        except (ValueError, TypeError):
            lora_weight = 1.0
        sha = self.__get_shortened_sha(lora_path)
        self.__loras[lora_name] = (lora_path, lora_weight, sha)

    def __get_shortened_sha(self, file_path: str) -> str:
        return get_sha256(file_path)[:10]
