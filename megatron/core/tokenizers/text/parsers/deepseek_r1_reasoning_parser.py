# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from megatron.core.tokenizers.text.parsers.base_parser import BaseParser


class DeepSeekR1ReasoningParser(BaseParser):
    """Parser for DeepSeek R1 style reasoning output."""

    @staticmethod
    def parse(text: str, **kwargs) -> tuple[str, dict[str, str]]:
        """
        Extracts the reasoning content from the text using <think>...</think> tags.
        Only extracts the first set of think tags.
        If an initial <think> tag is not present but a </think> tag is,
        it will infer a <think> tag at the beginning of the text.

        Args:
            text (str): The text to parse.

        Returns:
            tuple[str, dict[str, str]]: A tuple containing the unprocessed text
            and a dictionary with the extracted reasoning content.
        """
        if text is None:
            return "", {}

        close_idx = text.find("</think>")
        if close_idx == -1:
            return text, {}

        open_idx = text.find("<think>")
        if 0 <= open_idx < close_idx:
            pre_text = text[:open_idx]
            reasoning_content = text[open_idx + len("<think>") : close_idx]
            remaining_text = text[close_idx + len("</think>") :]
            return pre_text + remaining_text, {'reasoning': reasoning_content}

        # If the opening tag is missing or appears after the closing tag,
        # infer that reasoning started at the beginning of the text.
        reasoning_content = text[:close_idx]
        remaining_text = text[close_idx + len("</think>") :]
        return remaining_text, {'reasoning': reasoning_content}
