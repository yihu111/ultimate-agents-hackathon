from typing import Optional

from langchain_core.messages import HumanMessage


# ----------- Dimension proposal prompts -----------

DIMENSIONS_SYSTEM = """You are a creative product art director.
Given a product description and optional audience/brand guidance, propose exactly two orthogonal visual dimensions
that will diversify product images. Each dimension must have exactly three distinct, widely-applicable values.
Keep names concise and values specific but generalizable."""


def build_dimensions_human(product_description: str, extra_guidance: Optional[str]) -> HumanMessage:
    guidance = f"\nAdditional guidance:\n{extra_guidance}" if extra_guidance else ""
    text = f"""Product description:
{product_description}
{guidance}

Task:
- Propose exactly two dimensions (e.g., "palette", "tone"), each with exactly three values.
- Values must be mutually distinct within a dimension and make sense when cross-combined (3x3).

Return only the structured fields."""
    return HumanMessage(content=[{"type": "text", "text": text}])


# ----------- Prompt engineering prompts -----------

PROMPT_SYSTEM = """You are a prompt engineer for a modern diffusion model.
Create a single, self-contained, high-quality prompt for product image generation.
Explicitly incorporate the two provided dimension values in a natural way.
Do not mention 'dimension' or 'value'—write the final prompt as it should be used by the model."""


def build_prompt_human(
    product_description: str,
    extra_guidance: Optional[str],
    dim_a_name: str,
    dim_b_name: str,
    dim_a_value: str,
    dim_b_value: str,
) -> HumanMessage:
    guidance = f"\nAdditional guidance:\n{extra_guidance}" if extra_guidance else ""
    text = f"""Product description:
{product_description}
{guidance}

Target variation:
- {dim_a_name}: {dim_a_value}
- {dim_b_name}: {dim_b_value}

Write one final diffusion prompt, incorporating these target variations."""
    return HumanMessage(content=[{"type": "text", "text": text}])


