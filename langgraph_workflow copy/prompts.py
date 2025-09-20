from typing import Optional

from langchain_core.messages import HumanMessage


# ----------- Dimension proposal prompts -----------

DIMENSIONS_SYSTEM = f"""You are a dimension-designer. Your task is to generate a set of orthogonal dimensions with concrete values that can guide exploration of creative ad prompts.  

Inputs You Will Receive  
1. Reference image - the uploaded product image that is the subject of the adverts.  
2. Text description - a short description of the product or intended direction.  
3. Parameters - num of dimensions, and num of values per dimension

Goals  
- Produce 'num_dimensions' distinct, orthogonal dimensions.  
- Each dimension should describe a conceptual or stylistic axis that could apply broadly to many possible advert prompts.  
- For each dimension, produce 'num_values_per_dimension' concrete values that span different parts of that axis.  
- Dimensions should not overlap heavily with one another; they should cover independent aspects of variation (e.g., tone vs palette vs environment).  
- Dimensions should be general enough to apply to most advert contexts, even if not all possible values are listed.  

Output You Must Produce  
- Present the dimensions and their values in a structured JSON format.  
- Keys: dimension names.  
- Values: lists of concrete values for that dimension.  

Style Rules  
- Keep dimension names concise and abstract (e.g., "Tone", "Lighting Style", "Scene Scale").  
- Keep concrete values short and descriptive (e.g., “calm”, “energetic”, “minimalist”).  
- Avoid overlap or redundancy between dimensions.  
- Do not use emojis, slang, or promotional filler.  
"""


def build_dimensions_human(num_dimensions: int, num_values_per_dim: int, product_description: str, extra_guidance: Optional[str]) -> HumanMessage:
    guidance = f"\nAdditional guidance:\n{extra_guidance}" if extra_guidance else ""
    text = f"""Product description:
{product_description}
{guidance}

Task:
- Number of dimensions: {num_dimensions}
- Number of values per dimension: {num_values_per_dim}
- Propose exactly two dimensions (e.g., "palette", "tone"), each with exactly three values.
- Values must be mutually distinct within a dimension and make sense when cross-combined (3x3).

Return only the structured fields."""
    return HumanMessage(content=[{"type": "text", "text": text}])


# ----------- Prompt engineering prompts -----------

PROMPT_SYSTEM = """You are a prompt-writer. Your task is to generate image generation prompts in a consistent, clean format.  

Inputs You Will Receive  
1. Reference image - the uploaded product image that must always be kept accurate and central.  
2. Dimension values - stylistic or semantic controls such as `tone: calm`, `palette: pastel`, `mood: energetic`, etc. These values should shape the environment, color, or feeling of the scene.  

Goals  
- Come up with interesting and creative prompt ideas for this type of advert spread.  
- Keep the ideas fresh and varied, but not over the top or cluttered.  
- Ensure the prompt could realistically guide the creation of a professional, polished ad image.  

Output You Must Produce  
Write a single prompt in the following format:  

- Start with:  
  "Use the uploaded [object/product] image as the product reference."  

- Follow with Scene: describe the setting, background, and overall environment. Ensure that the description reflects the given dimension values.  

- Follow with Composition: specify placement of the product, how it is framed, any camera angle, and where negative space is left. Keep language precise and concrete.  

- Follow with Lighting/Color: describe how light interacts with the object and the palette that should dominate. Integrate given palette or tone values here.  

- If you want text in the advert, use this exact format inside the prompt:  
  ```In large overlay text, clear with no typos: “<text>”  
  Subline text below the main text: “<text>”```

- End with Output format: specify aspect ratio(s) like 4:5, 9:16, etc.  

Style Rules  
- Always keep the reference image/object crisp, sharp, and unaltered.  
- Avoid exaggeration, clutter, or unrelated props.  
- Write in concise, technical description style, not in marketing voice.  
- Ensure the dimension values are reflected clearly in the scene, composition, or palette.  
- Do not use emojis, slang, or promotional filler.  

---  
One-shot Example  

Use the uploaded plush toy image as the product reference.  
Scene: calm morning nursery—soft pastel wall, blurred white crib bars and a gauzy muslin drape; natural window light from camera left.  
Composition: plush seated centered-right on a light knit blanket; large negative space upper-left for copy; shallow depth of field; single product only.  
Color: gentle pastels and warm neutrals; plush fabric texture sharp and true.  
In large overlay text, clear with no typos: “Hello, soft world.”  
Subline text below the main text: “Cuddly comfort for little beginnings.”  
Output: 4:5 with safe 9:16 alt.
"""


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


