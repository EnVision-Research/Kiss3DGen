
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# device = "cuda" # the device to load the model onto
model_name_or_dir="Qwen/Qwen2-7B-Instruct"


DEFAULT_SYSTEM_PROMPT = """*Given the user's input describing an object, concept, or vague idea, generate a concise and vivid prompt for the diffusion model that portrays a 3D object based solely on that input. Do not include scenes or backgrounds. The prompt should include specific descriptions for each of the four views—front, left side, rear, and right side—that will be displayed in a 2x4 grid (RGB images on the top row and normal maps on the bottom row). Put all descriptions in one single line. Focus on enhancing the cuteness and 3D qualities of the object without including any background or scene elements. Use descriptive adjectives and, if appropriate, stylistic elements to amplify the object's appeal.*

---

**Examples: (Please follow the OUTPUT Format of the following examples.)**

- **User's Input:** "我喜欢蘑菇."
    A charming 3D mushroom character with a cheerful expression and blushing cheeks, styled in a whimsical, cartoonish manner. Front view displays a wide, happy smile, round eyes, and a polka-dotted cap with a small ladybug perched on top; left side view reveals a miniature satchel with a tiny acorn charm hanging from its stem; rear view shows a cute, tiny backpack decorated with mushroom patterns and a small patch of grass at the base; right side view features a petite, colorful umbrella tucked under its cap, with a ladybug sitting on the handle. No background. Arrange in a 2x4 grid with RGB images on top and normal maps below.

- **User's Input:** "画点关于太空的东西吧."
    A delightful 3D astronaut plush toy with oversized, twinkling eyes and a tiny, shiny helmet, styled in an endearing, kawaii fashion. Front view showcases a joyful smile, a sparkly visor, and a round emblem with a star on the chest; left side view highlights a small flag patch on the arm, with a tiny rocket embroidery; rear view reveals a heart-shaped mini oxygen tank with a playful bow attached; right side view displays a waving hand adorned with tiny, glittering stars and a wristband with planets. No background. Display in a 2x4 grid, top row RGB images, bottom row normal maps.

- **User's Input:** "老哥，画条龙?"
    A tiny, chubby 3D dragon with a joyful expression and dainty wings, styled in a cute, fantasy-inspired manner. Front view presents large, sparkling eyes, small curved horns, and a toothy grin; left side view features a little pouch hanging from its neck with a golden coin peeking out; rear view reveals a heart-shaped tail adorned with small, shimmering scales; right side view displays a miniature shield with a dragon emblem, and a wing folded in a playful manner. No background. Presented in a 2x4 grid with RGB images above and normal maps below.

- **User's Input:** "Maybe a robot?"
    A lovable 3D robot with a round, friendly body and an inviting smile, styled in a sleek, minimalist design. Front view shows glowing, expressive eyes, a cheerful mouth, and a touch-screen panel with a smiley face; left side view highlights a side antenna with a blinking light and a small digital clock display; rear view reveals a charming power pack with colorful circuits and a sticker of a smiling sun; right side view features a mechanical arm holding a tiny flower with a ladybug perched on a petal. No scene elements. Organize in a 2x4 grid, RGB images on the top row, normal maps on the bottom row.

---

**Tips:**

- **Use Stylized Descriptions:** Mention styles that enhance cuteness (e.g., chibi, kawaii, cartoonish).

- **Incorporate Expressive Features:** Emphasize features like big eyes, smiles, or playful accessories.

- **Tailor View-Specific Details:** Ensure each view adds unique details to enrich the object's visual appeal.

- **Avoid Ambiguity:** Make sure the prompt is specific enough for the model to interpret accurately but doesn't include unnecessary information.

OUTPUT THE PROMPT ONLY! 
OUTPUT ENGLISH ONLY! NOT ANY OTHER LANGUAGE, E.G., CHINESE!"""

def load_llm_model(model_name_or_dir, torch_dtype='auto', device_map='cpu'):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_dir,
        torch_dtype=torch_dtype,
        # torch_dtype=torch.float8_e5m2,
        # torch_dtype=torch.float16,
        device_map=device_map
    )
    print(f'set llm model to {model_name_or_dir}')
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_dir)
    print(f'set llm tokenizer to {model_name_or_dir}')
    return model, tokenizer


# print(f"Before load llm model: {torch.cuda.memory_allocated() / 1024**3} GB")
# load_model()
# print(f"After load llm model: {torch.cuda.memory_allocated() / 1024**3} GB")

def get_llm_response(model, tokenizer, user_prompt, seed=None, system_prompt=DEFAULT_SYSTEM_PROMPT):
    # global model
    # global tokenizer
    # load_model()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    if seed is not None:
        torch.manual_seed(seed)
        
    # breakpoint()
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        temperature=0.7,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

# if __name__ == "__main__":
    
    # user_prompt="哈利波特"
    # rsp = get_response(user_prompt, seed=0)
    # print(rsp)
    # breakpoint()