import torch
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images
from PIL import Image

# specify the path to the model
model_path = "deepseek-ai/deepseek-vl2-tiny"
vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# Single image example using your screenshot.png
conversation = [
    {
        "role": "<|User|>",
        "content": "what url is shown for WaveForms AI in this screenshot? Please only give me the url.",
        "images": ["test.png"]  # Your local image path
    },
    {"role": "<|Assistant|>", "content": ""}
]

# Load image (ensure it exists in your working directory)
try:
    pil_images = load_pil_images(conversation)
except FileNotFoundError:
    # Fallback to direct PIL loading if the deepseek utility fails
    pil_images = [Image.open("screenshot.png").convert("RGB")]

# Prepare inputs
prepare_inputs = vl_chat_processor(
    conversations=conversation,
    images=pil_images,
    force_batchify=True,
    system_prompt=""
).to(vl_gpt.device)

# Generate response
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
outputs = vl_gpt.language.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True
)

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
print("Model response:", answer)