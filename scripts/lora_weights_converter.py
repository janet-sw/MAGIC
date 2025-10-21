import torch
from safetensors.torch import load_file, save_file

def convert_keys(input_model_path, output_model_path):
    try:
        state_dict = load_file(input_model_path)
    except Exception as e:
        raise ValueError(f"Error loading file {input_model_path}: {e}")

    new_state_dict = {}

    # Convert each key
    for key, value in state_dict.items():
        # Remove "unet."
        new_key = key.replace("unet.", "")

        # Replace ".lora." with "_lora."
        new_key = new_key.replace(".lora.", "_lora.")
        new_key = new_key.replace("to_out.0_lora", "to_out_lora")

        # Insert ".processor." before "to_"
        parts = new_key.split(".")
        for i, part in enumerate(parts):
            if part.startswith("attn") and i + 1 < len(parts) and "to_" in parts[i + 1]:
                parts.insert(i + 1, "processor")
                break
        new_key = ".".join(parts)

        # Add updated key-value pair to the new state_dict
        new_state_dict[new_key] = value

    # Save the updated state_dict to .safetensors format
    try:
        save_file(new_state_dict, output_model_path)
        print(f"Converted model saved to {output_model_path}")
    except Exception as e:
        raise ValueError(f"Error saving file {output_model_path}: {e}")

if __name__ == "__main__":
    # Example usage
    input_model_path = "./models/lora_weights/fitzpatrick17k/pytorch_lora_weights.safetensors"  
    output_model_path = "./models/lora_weights/fitzpatrick17k/updated_pytorch_lora_weights.safetensors"  
    convert_keys(input_model_path, output_model_path)

