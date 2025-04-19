from diffusers.models.attention_processor import FluxAttnProcessor2_0
from safetensors import safe_open
import re
import torch
from .layers_cache import MultiDoubleStreamBlockLoraProcessor, MultiSingleStreamBlockLoraProcessor
from src.models.lora_blocks import LoRALinearLayer, MultiLoraBlock  # Keep existing EasyControl logic
device = "cuda"

def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def get_lora_rank(checkpoint):
    if checkpoint is None:
        return 8  # or 4, 16, whatever rank you want as default
    for k in checkpoint.keys():
        if k.endswith(".down.weight"):
            return checkpoint[k].shape[0]
    raise ValueError("Could not infer LoRA rank from checkpoint.")

def load_checkpoint(local_path):
    if local_path is not None:
        if '.safetensors' in local_path:
            print(f"Loading .safetensors checkpoint from {local_path}")
            checkpoint = load_safetensors(local_path)
        else:
            print(f"Loading checkpoint from {local_path}")
            checkpoint = torch.load(local_path, map_location='cpu')
    return checkpoint

def update_model_with_lora_sanasprint(transformer, rank=8, lora_weights=[0.0], cond_size=512):
    """
    Inject LoRA adapters into SANA-Sprint Attention modules' QKV projections.
    """
    print("🔧 Injecting EasyControl LoRA into SANA-Sprint transformer...")

    n_loras = len(lora_weights)

    for name, module in transformer.named_modules():
        if re.match(r"transformer_blocks\.\d+\.attn[12]$", name):
            print(f"🔍 Found attention block: {name}")
            
            for proj_name in ["to_q", "to_k", "to_v"]:
                full_proj_name = f"{name}.{proj_name}"
                proj_module = dict(module.named_children())[proj_name]
                
                # Wrap the Linear layer with EasyControl-style LoRALinearLayer
                wrapped = LoRALinearLayer(
                    proj_module,
                    rank=rank,
                    lora_alpha=rank,
                    n_loras=n_loras,
                    cond_dim=cond_size
                )

                # Replace in-place
                setattr(module, proj_name, wrapped)
                print(f"Injected LoRA into {full_proj_name}")

    print("Finished LoRA injection.\n")

def update_model_with_lora(checkpoint, lora_weights, transformer, cond_size):
        number = len(lora_weights)
        ranks = [get_lora_rank(checkpoint) for _ in range(number)]
        lora_attn_procs = {}
        double_blocks_idx = list(range(19))
        single_blocks_idx = list(range(38))
        for name, attn_processor in transformer.attn_processors.items():
            match = re.search(r'\.(\d+)\.', name)
            if match:
                layer_index = int(match.group(1))
            
            if name.startswith("transformer_blocks") and layer_index in double_blocks_idx:
                
                lora_state_dicts = {}

                if checkpoint is not None:
                    for key, value in checkpoint.items():
                        # Match based on the layer index in the key (assuming the key contains layer index)
                        if re.search(r'\.(\d+)\.', key):
                            layer_idx = int(re.search(r'\.(\d+)\.', key).group(1))
                            if layer_idx not in lora_state_dicts:
                                lora_state_dicts[layer_idx] = {}
                            lora_state_dicts[layer_idx][key] = value

                
                lora_attn_procs[name] = MultiDoubleStreamBlockLoraProcessor(
                    dim=3072, ranks=ranks, network_alphas=ranks, lora_weights=lora_weights, device=device, dtype=torch.bfloat16, cond_width=cond_size, cond_height=cond_size, n_loras=number
                )
                
                # Load the weights from the checkpoint dictionary into the corresponding layers
                for n in range(number):
                    q_down = lora_state_dicts.get(f'{name}.q_loras.{n}.down.weight', None)
                    q_up   = lora_state_dicts.get(f'{name}.q_loras.{n}.up.weight', None)
                    k_down = lora_state_dicts.get(f'{name}.k_loras.{n}.down.weight', None)
                    k_up   = lora_state_dicts.get(f'{name}.k_loras.{n}.up.weight', None)
                    v_down = lora_state_dicts.get(f'{name}.v_loras.{n}.down.weight', None)
                    v_up   = lora_state_dicts.get(f'{name}.v_loras.{n}.up.weight', None)
                
                    if q_down is not None:
                        lora_attn_procs[name].q_loras[n].down.weight.data = q_down
                    if q_up is not None:
                        lora_attn_procs[name].q_loras[n].up.weight.data   = q_up
                    if k_down is not None:
                        lora_attn_procs[name].k_loras[n].down.weight.data = k_down
                    if k_up is not None:
                        lora_attn_procs[name].k_loras[n].up.weight.data   = k_up
                    if v_down is not None:
                        lora_attn_procs[name].v_loras[n].down.weight.data = v_down
                    if v_up is not None:
                        lora_attn_procs[name].v_loras[n].up.weight.data   = v_up

                
            elif name.startswith("single_transformer_blocks") and layer_index in single_blocks_idx:
                
                lora_state_dicts = {}
                for key, value in checkpoint.items():
                    # Match based on the layer index in the key (assuming the key contains layer index)
                    if re.search(r'\.(\d+)\.', key):
                        checkpoint_layer_index = int(re.search(r'\.(\d+)\.', key).group(1))
                        if checkpoint_layer_index == layer_index and key.startswith("single_transformer_blocks"):
                            lora_state_dicts[key] = value
                
                lora_attn_procs[name] = MultiSingleStreamBlockLoraProcessor(
                    dim=3072, ranks=ranks, network_alphas=ranks, lora_weights=lora_weights, device=device, dtype=torch.bfloat16, cond_width=cond_size, cond_height=cond_size, n_loras=number
                )
                # Load the weights from the checkpoint dictionary into the corresponding layers
                for n in range(number):
                    lora_attn_procs[name].q_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.q_loras.{n}.down.weight', None)
                    lora_attn_procs[name].q_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.q_loras.{n}.up.weight', None)
                    lora_attn_procs[name].k_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.k_loras.{n}.down.weight', None)
                    lora_attn_procs[name].k_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.k_loras.{n}.up.weight', None)
                    lora_attn_procs[name].v_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.v_loras.{n}.down.weight', None)
                    lora_attn_procs[name].v_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.v_loras.{n}.up.weight', None)
                    lora_attn_procs[name].to(device)
            else:
                lora_attn_procs[name] = FluxAttnProcessor2_0()

        transformer.set_attn_processor(lora_attn_procs)
        

def update_model_with_multi_lora(checkpoints, lora_weights, transformer, cond_size):
        ck_number = len(checkpoints)
        cond_lora_number = [len(ls) for ls in lora_weights]
        cond_number = sum(cond_lora_number)
        ranks = [get_lora_rank(checkpoint) for checkpoint in checkpoints]
        multi_lora_weight = []
        for ls in lora_weights:
            for n in ls:
                multi_lora_weight.append(n)
        
        lora_attn_procs = {}
        double_blocks_idx = list(range(19))
        single_blocks_idx = list(range(38))
        for name, attn_processor in transformer.attn_processors.items():
            match = re.search(r'\.(\d+)\.', name)
            if match:
                layer_index = int(match.group(1))
            
            if name.startswith("transformer_blocks") and layer_index in double_blocks_idx:
                lora_state_dicts = [{} for _ in range(ck_number)]
                for idx, checkpoint in enumerate(checkpoints):
                    for key, value in checkpoint.items():
                        # Match based on the layer index in the key (assuming the key contains layer index)
                        if re.search(r'\.(\d+)\.', key):
                            checkpoint_layer_index = int(re.search(r'\.(\d+)\.', key).group(1))
                            if checkpoint_layer_index == layer_index and key.startswith("transformer_blocks"):
                                lora_state_dicts[idx][key] = value
                
                lora_attn_procs[name] = MultiDoubleStreamBlockLoraProcessor(
                    dim=3072, ranks=ranks, network_alphas=ranks, lora_weights=multi_lora_weight, device=device, dtype=torch.bfloat16, cond_width=cond_size, cond_height=cond_size, n_loras=cond_number
                )
                
                # Load the weights from the checkpoint dictionary into the corresponding layers
                num = 0
                for idx in range(ck_number):
                    for n in range(cond_lora_number[idx]):
                        lora_attn_procs[name].q_loras[num].down.weight.data = lora_state_dicts[idx].get(f'{name}.q_loras.{n}.down.weight', None)
                        lora_attn_procs[name].q_loras[num].up.weight.data = lora_state_dicts[idx].get(f'{name}.q_loras.{n}.up.weight', None)
                        lora_attn_procs[name].k_loras[num].down.weight.data = lora_state_dicts[idx].get(f'{name}.k_loras.{n}.down.weight', None)
                        lora_attn_procs[name].k_loras[num].up.weight.data = lora_state_dicts[idx].get(f'{name}.k_loras.{n}.up.weight', None)
                        lora_attn_procs[name].v_loras[num].down.weight.data = lora_state_dicts[idx].get(f'{name}.v_loras.{n}.down.weight', None)
                        lora_attn_procs[name].v_loras[num].up.weight.data = lora_state_dicts[idx].get(f'{name}.v_loras.{n}.up.weight', None)
                        lora_attn_procs[name].proj_loras[num].down.weight.data = lora_state_dicts[idx].get(f'{name}.proj_loras.{n}.down.weight', None)
                        lora_attn_procs[name].proj_loras[num].up.weight.data = lora_state_dicts[idx].get(f'{name}.proj_loras.{n}.up.weight', None)
                        lora_attn_procs[name].to(device)
                        num += 1
                
            elif name.startswith("single_transformer_blocks") and layer_index in single_blocks_idx:
                
                lora_state_dicts = [{} for _ in range(ck_number)]
                for idx, checkpoint in enumerate(checkpoints):
                    for key, value in checkpoint.items():
                        # Match based on the layer index in the key (assuming the key contains layer index)
                        if re.search(r'\.(\d+)\.', key):
                            checkpoint_layer_index = int(re.search(r'\.(\d+)\.', key).group(1))
                            if checkpoint_layer_index == layer_index and key.startswith("single_transformer_blocks"):
                                lora_state_dicts[idx][key] = value
                
                lora_attn_procs[name] = MultiSingleStreamBlockLoraProcessor(
                    dim=3072, ranks=ranks, network_alphas=ranks, lora_weights=multi_lora_weight, device=device, dtype=torch.bfloat16, cond_width=cond_size, cond_height=cond_size, n_loras=cond_number
                )
                # Load the weights from the checkpoint dictionary into the corresponding layers
                num = 0
                for idx in range(ck_number):
                    for n in range(cond_lora_number[idx]):
                        lora_attn_procs[name].q_loras[num].down.weight.data = lora_state_dicts[idx].get(f'{name}.q_loras.{n}.down.weight', None)
                        lora_attn_procs[name].q_loras[num].up.weight.data = lora_state_dicts[idx].get(f'{name}.q_loras.{n}.up.weight', None)
                        lora_attn_procs[name].k_loras[num].down.weight.data = lora_state_dicts[idx].get(f'{name}.k_loras.{n}.down.weight', None)
                        lora_attn_procs[name].k_loras[num].up.weight.data = lora_state_dicts[idx].get(f'{name}.k_loras.{n}.up.weight', None)
                        lora_attn_procs[name].v_loras[num].down.weight.data = lora_state_dicts[idx].get(f'{name}.v_loras.{n}.down.weight', None)
                        lora_attn_procs[name].v_loras[num].up.weight.data = lora_state_dicts[idx].get(f'{name}.v_loras.{n}.up.weight', None)
                        lora_attn_procs[name].to(device)
                        num += 1

            else:
                lora_attn_procs[name] = FluxAttnProcessor2_0()

        transformer.set_attn_processor(lora_attn_procs)


def set_single_lora(transformer, local_path, lora_weights=[], cond_size=512):
    checkpoint = load_checkpoint(local_path)
    update_model_with_lora(checkpoint, lora_weights, transformer, cond_size)
   
def set_multi_lora(transformer, local_paths, lora_weights=[[]], cond_size=512):
    checkpoints = [load_checkpoint(local_path) for local_path in local_paths]
    update_model_with_multi_lora(checkpoints, lora_weights, transformer, cond_size)

def unset_lora(transformer):
    lora_attn_procs = {}
    for name, attn_processor in transformer.attn_processors.items():
        lora_attn_procs[name] = FluxAttnProcessor2_0()
    transformer.set_attn_processor(lora_attn_procs)


'''
unset_lora(pipe.transformer)
lora_path = "./lora.safetensors"
lora_weights = [1, 1]
set_lora(pipe.transformer, local_path=lora_path, lora_weights=lora_weights, cond_size=512)
'''
