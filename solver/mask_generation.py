import os
import json
import argparse
import torch
import sys

def parse_args():

    parser = argparse.ArgumentParser(description="Mask generation tool")

    parser.add_argument(
        "-mask", "--mask_path", 
        type=str, 
        default="",
        help="Path to the mask pt file"
    )
    

    parser.add_argument(
        "-conflict", "--conflict_node", 
        type=str, 
        default="",
        help="Path to the conflict node list JSON file"
    )
    

    parser.add_argument(
        "-false", "--false_node", 
        type=str, 
        default="",
        help="Path to the false node list JSON file"
    )
    

    parser.add_argument(
        "-output", "--output", 
        type=str, 
        default="",
        help="Output directory path (if empty, uses mask path directory)"
    )
    
    args = parser.parse_args()

    if not args.output:
        args.output = os.path.dirname(args.mask_path)
    
    return args

def load_mask_file(mask_path):

    try:
       
        mask_data = torch.load(mask_path, map_location='cpu')
       
        
        return mask_data
        
    except FileNotFoundError:
       
        return None
    except Exception as e:
       
        return None

def load_node_lists(conflict_path, false_path):
   
    conflict_nodes = []
    false_nodes = []
    
   
  
    if os.path.exists(conflict_path):
        with open(conflict_path, 'r') as f:
            conflict_nodes = json.load(f)
        

           
    
    # 加载false node列表
    
    if os.path.exists(false_path):
        with open(false_path, 'r') as f:
            false_nodes = json.load(f)
           
    
    return conflict_nodes, false_nodes

def count_true_false_percentage(data):

    true_count = 0
    false_count = 0
    total_count = 0
    
    if isinstance(data, torch.Tensor):

        true_count = torch.sum(data == True).item()
        false_count = torch.sum(data == False).item()
        total_count = data.numel()
    elif isinstance(data, dict):

        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                true_count += torch.sum(value == True).item()
                false_count += torch.sum(value == False).item()
                total_count += value.numel()
            elif isinstance(value, bool):
                if value:
                    true_count += 1
                else:
                    false_count += 1
                total_count += 1
    else:

        return 0, 0, 0, 0.0, 0.0
    

    true_percentage = (true_count / total_count * 100) if total_count > 0 else 0.0
    false_percentage = (false_count / total_count * 100) if total_count > 0 else 0.0
    
    return true_count, false_count, total_count, true_percentage, false_percentage

def create_mask_structure(mask_data):

    mask_structure = {}
    mask_structure_ture = {}
    for key, tensor in mask_data.items():
        if not isinstance(tensor, torch.Tensor):
            mask_structure[key] = tensor
            continue
        mask_structure[key] = tensor.clone()
        mask_structure_ture[key] = torch.ones_like(tensor, dtype=torch.bool)

    return mask_structure, mask_structure_ture

def parse_node_name(node_name):

    if node_name.startswith('m'):

        try:

            dot_pos = node_name.find('.')
            if dot_pos == -1:
                layer_str = node_name[1:]  
            else:
                layer_str = node_name[1:dot_pos]  
            
            layer_num = int(layer_str)
            
            
            base_key = layer_num * 7
            keys = [base_key + 4, base_key + 5, base_key + 6]  
            
            return {
                'type': 'm',
                'layer': layer_num,
                'keys': keys,
                'param_type': ['gate', 'up', 'down']  
            }
        except (ValueError, IndexError):
            return None
    
    elif node_name.startswith('a'):

        try:
            parts = node_name.split('.')
            if len(parts) != 3:
                return None
            

            layer_str = parts[0][1:]  
            layer_num = int(layer_str)
            
          
            h_part = parts[1]
            if h_part.startswith('h'):
                h_num = int(h_part[1:])  
                slice_start = h_num * 128
                slice_end = slice_start + 128
                slice_info = (slice_start, slice_end)
            elif h_part.startswith('H'):
                h_num = int(h_part[1:])  
                slice_start = h_num * 128
                slice_end = slice_start + 128
                slice_info = (slice_start, slice_end)
            else:
                return None
            
            param_type = parts[2]
            if param_type not in ['q', 'k', 'v', 'o']:
                return None
            

            param_to_key = {'q': 0, 'k': 1, 'v': 2, 'o': 3}
            key = layer_num * 7 + param_to_key[param_type]
            
            return {
                'type': 'a',
                'layer': layer_num,
                'key': key,
                'param_type': param_type,
                'slice_info': slice_info,
                'h_part': h_part
            }
            
        except (ValueError, IndexError) as e:
            return None
    
    else:
        return None

def apply_node_to_mask(mask_structure, node_name, mask_data):

    parsed = parse_node_name(node_name)
    if parsed is None:
        return
    
    if parsed['type'] == 'm':

        for key in parsed['keys']:
            key_str = int(key)
            mask_structure[key_str] = mask_data[key_str].clone()
          
    
    elif parsed['type'] == 'a':

        key_str = int(parsed['key'])
        slice_start, slice_end = parsed['slice_info']
        

        mask_structure[key_str][slice_start:slice_end, :] = mask_data[key_str][slice_start:slice_end, :]


def count_mask_percentage(mask_structure, mask_name):

    true_count = 0
    false_count = 0
    total_count = 0
    
    for key, tensor in mask_structure.items():
        if isinstance(tensor, torch.Tensor):
            true_count += torch.sum(tensor).item()
            false_count += torch.sum(~tensor).item()
            total_count += tensor.numel()
    
    true_percentage = (true_count / total_count * 100) if total_count > 0 else 0.0
    false_percentage = (false_count / total_count * 100) if total_count > 0 else 0.0
    

    
    return true_count, false_count, total_count, true_percentage, false_percentage

def save_mask_structure(mask_structure, file_path, mask_name):


       
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    
    torch.save(mask_structure, file_path)
    
        


def main():

    

    args = parse_args()
    

    print(f"  mask_path: {args.mask_path}")
    print(f"  conflict_node: {args.conflict_node}")
    print(f"  false_node: {args.false_node}")
    print(f"  output: {args.output}")
    
    if not os.path.exists(args.mask_path):

        return

    mask_data = load_mask_file(args.mask_path)
    if mask_data is None:

        return
    

    conflict_nodes, false_nodes = load_node_lists(args.conflict_node, args.false_node)
    



    true_count, false_count, total_count, true_percentage, false_percentage = count_true_false_percentage(mask_data)
    

    

    total_percentage = true_percentage + false_percentage

    
    
    safe_mask,safe_mask_ture = create_mask_structure(mask_data)
    conflict_mask,conflict_mask_ture = create_mask_structure(mask_data)
    
  
    
   
    for node in false_nodes:
        apply_node_to_mask(safe_mask, node, safe_mask_ture)
    
    for node in conflict_nodes:
        apply_node_to_mask(safe_mask, node, safe_mask_ture)
    
  
    for node in conflict_nodes:
        apply_node_to_mask(conflict_mask, node, conflict_mask_ture)
    
  
    

    safe_stats = count_mask_percentage(safe_mask, "safe_mask")
    conflict_stats = count_mask_percentage(conflict_mask, "conflict_mask")
    
   
    safe_mask_path = os.path.join(args.output, "safe_mask.pt")
    conflict_mask_path = os.path.join(args.output, "conflict_mask.pt")
    
   
    save_mask_structure(safe_mask, safe_mask_path, "safe_mask")
    save_mask_structure(conflict_mask, conflict_mask_path, "conflict_mask")
    
   

if __name__ == "__main__":
    main()
