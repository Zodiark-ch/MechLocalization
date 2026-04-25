import os
import json
import argparse
import torch
import sys

def parse_args():

    parser = argparse.ArgumentParser(description="Mask generation tool")
    
    # mask path参数
    parser.add_argument(
        "-mask1", "--mask_path1", 
        type=str, 
        default="",
        help="Path to the mask pt file"
    )
    
    parser.add_argument(
        "-mask2", "--mask_path2", 
        type=str, 
        default="",
        help="Path to the mask pt file"
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
        
        if isinstance(mask_data, torch.Tensor):
            
        elif isinstance(mask_data, dict):
            for key, value in mask_data.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key}: Tensor of shape {value.shape}, dtype {value.dtype}")
                    
        
        return mask_data
        

def save_mask_structure(mask_structure, file_path, mask_name):

    try:
       
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
       
        torch.save(mask_structure, file_path)
        
        
    except Exception as e:


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

def main():
    args = parse_args()
    mask_data1 = load_mask_file(args.mask_path1)
    mask_data2 = load_mask_file(args.mask_path2)
    
    print(f"mask_data1: {mask_data1}")
    print(f"mask_data2: {mask_data2}")
    
    
    if len(mask_data1) != len(mask_data2):
        return
    
    

    for i in range(len(mask_data1)):

        if (i + 1) % 7 != 0 and (i + 1) % 7 != 5 and (i + 1) % 7 != 6:

            mask_data1[i] = mask_data2[i]
            
    
    true_count, false_count, total_count, true_percentage, false_percentage = count_true_false_percentage(mask_data1)
    

    

    total_percentage = true_percentage + false_percentage

        
    baseline_mask_path = os.path.join(args.output, "")

    

    save_mask_structure(mask_data1, baseline_mask_path, "")
    
if __name__ == "__main__":
    main()