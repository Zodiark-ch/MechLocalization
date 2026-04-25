import os
import json
import argparse
import sys
import re
sys.path.append(
    os.path.join(
        os.getcwd(),
        "src/solver/"
    )
)

class Circuit:
    def __init__(self, json_file_path, leaf_nodes=None):
        
        self.circuit = {}
        self.NOR = False 
        self.load_circuit(json_file_path, leaf_nodes)
    
    def load_circuit(self, json_file_path, leaf_nodes=None):

        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            

            known_node_list = set(leaf_nodes) if leaf_nodes else set()
            

            to_node_dependencies = {}
            for item in data:
                if len(item) == 3:
                    from_node, to_node, node_type = item
                    if to_node not in to_node_dependencies:
                        to_node_dependencies[to_node] = []
                    to_node_dependencies[to_node].append((from_node, node_type))
            

            while to_node_dependencies:

                ready_to_process = []
                for to_node, dependencies in to_node_dependencies.items():
                    all_dependencies_met = True
                    for from_node, _ in dependencies:
                        if from_node not in known_node_list:
                            all_dependencies_met = False
                            break
                    
                    if all_dependencies_met:
                        ready_to_process.append(to_node)
                


                

                for to_node in ready_to_process:
                    dependencies = to_node_dependencies[to_node]
                    

                    if to_node not in self.circuit:
                        self.circuit[to_node] = {}
                    
                    for from_node, node_type in dependencies:
                        if node_type not in self.circuit[to_node]:
                            self.circuit[to_node][node_type] = []
                        self.circuit[to_node][node_type].append(from_node)
                    

                    known_node_list.add(to_node)
                    

                    del to_node_dependencies[to_node]
                    
        
    
    def bool_validation(self, VALUE_retain, VALUE_forget, input_retain=None, input_forget=None, NOR=None):
       
        if NOR is None:
            NOR = self.NOR
        
        
        real_time_values = {}
        

        all_nodes = set()
        for to_node, type_groups in self.circuit.items():
            all_nodes.add(to_node)
            for from_nodes in type_groups.values():
                all_nodes.update(from_nodes)
        

        for node in all_nodes:
            node_value = None
            
            if NOR:

                for n, value in VALUE_forget:
                    if n == node:
                        node_value = value
                        break
            else:

                for n, value in VALUE_retain:
                    if n == node:
                        node_value = value
                        break
            
            real_time_values[node] = node_value
        

        if NOR:

            for node, value in input_forget:
                if node in real_time_values:
                    real_time_values[node] = value
        else:

            if input_retain is not None:
                for node, value in input_retain:
                    if node in real_time_values:
                        real_time_values[node] = value
        
        

        for to_node, type_groups in self.circuit.items():
            for node_type, from_nodes in type_groups.items():

                from_node_values = []
                for from_node in from_nodes:
                    from_node_value = real_time_values.get(from_node)
                    if from_node_value is None:

                        return False, []
                    from_node_values.append(from_node_value)

                if NOR:

                    inverted_from_values = [not val for val in from_node_values]
                    
                    if node_type in ["OR", "ADDER"]:
                       
                        compute_result = all(inverted_from_values)
                    elif node_type == "AND":
                  
                        compute_result = any(inverted_from_values)
                    else:
            
                        return False, []
                else:
                  
                    if node_type in ["AND", "ADDER"]:
                    
                        compute_result = all(from_node_values)
                    elif node_type == "OR":
                    
                        compute_result = any(from_node_values)
                    else:
                     
                        return False, []
                
           
                to_node_value = real_time_values.get(to_node)
                
                if to_node_value is None:
                  
                    if NOR:
                        real_time_values[to_node] = not compute_result
                    else:
                        real_time_values[to_node] = compute_result
                else:
                  
                    if NOR:
                        if compute_result == to_node_value:
                            return False, []
                    else:
                        if compute_result != to_node_value:
                            return False, []
        

        
        return True, real_time_values

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-circuit1", "--circuit1", type=str, default="")
    parser.add_argument("-circuit2", "--circuit2", type=str, default="")
    parser.add_argument("-circuit3", "--circuit3", type=str, default="")
    parser.add_argument("-circuit4", "--circuit4", type=str, default="")
    parser.add_argument("-circuit5", "--circuit5", type=str, default="")
    parser.add_argument("-circuit6", "--circuit6", type=str, default="")
    parser.add_argument("-circuit7", "--circuit7", type=str, default="")
    parser.add_argument("-w", "--with_embedding_nodes", action="store_true", default=True)
    
    args = parser.parse_args()
    
    return args




def initialize_values(args, leaf_nodes_by_circuit):

    VALUE_retain = []
    VALUE_forget = []
    input_retain = []
    input_forget = []
    

    circuit_files = []
    for i in range(1, 7):
        circuit_attr = f"circuit{i}"
        if hasattr(args, circuit_attr):
            circuit_path = getattr(args, circuit_attr)
            if circuit_path and os.path.exists(circuit_path):
                circuit_files.append((i, circuit_path))
    
    if not circuit_files:
        return VALUE_retain, VALUE_forget, input_retain, input_forget
    

    

    for circuit_num, circuit_path in circuit_files:
        try:

            with open(circuit_path, 'r') as f:
                data = json.load(f)
            

            circuit_nodes = set()
            for item in data:
                if len(item) == 3:
                    from_node, to_node, node_type = item
                    circuit_nodes.add(from_node)
                    circuit_nodes.add(to_node)
            
            circuit_nodes_list = list(circuit_nodes)
            print(f"Circuit{circuit_num} ({circuit_path}) 包含 {len(circuit_nodes_list)} 个节点")
            

            if circuit_num == 1:

                for node in circuit_nodes_list:
                    VALUE_forget.append([node, False])
              
            else:

                for node in circuit_nodes_list:
                    VALUE_retain.append([node, True])

                
        except Exception as e:

            continue
    

    

    if 1 in leaf_nodes_by_circuit:
        circuit1_leaf_nodes = leaf_nodes_by_circuit[1]
        for node in circuit1_leaf_nodes:
            input_forget.append([node, False])



    all_other_leaf_nodes = set()
    for circuit_num in range(2, 7):  
        if circuit_num in leaf_nodes_by_circuit:
            all_other_leaf_nodes.update(leaf_nodes_by_circuit[circuit_num])
    
    for node in all_other_leaf_nodes:
        input_retain.append([node, True])

    

    seen_nodes = set()
    unique_VALUE_retain = []
    for node, value in VALUE_retain:
        if node not in seen_nodes:
            seen_nodes.add(node)
            unique_VALUE_retain.append([node, value])
    VALUE_retain = unique_VALUE_retain

    

    seen_nodes = set()
    unique_VALUE_forget = []
    for node, value in VALUE_forget:
        if node not in seen_nodes:
            seen_nodes.add(node)
            unique_VALUE_forget.append([node, value])
    VALUE_forget = unique_VALUE_forget

    

    seen_nodes = set()
    unique_input_retain = []
    for node, value in input_retain:
        if node not in seen_nodes:
            seen_nodes.add(node)
            unique_input_retain.append([node, value])
    input_retain = unique_input_retain

    

    seen_nodes = set()
    unique_input_forget = []
    for node, value in input_forget:
        if node not in seen_nodes:
            seen_nodes.add(node)
            unique_input_forget.append([node, value])
    input_forget = unique_input_forget

    
    return VALUE_retain, VALUE_forget, input_retain, input_forget



def modify(VALUE_retain, VALUE_forget, input_retain, input_forget, circuits):

    import copy
    

    VALUE_retain = copy.deepcopy(VALUE_retain)
    VALUE_forget = copy.deepcopy(VALUE_forget)
    input_retain = copy.deepcopy(input_retain)
    input_forget = copy.deepcopy(input_forget)
    

    


    for i in range(len(VALUE_forget)):
        VALUE_forget[i][1] = None
    for i in range(len(VALUE_retain)):
        VALUE_retain[i][1] = None
    

    circuit1_nodes = set()
    other_circuit_nodes = set()
    

    if 1 in circuits:
        circuit1 = circuits[1]
        for to_node, type_groups in circuit1.circuit.items():
            circuit1_nodes.add(to_node)
            for from_nodes in type_groups.values():
                circuit1_nodes.update(from_nodes)
    

    for circuit_num, circuit in circuits.items():
        if circuit_num != 1:
            for to_node, type_groups in circuit.circuit.items():
                other_circuit_nodes.add(to_node)
                for from_nodes in type_groups.values():
                    other_circuit_nodes.update(from_nodes)

    if 1 in circuits:
        circuit1 = circuits[1]
        processed_nodes = set()
        
        def process_circuit1_node(node):

            if node in processed_nodes:
                return
            
            processed_nodes.add(node)
            
            if node in circuit1.circuit:
                for node_type, from_nodes in circuit1.circuit[node].items():
                    if node_type in ["OR", "ADDER"]:
                        for from_node in from_nodes:

                            from_node_value = None
                            for i, (n, v) in enumerate(VALUE_forget):
                                if n == from_node:
                                    from_node_value = v
                                    break
                            
                            if from_node_value is None:

                                for i, (n, v) in enumerate(VALUE_forget):
                                    if n == from_node:
                                        VALUE_forget[i][1] = False
                                        break




                            

                            process_circuit1_node(from_node)
        

        resid_post_nodes = []
        for to_node in circuit1.circuit.keys():
            if "resid_post" in to_node:
                for i, (n, v) in enumerate(VALUE_forget):
                                    if n == 'resid_post':
                                        VALUE_forget[i][1] = False
                                        break
                resid_post_nodes.append(to_node)
        
        for resid_post in resid_post_nodes:

            process_circuit1_node(resid_post)
    

    for circuit_num, circuit in circuits.items():
        if circuit_num != 1:
            processed_nodes = set()
            
            def process_other_circuit_node(node):

                if node in processed_nodes:
                    return
                
                processed_nodes.add(node)
                
                if node in circuit.circuit:
                    for node_type, from_nodes in circuit.circuit[node].items():
                        if node_type in ["AND", "ADDER"]:
                            for from_node in from_nodes:

                                from_node_value = None
                                for i, (n, v) in enumerate(VALUE_retain):
                                    if n == from_node:
                                        from_node_value = v
                                        break
                                
                                if from_node_value is None:
  
                                    for i, (n, v) in enumerate(VALUE_retain):
                                        if n == from_node:
                                            VALUE_retain[i][1] = True
                                            break


                                process_other_circuit_node(from_node)
            

            resid_post_nodes = []
            for to_node in circuit.circuit.keys():
                if "resid_post" in to_node:
                    for i, (n, v) in enumerate(VALUE_retain):
                                        if n == "resid_post":
                                            VALUE_retain[i][1] = True
                                            break
                    resid_post_nodes.append(to_node)
            
            for resid_post in resid_post_nodes:

                process_other_circuit_node(resid_post)
  
    for i in range(len(input_retain)):
        input_retain[i][1] = None
    

    for node, value in VALUE_retain:
        if value is not None:
            for i, (n, v) in enumerate(input_retain):
                if n == node:
                    input_retain[i][1] = value

                    break

    for i in range(len(input_forget)):
        input_forget[i][1] = None
    

    for node, value in VALUE_forget:
        if value is not None:
            for i, (n, v) in enumerate(input_forget):
                if n == node:
                    input_forget[i][1] = value

                    break

    none_count_retain = sum(1 for _, v in VALUE_retain if v is None)
    none_count_forget = sum(1 for _, v in VALUE_forget if v is None)
    none_count_input_retain = sum(1 for _, v in input_retain if v is None)
    none_count_input_forget = sum(1 for _, v in input_forget if v is None)


    uncertain_node_list_retain = []
    uncertain_node_list_forget = []
    
    for node, value in input_retain:
        if value is None:
            uncertain_node_list_retain.append(node)
    
    for node, value in input_forget:
        if value is None:
            uncertain_node_list_forget.append(node)
    
    return VALUE_retain, VALUE_forget, input_retain, input_forget, uncertain_node_list_retain, uncertain_node_list_forget


def investigate_leaf_nodes(circuit_files):

    leaf_nodes_by_circuit = {}
    
    for circuit_num, circuit_path in circuit_files:
        try:
           
            with open(circuit_path, 'r') as f:
                data = json.load(f)
            

            to_nodes = set()
            from_nodes = set()
            
            for item in data:
                if len(item) == 3:
                    from_node, to_node, node_type = item
                    to_nodes.add(to_node)
                    from_nodes.add(from_node)

            leaf_nodes = from_nodes - to_nodes
            leaf_nodes_list = list(leaf_nodes)
            
            leaf_nodes_by_circuit[circuit_num] = leaf_nodes_list
            

            
        except Exception as e:

            leaf_nodes_by_circuit[circuit_num] = []
    
    return leaf_nodes_by_circuit


def main():

    args = parse_args()
    

    circuits = {}
    circuit_files = []
    
   
    for i in range(1, 7):  
        circuit_attr = f"circuit{i}"
        if hasattr(args, circuit_attr):
            circuit_path = getattr(args, circuit_attr)
            if circuit_path and os.path.exists(circuit_path):
                circuit_files.append((i, circuit_path))
    
    if not circuit_files:

        return
    

    leaf_nodes_by_circuit = investigate_leaf_nodes(circuit_files)
    

    for circuit_num, circuit_path in circuit_files:
        try:

            leaf_nodes = leaf_nodes_by_circuit.get(circuit_num, [])
            
            circuit = Circuit(circuit_path, leaf_nodes)
            
            
            if circuit_num == 1:
                circuit.NOR = True

            else:
                circuit.NOR = False

            
            circuits[circuit_num] = circuit
           
            
        except Exception as e:
         
            continue
    
    if not circuits:
  
        return
    
    
    VALUE_retain, VALUE_forget, input_retain, input_forget = initialize_values(args, leaf_nodes_by_circuit)
    
    
    
    
   
    initial_validations = {}
    has_validation_failures = False
    
    for circuit_num, circuit in circuits.items():
        is_valid, _ = circuit.bool_validation(VALUE_retain, VALUE_forget, input_retain, input_forget)
        initial_validations[circuit_num] = is_valid
        if not is_valid:
            has_validation_failures = True
           
    
   
    
    
    VALUE_retain, VALUE_forget, input_retain, input_forget, uncertain_node_list_retain, uncertain_node_list_forget = modify(
        VALUE_retain, VALUE_forget, input_retain, input_forget, circuits
    )
    
   
    import itertools
    import copy
    
   
    all_result_lists = {
        'circuit1': [],
        'circuit2': [],
        'circuit3': [],
        'circuit4': [],
        'circuit5': [],
        'circuit6': [],
        'circuit7': []
    }
    

    if uncertain_node_list_forget:
        
        N = len(uncertain_node_list_forget)
        total_combinations = 2 ** N

        
        for i, combination in enumerate(itertools.product([0, 1], repeat=N)):
            
            
           
            temp_input_forget = copy.deepcopy(input_forget)
            
            
            for j, node in enumerate(uncertain_node_list_forget):
                for k, (n, v) in enumerate(temp_input_forget):
                    if n == node:
                        temp_input_forget[k][1] = bool(combination[j])
                        break
            
           
            if 1 in circuits:
                circuit1 = circuits[1]
                is_valid, result_list = circuit1.bool_validation(VALUE_retain, VALUE_forget, input_retain, temp_input_forget)
                
                if is_valid:
                    
                    all_result_lists['circuit1'].append({
                        'type': 'forget',
                        'combination': combination,
                        'nodes': uncertain_node_list_forget,
                        'result_lists': result_list
                    })
                
        if 1 in circuits:
            circuit1 = circuits[1]
            is_valid, result_list = circuit1.bool_validation(VALUE_retain, VALUE_forget, input_retain, input_forget)
            
            if is_valid:
               
                all_result_lists['circuit1'].append({
                    'type': 'forget',
                    'combination': (),  
                    'nodes': [], 
                    'result_lists': result_list
                })

    

    if uncertain_node_list_retain:
      
        M = len(uncertain_node_list_retain)
        total_combinations = 2 ** M
       
        
        for i, combination in enumerate(itertools.product([0, 1], repeat=M)):
          
            
           
            temp_input_retain = copy.deepcopy(input_retain)
            
          
            for j, node in enumerate(uncertain_node_list_retain):
                for k, (n, v) in enumerate(temp_input_retain):
                    if n == node:
                        temp_input_retain[k][1] = bool(combination[j])
                        break
            
      
            all_other_circuits_valid = True
            result_lists = {}
            
            for circuit_num, circuit in circuits.items():
                if circuit_num != 1:  
                    is_valid, result_list = circuit.bool_validation(VALUE_retain, VALUE_forget, temp_input_retain, input_forget)
                    
                    if is_valid:
                        result_lists[circuit_num] = result_list
                    else:
                        all_other_circuits_valid = False

                        break
            
            if all_other_circuits_valid:

                for circuit_num, result_list in result_lists.items():
                    circuit_key = f'circuit{circuit_num}'
                    all_result_lists[circuit_key].append({
                        'type': 'retain',
                        'combination': combination,
                        'nodes': uncertain_node_list_retain,
                        'result_lists': result_list
                    })
    else:

        all_other_circuits_valid = True
        result_lists = {}
        
        for circuit_num, circuit in circuits.items():
            if circuit_num != 1:
                is_valid, result_list = circuit.bool_validation(VALUE_retain, VALUE_forget, input_retain, input_forget)
                
                if is_valid:
                    result_lists[circuit_num] = result_list
                else:
                    all_other_circuits_valid = False

                    break
        
        if all_other_circuits_valid:
           
            for circuit_num, result_list in result_lists.items():
                circuit_key = f'circuit{circuit_num}'
                all_result_lists[circuit_key].append({
                    'type': 'retain',
                    'combination': (),  
                    'nodes': [],  
                    'result_lists': result_list
                })
    
    
    total_combinations = 0
    for circuit_key, combinations in all_result_lists.items():
      
        total_combinations += len(combinations)

    

    
    def merge_result_lists(result_lists_dict):
       
        merged_result = {}
        all_nodes = set()
        

        for result_list in result_lists_dict.values():
            all_nodes.update(result_list.keys())
        
        for node in all_nodes:
            values = []
            for circuit_num, result_list in result_lists_dict.items():
                if node in result_list:
                    values.append(result_list[node])
                else:
                    values.append(None)
            
     
            non_none_values = [v for v in values if v is not None]
            
           
            
            if len(non_none_values) < len(values):
  
                unique_values = set(non_none_values)
               
                merged_result[node] = non_none_values[0]
            else:
                
                unique_values = set(values)
                
                
                merged_result[node] = values[0]
        
        return merged_result
    
   
    forget_all_node = []
    for combo in all_result_lists['circuit1']:
        forget_all_node.append(combo['result_lists'])
    

    

    retain_all_node = []
    

    other_circuit_combinations = {}
    for circuit_key, combinations in all_result_lists.items():
        if circuit_key != 'circuit1':
            circuit_num = int(circuit_key.replace('circuit', ''))
            other_circuit_combinations[circuit_num] = combinations
    
    if other_circuit_combinations:
 
        valid_circuit_combinations = {}
        for circuit_num, combinations in other_circuit_combinations.items():
            if len(combinations) > 0:
                valid_circuit_combinations[circuit_num] = combinations
        
        if valid_circuit_combinations:

            circuit_nums = sorted(valid_circuit_combinations.keys())
            combination_counts = [len(valid_circuit_combinations[circuit_num]) for circuit_num in circuit_nums]
            

            for combo_indices in itertools.product(*[range(count) for count in combination_counts]):
                try:
  
                    result_lists_dict = {}
                    for i, circuit_num in enumerate(circuit_nums):
                        combo_idx = combo_indices[i]
                        if combo_idx < len(valid_circuit_combinations[circuit_num]):
                            result_lists_dict[circuit_num] = valid_circuit_combinations[circuit_num][combo_idx]['result_lists']
                    
      
                    merged_result = merge_result_lists(result_lists_dict)
                    retain_all_node.append(merged_result)
                    
                except ValueError as e:
             
                    continue
       
    if len(forget_all_node) == 0:

        return circuits, VALUE_retain, VALUE_forget, input_retain, input_forget, leaf_nodes_by_circuit, all_result_lists, [], [], None, None, [], [], []
    
    if len(retain_all_node) == 0:

        return circuits, VALUE_retain, VALUE_forget, input_retain, input_forget, leaf_nodes_by_circuit, all_result_lists, forget_all_node, [], None, None, [], [], []
    
    
    
    def calculate_hamming_distance(result_list1, result_list2):

        nodes1 = set(result_list1.keys())
        nodes2 = set(result_list2.keys())
        common_nodes = nodes1.intersection(nodes2)
        
        hamming_distance = 0
        for node in common_nodes:

            if result_list1[node] != result_list2[node]:
                hamming_distance += 1
        
        return hamming_distance
    

    hamming_distances = {}
    best_forget_idx = None
    best_retain_idx = None
    min_distance = float('inf')
    
    for forget_idx, forget_result in enumerate(forget_all_node):
        for retain_idx, retain_result in enumerate(retain_all_node):
            distance = calculate_hamming_distance(forget_result, retain_result)
            hamming_distances[(forget_idx, retain_idx)] = distance
            
            if distance < min_distance:
                min_distance = distance
                best_forget_idx = forget_idx
                best_retain_idx = retain_idx
            
            print(f"  forget[{forget_idx}] vs retain[{retain_idx}]: 汉明距离 = {distance}")
    

    

    if best_forget_idx is None or best_retain_idx is None:

        return circuits, VALUE_retain, VALUE_forget, input_retain, input_forget, leaf_nodes_by_circuit, all_result_lists, forget_all_node, retain_all_node, None, None, [], [], []
    

    
    best_forget_result = forget_all_node[best_forget_idx]
    best_retain_result = retain_all_node[best_retain_idx]
    

    forget_nodes = set(best_forget_result.keys())
    retain_nodes = set(best_retain_result.keys())
    
    common_nodes = forget_nodes.intersection(retain_nodes)
    non_common_nodes = forget_nodes.symmetric_difference(retain_nodes)

    

    conflict_node_list = []
    same_value_nodes = []
    
    for node in common_nodes:
        if best_forget_result[node] != best_retain_result[node]:
            conflict_node_list.append(node)
        else:
            same_value_nodes.append(node)
    

    non_common_nodes.update(same_value_nodes)
    

    false_node_list = []
    true_node_list = []
    
    for node in non_common_nodes:
        if node in best_forget_result:
            value = best_forget_result[node]
        else:
            value = best_retain_result[node]
        
        if value is False:
            false_node_list.append(node)
        elif value is True:
            true_node_list.append(node)
    

    conflict_set = set(conflict_node_list)
    false_set = set(false_node_list)
    true_set = set(true_node_list)
    

    all_nodes = conflict_set.union(false_set).union(true_set)
    if len(all_nodes) != len(conflict_node_list) + len(false_node_list) + len(true_node_list):

        conflict_node_list = list(conflict_set)
        false_node_list = list(false_set)
        true_node_list = list(true_set)
    
   
    
    def extract_circuit_name(circuit_path):

        parts = circuit_path.split('/')

        for part in parts:
            if '_edges' in part:

                return part.split('_')[0]
        return "unknown"
    
   
    circuit_names = []
    for i in range(1, 7):
        circuit_attr = f"circuit{i}"
        if hasattr(args, circuit_attr):
            circuit_path = getattr(args, circuit_attr)
            if circuit_path and os.path.exists(circuit_path):
                circuit_name = extract_circuit_name(circuit_path)
                circuit_names.append(circuit_name)
    

    suffix = "_".join(circuit_names)
    

    save_dir = ""
    os.makedirs(save_dir, exist_ok=True)
    
   
    conflict_file = os.path.join(save_dir, f"conflict_node_list_{suffix}.json")
    with open(conflict_file, 'w') as f:
        json.dump(conflict_node_list, f, indent=2)
  
    
 
    false_file = os.path.join(save_dir, f"false_node_list_{suffix}.json")
    with open(false_file, 'w') as f:
        json.dump(false_node_list, f, indent=2)
   
    
   
    true_file = os.path.join(save_dir, f"true_node_list_{suffix}.json")
    with open(true_file, 'w') as f:
        json.dump(true_node_list, f, indent=2)
   
    
    return circuits, VALUE_retain, VALUE_forget, input_retain, input_forget, leaf_nodes_by_circuit, all_result_lists, forget_all_node, retain_all_node, best_forget_idx, best_retain_idx, conflict_node_list, false_node_list, true_node_list

if __name__ == "__main__":

    circuits, VALUE_retain, VALUE_forget, input_retain, input_forget, leaf_nodes_by_circuit, all_result_lists, forget_all_node, retain_all_node, best_forget_idx, best_retain_idx, conflict_node_list, false_node_list, true_node_list = main()
    

   
    
   

