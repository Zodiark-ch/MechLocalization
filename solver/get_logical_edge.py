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

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-edge1", "--edge1", type=str, default="")
    parser.add_argument("-edge2", "--edge2", type=str, default="")
    parser.add_argument("-w", "--with_embedding_nodes", action="store_true", default=True)
    parser.add_argument("-o", "--output", type=str, default="")
    
    args = parser.parse_args()
    
    if args.output == "":
        
        args.output = os.path.join("", "")
    else:
        raise ValueError("Output file must be specified when comparing two models") 
    
    return args

def add_o_suffix_to_nodes(edges):

    modified_edges = []
    for edge in edges:
        from_node, to_node = edge
        

        if from_node.startswith("a") and from_node[-1].isdigit() and from_node[-1] not in ['q', 'k', 'v', 'o']:
            from_node = from_node + ".o"
        
 
        if to_node.startswith("a") and to_node[-1].isdigit() and to_node[-1] not in ['q', 'k', 'v', 'o']:
            to_node = to_node + ".o"
        
        modified_edges.append((from_node, to_node))
    

    additional_edges = []
    

    all_edges_set = set(modified_edges)
    

    from_nodes_set = set(edge[0] for edge in modified_edges)
    
    for edge in modified_edges:
        from_node, to_node = edge
        
    
        if (to_node.startswith("a") and 
            to_node[-1] in ['q', 'k', 'v']):
            
     
            if to_node not in from_nodes_set:
                
      
                o_node = to_node[:-1] + 'o' 
                
                if o_node in from_nodes_set:

                    additional_edges.append((to_node, o_node))
    

    modified_edges.extend(additional_edges)
    
    return modified_edges

def classify_h_numbers(edges):

    modified_edges = []
    for edge in edges:
        from_node, to_node,type = edge
        
        if from_node.startswith("a") and from_node[-1] in ['k', 'v']:
            from_node = process_h_classification(from_node)
        
      
        if to_node.startswith("a") and to_node[-1] in ['k', 'v']:
            to_node = process_h_classification(to_node)
        
        modified_edges.append((from_node, to_node,type))
    
    return modified_edges

def process_h_classification(node_name):

    if not node_name.startswith("a") or node_name[-1] not in ['k', 'v']:
        return node_name
    

    pattern = r'^([a-z]\d+)\.(h\d+)\.([qkv])$'
    match = re.match(pattern, node_name)
    
    if match:
        prefix, h_part, suffix = match.groups()
        

        h_number = int(h_part[1:]) 
        
       
        h_group = h_number // 4
        
        
        new_node_name = f"{prefix}.H{h_group}.{suffix}"
        return new_node_name
    
    return node_name

def find_OR_gate_edges(edges_ns, edges_dn):

    edges_ns_set = set(edges_ns)
    edges_dn_set = set(edges_dn)
    
 
    candidate_edges = edges_dn_set - edges_ns_set
    

    filtered_edges = []
    for edge in candidate_edges:
        from_node, to_node = edge
        if not to_node.endswith(".o"):
            filtered_edges.append(edge)
    

    to_node_count = {}
    for edge in filtered_edges:
        to_node = edge[1]
        to_node_count[to_node] = to_node_count.get(to_node, 0) + 1
 
    ns_to_node_count = {}
    for edge in edges_ns_set:
        to_node = edge[1]
        ns_to_node_count[to_node] = ns_to_node_count.get(to_node, 0) + 1
    
   
    for to_node, count in ns_to_node_count.items():
        if count == 1:
            to_node_count[to_node] = to_node_count.get(to_node, 0) + 1
    
    
    sub_edges = []
    for edge in filtered_edges:
        to_node = edge[1]
        if to_node_count[to_node] >= 2:
            sub_edges.append(edge)
    
    return sub_edges

def find_AND_gate_edges(edges_ns, edges_dn):

    edges_ns_set = set(edges_ns)
    edges_dn_set = set(edges_dn)
    
   
    candidate_edges = edges_ns_set-edges_dn_set 
    
  
    filtered_edges = []
    for edge in candidate_edges:
        from_node, to_node = edge
        if not to_node.endswith(".o"):
            filtered_edges.append(edge)
    
    
    to_node_count = {}
    for edge in filtered_edges:
        to_node = edge[1]
        to_node_count[to_node] = to_node_count.get(to_node, 0) + 1
    
   
    dn_to_node_count = {}
    for edge in edges_dn_set:
        to_node = edge[1]
        dn_to_node_count[to_node] = dn_to_node_count.get(to_node, 0) + 1
    
   
    for to_node, count in dn_to_node_count.items():
        if count == 1:
            to_node_count[to_node] = to_node_count.get(to_node, 0) + 1
    
   
    sub_edges = []
    for edge in filtered_edges:
        to_node = edge[1]
        if to_node_count[to_node] >= 2:
            sub_edges.append(edge)
    
    return sub_edges

def create_bool_edges(edges_ns, edges_dn, or_edges, and_edges):

    all_edges = list(set(edges_ns + edges_dn))
    
   
    or_edges_set = set(or_edges)
    and_edges_set = set(and_edges)
    
    
    bool_edges = []
    for edge in all_edges:
        from_node, to_node = edge
        
        
        if edge in or_edges_set:
            label = "OR"
        elif edge in and_edges_set:
            label = "AND"
        else:
            label = "ADDER"
        
       
        bool_edges.append([from_node, to_node, label])
    
    return bool_edges

def validate_paths_to_resid_post(bool_edges):

    current_bool_edges = bool_edges.copy()
    


    adjacency_list = {}
    for edge in current_bool_edges:
        from_node, to_node, label = edge
        if from_node not in adjacency_list:
            adjacency_list[from_node] = []
        adjacency_list[from_node].append(to_node)
    

    all_from_nodes = set(edge[0] for edge in current_bool_edges)
    

    failed_nodes = []
    successful_nodes = []
    added_edges = []
    
    for from_node in all_from_nodes:
        success, failed_node = can_reach_resid_post(from_node, adjacency_list, set(),current_bool_edges)
        if success:
            successful_nodes.append(from_node)
        else:
            failed_nodes.append(failed_node)

    result = {
        "total_from_nodes": len(all_from_nodes),
        "successful_nodes": len(successful_nodes),
        "failed_nodes": len(failed_nodes),
        "failed_node_list": failed_nodes,
        "is_valid": len(failed_nodes) == 0,
        "updated_bool_edges": current_bool_edges
    }
    
    return current_bool_edges

def can_reach_resid_post(node, adjacency_list, visited,current_bool_edges):

    if node in visited:
        return False, node
    visited.add(node)
    
  
    if node == "resid_post":
        return True, None
    
   
    if node not in adjacency_list:
        new_edge = [node, "resid_post", "ADDER"]
        current_bool_edges.append(new_edge)
        return False, node

    for neighbor in adjacency_list[node]:
        success, node = can_reach_resid_post(neighbor, adjacency_list, visited.copy(),current_bool_edges)
        if success:
            return True, None
  
    return False, node

def save_bool_edges(bool_edges, output_path):

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
   
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(bool_edges, f, indent=2, ensure_ascii=False)
    


def sanitize_edges(edges):
    # First, add all q,k,v -> o edges
    new_edges_ = set()
    for edge in edges:
        if edge[0][0] == "a" and edge[0][-1] not in ["q", "k", "v"]:
            new_edges_.add(edge[0])
    for to in new_edges_:
        for suffix in [".q", ".k", ".v"]:
            from_ = to +  suffix
            edges.append((from_, to))
    while True:
        orig_len = len(edges)
        # Find all nodes that are destinations but not sources
        froms = set()
        tos = set()
        for edge in edges:
            froms.add(edge[0])
            if edge[1] != "resid_post":
                tos.add(edge[1])
        banned_tos = tos.difference(froms)
        edges = [e for e in edges if e[1] not in banned_tos]

        # Find qkv nodes that have no incoming edges, and remove the q -> o edge for them
        qkv_nodes = set()
        for edge in edges:
            if edge[1].endswith(".q"):
                qkv_nodes.add(edge[1])
            elif edge[1].endswith(".k"):
                qkv_nodes.add(edge[1])
            elif edge[1].endswith(".v"):
                qkv_nodes.add(edge[1])

        edges = [
            e for e in edges if not (
                (e[0].endswith(".q") and e[0] not in qkv_nodes) or
                (e[0].endswith(".k") and e[0] not in qkv_nodes) or
                (e[0].endswith(".v") and e[0] not in qkv_nodes)
            )
        ]
        if orig_len == len(edges):
           break

    return edges

def main():
    args = parse_args()
    edges_ns = json.load(open(args.edge1))
    edges_dn = json.load(open(args.edge2))

    

    edges_ns = add_o_suffix_to_nodes(edges_ns)
    edges_dn = add_o_suffix_to_nodes(edges_dn)
  
    

    edges_ns = list(set(edges_ns))
    edges_dn = list(set(edges_dn))

    

    OR_edges = find_OR_gate_edges(edges_ns, edges_dn)
    AND_edges = find_AND_gate_edges(edges_ns, edges_dn)

    

    bool_edges = create_bool_edges(edges_ns, edges_dn, OR_edges,AND_edges)

    

    validation_result = validate_paths_to_resid_post(bool_edges)

    validation_result=classify_h_numbers(validation_result)
    validation_result=list(set(tuple(edge) for edge in validation_result))

    

    save_bool_edges(validation_result, args.output)

    
    
    
    
if __name__ == '__main__':
    main()