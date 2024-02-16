import sys

def parse_input(input_file):
    graphs = []
    with open(input_file, "r") as file:
        graph = {"id": None, "nodes": [], "edges": []}
        for line in file:
            line = line.strip()
            if not line:
                continue
            elif line.startswith("#"):
                if graph["id"]:
                    graphs.append(graph)
                    #print(graph["nodes"])
                    graph = {"id": None, "nodes": [], "edges": []}
                    
                graph["id"] = line[1:]
                
            elif line.isdigit():
                if not graph["nodes"]:
                    num_nodes = int(line)
                else:
                    num_edges = int(line)
            elif line.isalpha():
                graph["nodes"].extend(line.split())
            else:
                source, dest, label = map(int, line.split())
                graph["edges"].append((source, dest, label))
        if graph["id"]:
            graphs.append(graph)
            
                
    return graphs

def convert_to_gspan(graphs):
    gspan_graphs = []
    for i, graph in enumerate(graphs):
        # gspan_graph = f"t # {graph['id']}\n"
        gspan_graph = f"t # {i}\n"
        node_labels = {}
        k=0
        a=0
        for j,node_label in enumerate(graph["nodes"]):
            if node_label in node_labels:
                a=1
            else:
                node_labels[node_label] = k
                k=k+1
            gspan_graph += f"v {j} {node_labels[node_label]}\n"
        for source, dest, label in graph["edges"]:
            gspan_graph += f"e {source} {dest} {label}\n"
        gspan_graphs.append(gspan_graph)
    return gspan_graphs

def convert_to_gaston(graphs):
    gspan_graphs = []
    for i, graph in enumerate(graphs):
        # gspan_graph = f"t # {graph['id']}\n"
        gspan_graph = f"t # {i}\n"
        node_labels = {}
        k=0
        a=0
        for j,node_label in enumerate(graph["nodes"]):
            if node_label in node_labels:
                a=1
            else:
                node_labels[node_label] = k
                k=k+1
            gspan_graph += f"v {j} {node_labels[node_label]}\n"
        for source, dest, label in graph["edges"]:
            gspan_graph += f"e {source} {dest} {label}\n"
        gspan_graphs.append(gspan_graph)
    return gspan_graphs

def convert_to_fsg(graphs):
    gspan_graphs = []
    for i, graph in enumerate(graphs):
        # gspan_graph = f"t # {graph['id']}\n"
        gspan_graph = f"t \n"
        node_labels = {}
        k=0
        a=0
        for j,node_label in enumerate(graph["nodes"]):
            if node_label in node_labels:
                a=1
            else:
                node_labels[node_label] = k
                k=k+1
            gspan_graph += f"v {j} {node_labels[node_label]}\n"
        for source, dest, label in graph["edges"]:
            gspan_graph += f"u {source} {dest} {label}\n"
        gspan_graphs.append(gspan_graph)
    return gspan_graphs



def write_output(output_file, graphs):
    with open(output_file, "w+") as outfile:
        for graph in graphs:
            outfile.write(graph)

# def main(input_file):
#     graphs = parse_input(input_file)
#     gspan_graphs = convert_to_gspan(graphs)
#     write_output("gspan2.txt", gspan_graphs)
#     with open("gspan-graphs2.txt", "w+") as dup:
#         dup.write(str(len(graphs)))
            

def main(input_file, technique):
    if technique not in {"GSPAN", "GASTON", "FSG"}:
        print("Unsupported technique")
        return
    graphs = parse_input(input_file)
    if technique == "GSPAN":
        gspan_graphs = convert_to_gspan(graphs)
        write_output("gspanfile.txt", gspan_graphs)
        with open("gspanfile1.txt", "w+") as dup:
            dup.write(str(len(graphs)))
    elif technique == "GASTON":
        gaston_graphs = convert_to_gaston(graphs)
        write_output("gastonfile.txt", gaston_graphs)
        with open("gastonfile1.txt", "w+") as dup:
            dup.write(str(len(graphs)))
    elif technique == "FSG":
        fsg_graphs = convert_to_fsg(graphs)
        write_output("fsgfile.txt", fsg_graphs)
        with open("fsgfile1.txt", "w+") as dup:
            dup.write(str(len(graphs)))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input_file technique")
    else:
        input_file = sys.argv[1]
        technique = sys.argv[2]
        main(input_file, technique)
