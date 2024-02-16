import os
import sys
import time
import matplotlib.pyplot as plt



def run_gaston(Threshold_list, gaston_graphs_file):
    gaston_runtimes = []
    for Threshold in Threshold_list:
        count = get_graph_count(gaston_graphs_file)
        Thresholdnum = float(Threshold * count / 100)
        command = "./gaston " + str(Thresholdnum) + " gastonfile.txt gaston-output.txt"
        start_time = time.time()
        os.system(command)
        end_time = time.time()
        gaston_runtimes.append(end_time - start_time)
    return gaston_runtimes

def run_gspan(Threshold_list,gspan_graphs_file):
    gspan_runtimes = []
    
    for Threshold in Threshold_list:
        count1 = get_graph_count(gspan_graphs_file)
        Thresholdnum = float(Threshold / 100)
        start_time = time.time()
        command = "./gSpan-64 -f gspanfile.txt -s " + str(Thresholdnum) + " -o"
        os.system(command)
        end_time = time.time()
        gspan_runtimes.append(end_time - start_time)
    return gspan_runtimes

def run_fsg(Threshold_list,fsg_graphs_file):
    fsg_runtimes = []
    for Threshold in Threshold_list:
        count2 = get_graph_count(fsg_graphs_file)
        comnd = "./fsg -s " + str(Threshold) + " " + "fsgfile.txt"
        start_time = time.time()
        os.system(comnd)
        end_time = time.time()
        new_count = end_time-start_time
        fsg_runtimes.append(new_count)
    return fsg_runtimes

def get_graph_count(file_path):
    with open(file_path, "r") as file:
        for line in file:
            count = int(line)
            return count

def plot_runtimes(Threshold_list, fsg_runtimes, gspan_runtimes, gaston_runtimes,output_file):
    
    plt.plot(Threshold_list, gspan_runtimes,color='red', label="GSPAN")
    plt.plot(Threshold_list, gaston_runtimes,color='green', label="GASTON")
    plt.plot(Threshold_list, fsg_runtimes,color ='blue', label="FSG")
    plt.xlabel("Threshold")
    plt.ylabel("Runtime")
    plt.title("Runtime vs Threshold")
    plt.legend()
    plt.savefig(output_file)

def main(output_file):
    Threshold_list = [5, 10, 25, 50, 95]
    fsg_runtimes = run_fsg(Threshold_list,"fsgfile1.txt")
    gspan_runtimes = run_gspan(Threshold_list,"gspanfile1.txt")
    gaston_runtimes = run_gaston(Threshold_list, "gastonfile1.txt")
    print(fsg_runtimes)
    print(gspan_runtimes)
    print(gaston_runtimes)
    plot_runtimes(Threshold_list, fsg_runtimes, gspan_runtimes, gaston_runtimes,output_file)

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python script.py input_file algorithm")
    else:
        result_path = sys.argv[1]
        file_path = os.path.join(result_path, "q1_plot_cs1200405.png")
        main(file_path)
