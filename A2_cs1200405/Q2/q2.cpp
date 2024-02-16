#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>


using namespace std;

// Define structures to represent graphs, nodes, and edges
struct Node {
    string id;
    string label;
};

struct Edge {
    string source;
    string target;
    string label;
};

struct Graph {
    string id;
    string classLabel;
    vector<Node> nodes;
    vector<Edge> edges;
};


vector<Graph> readGraphs(const string& filename,int &gcount) {
    ifstream file(filename);
    if (!file) {
        cerr << "Error: Unable to open input file" << endl;
        exit(1);
    }

    vector<Graph> graphs;
    Graph currentGraph;
    string line;
    while (getline(file, line)) {
       
        if (line.empty()) continue;
        string input = line;
        istringstream iss(input);

    vector<string> words;
    string word;
    while (getline(iss, word, ' ')) {
        words.push_back(word);
    }
    //cout<<words[0]<<" "<<words[words.size()-1]<<" "<<words.size()<<endl;
        if (words[0] == "#") {
            gcount++;
            if (!currentGraph.nodes.empty()) {
                graphs.push_back(currentGraph);
                currentGraph.nodes.clear();
                currentGraph.edges.clear();
            }
            
            currentGraph.id = words[1];
            currentGraph.classLabel = words[2];
        } else {
            if (words[0]=="v"){
                currentGraph.nodes.push_back({words[1], words[2]});
            }
            if(words[0]=="e"){
                currentGraph.edges.push_back({words[1], words[2], words[3]});
            }
        }
    }
    if (!currentGraph.nodes.empty()) {
        graphs.push_back(currentGraph);
    }
    return graphs;
}


int main(int argc, char* argv[]) {
    int gcount=0;
    string inputfile= argv[1];
    vector<Graph> graphs = readGraphs(inputfile,gcount);
    ofstream outfile1("gspanin.txt");
    //ofstream outputFile("features_kerberosid.txt");
    for (auto it:graphs){
        //cout<<it.classLabel<<endl;
        //outputFile<<it.id<<" # "<<it.classLabel<<endl;
        outfile1<<"t"<<" # "<<it.id<<endl;
        for( auto node:it.nodes){
            outfile1<<"v"<<" "<<node.id<<" "<<node.label<<endl;
        }
        for (auto edge:it.edges){
            outfile1<<"e"<<" "<<edge.source<<" "<<edge.target<<" "<<edge.label<<endl;
        }

    }
    ofstream outfile2("gcount.txt");
    outfile2<<to_string(gcount)<<endl;
    cout<<gcount<<endl;
    return 0;
}
