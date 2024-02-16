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


void readGraphs(const string& filename , vector<vector<int>>&graphs) {
    ifstream file(filename);
    if (!file) {
        cerr << "Error: Unable to open input file" << endl;
        exit(1);
    }
    int idx=0;
    string line;
    
    while (getline(file, line)) {

       
    if (line.empty()) continue;
    if( line[0]=='x'){
        cout<<idx<<endl;
        
        string input = line;
        istringstream iss(input);
        //cout<<input<<endl;

    vector<int> words;
    string word;
    iss>>word;
    while (iss>>word) {
        // cout<<"hii"<<endl;
        // cout<<word<<endl;
        int j=stoi(word);
        graphs[j][idx]=1;
    }
    idx++;


    }
    //cout<<"loop"<<endl;
    
    
    
}
    return;
}


int main(int argc, char* argv[]) {
    string resultp=argv[1];
    string outpath;
    if (resultp[resultp.size()-1]=='/'){
        outpath=resultp+"features_cs1200405.txt";

    }
    else{
        outpath= resultp+"/features_cs1200405.txt";
    }
    

    ifstream file("gcount.txt");
    int count=0;
    int f_count=0;
    ifstream file1("f_count.txt");
    if (!file1) {
        f_count=100;
    }
    else{
        string line1;
        while(getline(file1, line1)){
        f_count=stoi(line1);


    }

    }
    string line;
    while(getline(file, line)){
        cout<<line<<endl;
        count=stoi(line);


    }
    cout<<"count"<<count<<endl;
    vector<vector<int>>graphs(count,vector<int>(f_count,0));


    readGraphs("100.txt",graphs);
    ofstream outfile1(outpath);
    int id=0;
    for (auto it:graphs){
        outfile1<<to_string(id)<<" # ";
        cout<<it.size()<<endl;
        for(auto gg:it){
            outfile1<<to_string(gg)<<" ";
        }
        outfile1<<endl;
        id++;

    }
    
    return 0;
}
