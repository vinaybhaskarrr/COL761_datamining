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


void readGraphs(const string& filename,string outputfilename) {
    ifstream file(filename);
    if (!file) {
        cerr << "Error: Unable to open input file" << endl;
        exit(1);
    }
    ofstream outputFile(outputfilename);
    int f_count=0;

    
    string line;
    while (getline(file, line)) {
       
        if (line.empty()) {
            outputFile<<line<<endl;
            continue;
        }
        string input = line;
        istringstream iss(input);

    vector<string> words;
    string word;
    while (getline(iss, word, ' ')) {
        words.push_back(word);
    }
    //cout<<words[0]<<" "<<words[words.size()-1]<<" "<<words.size()<<endl;
        if (words[1] == "#") {
            if(f_count>100) {cout<<"break"<<endl;return;}
            f_count++;

        } 
        outputFile<<line<<endl;
   
}
ofstream outputFile1("fcount.txt");
outputFile1<<to_string(f_count)<<endl;

cout<<f_count<<endl;
 return;
}


int main(int argc, char* argv[]) {
    string filename = "gspanin.txt.fp";


    readGraphs(filename,"100.txt");
    //  ofstream outputFile("features_kerberosid.txt");
    // for (auto it:graphs){
    //     //cout<<it.classLabel<<endl;
    //     outputFile<<it.id<<" # "<<it.classLabel<<endl;

    // }
    return 0;
}
