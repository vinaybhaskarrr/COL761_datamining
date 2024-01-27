#include<bits/stdc++.h>
using namespace std;

#define int long long


void decompress(std::vector<int> &result,int key, std::map<int, std::vector<int>> &decode ) {
    if (decode.find(key)==decode.end()) {result.push_back(key);return;}
    
        for (int item : decode[key]) {
            decompress(result,item, decode);
        }
    
    
}


void decompressAndWriteToFile(const std::string& compressedfile, const std::string& decompressedfile) {
    // Reading the mapping from the input file

	std::string line;

	map<int, vector<int>> decryptionMap;

    std::ifstream input(compressedfile);
    
    
    int numIntegers = 0;

    
    while (getline(input, line) && !line.empty()) {}

    // Parsing the mapping
    while (getline(input, line)) {

		numIntegers++;

		int currentKey;

        std::stringstream ss(line);

        std::string duplicate;
        

        ss >> duplicate;
        currentKey = stoi(duplicate);
        

        while (ss >> duplicate) {
            
            decryptionMap[currentKey].push_back(stoi(duplicate));

			numIntegers++;
        }
    }

    input.close();

    // Creating the original file using the mapping
    std::ofstream destination;
    destination.open(decompressedfile);
    std::ifstream source_read(compressedfile);
    line.clear();

    while (getline(source_read, line) && !line.empty()) {
        std::stringstream ss1(line);
        std::string duplicate;

        while (ss1 >> duplicate) {
            numIntegers++;

            if (decryptionMap.find(stoi(duplicate)) == decryptionMap.end()) {

				destination << duplicate << " ";
                
            } else {
                

				std::vector<int> values;
                decompress(values, stoi(duplicate), decryptionMap);

                for (auto it : values) {
                    destination << it << " ";
                }
            }
        }

        destination << "\n";
    }

    destination.close();

    source_read.close();
    

    std::cout << "Total Integers count in the compressed file: " << numIntegers << std::endl;
}
signed main(int32_t argc, char* argv[]){
  
    decompressAndWriteToFile(argv[1],argv[2]);
    
    return 0;
}
