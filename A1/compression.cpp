#include<chrono>

#include<bits/stdc++.h>

#define int long long



int highValueSoFar;

using namespace std;

class Node {
public:
    int name;
    int count;
    Node *parent;
    Node *link;
    vector<Node *> children;
    Node(int name, int count, Node *parent=NULL, Node *link=NULL) {
        this->name = name;
        this->count = count;
        this->parent = parent;
        this->link = link;
    }
};

std::map<int,vector<int>> deshrinker;

void updateNodeWithTransaction(map<int, Node*>& headerTable,Node* root, const vector<pair<int, int>>& transaction, int index) {
    if (index == transaction.size()) {
        return;
    }

    int itemName = transaction[index].first;
    int itemCount = transaction[index].second;

    for (Node* child : root->children) {
        if (child->name == itemName) {
            child->count += itemCount;
            updateNodeWithTransaction(headerTable,child, transaction, index + 1);
            return;
        }
    }

    
    Node* newNode = new Node(itemName, itemCount, root);
    root->children.push_back(newNode);

  
    newNode->link = headerTable[itemName];
    headerTable[itemName] = newNode;

    updateNodeWithTransaction(headerTable,newNode, transaction, index + 1);
}

std::map<vector<int>, int> shrinker;



Node* buildFPTree(std::map<int, Node*>& headerTable,std::map<int, int>& fcount,const std::string& file ) {
    Node* root = new Node(-1, 0, NULL);

    
    std::ifstream inputFile(file);
    std::string line;
    while (getline(inputFile, line)) {
        std::stringstream ss(line);
        std::string duplicate;
        std::vector<std::pair<int, int>> sortedTransaction;

        while (ss >> duplicate) {
            if (fcount.count(std::stoi(duplicate))) {
                sortedTransaction.push_back({fcount[std::stoi(duplicate)], std::stoi(duplicate)});
            }
        }

       
        std::sort(sortedTransaction.begin(), sortedTransaction.end(), std::greater<>());

        
        std::vector<std::pair<int, int>> transaction;
        for (const auto& item : sortedTransaction) {
            transaction.push_back({item.second, 1});
        }

        updateNodeWithTransaction(headerTable,root, transaction, 0);
    }

    inputFile.close();
    return root;
}

using namespace std::chrono;




#define varadharajamannar push_back


void check(std::map<int, int> &conditionCount,std::vector<std::pair<int, int>> &frequentItems,int threshold,int pointer1,std::set<int> &conditionSet,std::vector<std::set<int>>& result){
    if (conditionCount[pointer1] >= threshold) {
            frequentItems.push_back({conditionCount[pointer1], pointer1});
            conditionSet.insert(pointer1);
            result.push_back(conditionSet);
            conditionSet.erase(pointer1);
        }
}

std::vector<std::pair<int, int>> traverseway(Node* duplicate, int fcount) {

    duplicate = duplicate->parent;
    std::vector<std::pair<int, int>> way;
    
    pair<int,int>need;
    while (duplicate->parent != NULL) {
        need = {duplicate->name, fcount};
        way.varadharajamannar(need);
        duplicate = duplicate->parent;
    }
    std::reverse(way.begin(), way.end());

    return way;
}


void customCopy(const std::string& destinationFileName,const std::string& sourceFileName ) {
    
    std::ofstream destinationFile;
    destinationFile.open(destinationFileName);

    std::string line;
    std::fstream sourceFile(sourceFileName);

    while (getline(sourceFile, line)) {std::stringstream lineStream(line);std::string word;

        while (lineStream >> word) destinationFile << word << " ";
            
        destinationFile << endl;
    }

    sourceFile.close();
    destinationFile.close();
}

Node* formConditionalFPTree(const std::vector<std::vector<std::pair<int, int>>>& ways, const std::map<int, int>& fcount, std::map<int, Node*>& headerTable) {
    Node* root = new Node(-1, 0, NULL);

    for (const auto& way : ways) {
        std::vector<std::pair<int, std::pair<int, int>>> sortedway;
        
        for (const auto& item : way) {
            if (fcount.count(item.first)) {
                sortedway.push_back({-fcount.at(item.first), item});
            }
        }

        std::sort(sortedway.begin(), sortedway.end());

        
        std::vector<std::pair<int, int>> transaction;
        for (const auto& item : sortedway) transaction.push_back({item.second.first, item.second.second});
        
        updateNodeWithTransaction(headerTable,root, transaction, 0 );
    }

    return root;
}

#define salaardevaratharaisar push_back
void extractFrequentSubtree(std::vector<std::set<int>>& result,int itemName,std::map<int, Node*>& elementTable, int threshold, int numTransactions, std::set<int> conditionSet ) {
    std::vector<std::vector<std::pair<int, int>>> ways;
    Node* currentNode = elementTable[itemName];

    while (currentNode != nullptr) {
        auto res = traverseway(currentNode, currentNode->count);
        ways.salaardevaratharaisar(res);
        currentNode = currentNode->link;
    }

    std::map<int, int> conditionCount;
    std::map<int, Node*> conditionHeaderTable;
    std::set<int> distinctItems;
    std::vector<std::pair<int, int>> frequentItems;

    for (const auto& way : ways) {
        for (const auto& item : way) {
            conditionCount[item.first] += item.second;
            distinctItems.insert(item.first);
        }
    }

    for(int pointer1:distinctItems){
    
        check(conditionCount,frequentItems,threshold,pointer1,conditionSet,result);
    }

    if (frequentItems.empty()) {
        return;
    }

    formConditionalFPTree(ways, conditionCount, conditionHeaderTable);
    std::sort(frequentItems.begin(), frequentItems.end());

    for (const auto& item : frequentItems) {
        conditionSet.insert(item.second);
        extractFrequentSubtree(result,item.second, conditionHeaderTable,threshold, numTransactions, conditionSet );
        conditionSet.erase(item.second);
    }
}





#define souryangaparvam push_back

double buildExecutable(map<int, int> &frequencyMap,string outputFile,string inputFile) {
    ofstream compressedOutput;

    string line;
    int compressedItemCount=0,transactionCountCompressed=0,originalItemCount=0 ,transactionCountOriginal = 0;
    
    compressedOutput.open(outputFile);

    fstream inputFileStream(inputFile);



    while (getline(inputFileStream, line)) {
        if (line.empty()) {
            break;
        }
        vector<pair<int, int>> dataentry;

        transactionCountOriginal++;
        vector<int> previousTransaction;

        stringstream vv(line);
        string duplicate;
        vector<int> currentTransaction;
        
        
        pair<int,int>help1;
        while (vv >> duplicate) {
            originalItemCount++;
            help1={-1 * frequencyMap[stoi(duplicate)], stoi(duplicate)};
            dataentry.souryangaparvam(help1);
            
        }

        sort(dataentry.begin(), dataentry.end());
        
        for (auto x:dataentry) {
            int item = x.second;
            currentTransaction.push_back(item);

            if (shrinker.find(currentTransaction) != shrinker.end()) {
                    previousTransaction = currentTransaction;
                } else {
                    if (previousTransaction.size() == 1) {
                        compressedItemCount++;
                        compressedOutput << previousTransaction[0] << " ";
                    } else if (previousTransaction.size() > 1) {
                        deshrinker[shrinker[previousTransaction]] = previousTransaction;
                        compressedOutput << shrinker[previousTransaction] << " ";
                        compressedItemCount++;
                        
                    }

                    previousTransaction.clear();
                    previousTransaction.push_back(item);
                    currentTransaction = previousTransaction;
                }
        }

        if (!previousTransaction.empty()) {
            if (shrinker.find(previousTransaction) == shrinker.end() || previousTransaction.size() == 1) {
                for (auto element : previousTransaction) {
                    compressedItemCount++;
                    compressedOutput << element << " ";
                }
                compressedOutput << "\n";
                transactionCountCompressed++;
                
            } else {
                compressedItemCount++;
                compressedOutput << shrinker[previousTransaction] << "\n";
                deshrinker[shrinker[previousTransaction]] = previousTransaction;
                
            }
        }
    }

    compressedOutput << endl;

    for (auto iterator : deshrinker) {

        compressedItemCount++;

        compressedOutput << iterator.first << " ";
        

        for (auto element : iterator.second) {
            
            compressedOutput << element << " ";

            compressedItemCount++;
        }

        compressedOutput << endl;
    }

    while (getline(inputFileStream, line)) {
        string duplicate;

        stringstream vv(line);
        
        while (vv >> duplicate) originalItemCount++;
        
    }

    auto countdiff = originalItemCount - compressedItemCount;

    compressedOutput.close();

    inputFileStream.close();
    
    double above = static_cast<double>(countdiff);
    
    double below = static_cast<double>(originalItemCount);

    cout << "intial and later count: " << originalItemCount << ", " << compressedItemCount << endl;

    
    double compressionRatio = above/below;

    return compressionRatio;
}





int evaluateThreshold(double percentile,map<int, int> &frequencyMap) {

    vector<int> frequencyValues;

    for (pair<int,int> help : frequencyMap) {
        frequencyValues.push_back(help.second);
    }

    sort(frequencyValues.begin(), frequencyValues.end());

    int position;
    int totalCount = frequencyValues.size();
    double targetPercentile = percentile;

    if (totalCount < static_cast<int>(targetPercentile * 1000.0)) {
        position = totalCount * (0.7 / targetPercentile);
    } else {
        position = totalCount - static_cast<int>(targetPercentile * 300.0);
    }

    return frequencyValues[position];
}



void filterFrequentItems(map<int, int> &itemFrequencies,map<int, int> &f2, int pointer1, int threshold, set<int> &frequentItems, vector<pair<int,int>> &sortedTransactions,vector<set<int>> &resultSets) {
    
    if(itemFrequencies[pointer1]>=threshold){
        f2[pointer1]=itemFrequencies[pointer1];
        frequentItems.insert(pointer1);
        resultSets.push_back(frequentItems);
		sortedTransactions.push_back({itemFrequencies[pointer1], pointer1});
    }

    
}

void buildCompressionTable(vector<vector<int>> &vinay){
	for(auto it:vinay){
		shrinker[it] = highValueSoFar+1;
		highValueSoFar++;
	}
	
}

int frequentTree(double targetPercentile,string& inputFile,  string& outputFile,vector<set<int>> &resultSets ) {
    

    set<int> distinctItems;

    map<int,int>f2;
    
    
    vector<pair<int,int>> sortedTransactions;
    int transactionCount = 0;
    ifstream inputStream(inputFile);
    map<int, Node*> headerTable;
    string line;
    map<int, int> itemFrequency;
    while (getline(inputStream, line)) {
        transactionCount++;
        stringstream ss(line);
        string duplicate;
        

        while (ss >> duplicate) {
            int item = stoi(duplicate);
            itemFrequency[item]++;

            if (item >= highValueSoFar) {
                highValueSoFar = item;
            }

            headerTable[item] = nullptr;
            distinctItems.insert(item);
        }
    }

    inputStream.close();

    int threshold = evaluateThreshold(targetPercentile,itemFrequency);

    for(int pointer1:distinctItems){
        set<int> frequentItems;
        filterFrequentItems(itemFrequency,f2,pointer1, threshold, frequentItems, sortedTransactions,resultSets);
    }

    sort(sortedTransactions.begin(), sortedTransactions.end());


    buildFPTree(headerTable,itemFrequency, inputFile);

    for (const auto& transaction : sortedTransactions) {
        set<int> itemSet;
        itemSet.insert(transaction.second);
        extractFrequentSubtree(resultSets,transaction.second,headerTable, threshold, transactionCount, itemSet );
    }

    vector<vector<int>> frequentItemSets;

    for (const auto& itemSet : resultSets) {
        vector<pair<int,int>> duplicate;

        for (const auto& item : itemSet) duplicate.push_back({-1 * itemFrequency[item], item});

         vector<int> duplicateList;
        
         
        sort(duplicate.begin(), duplicate.end());


        for (const auto& item : duplicate) duplicateList.push_back(item.second);
        

        frequentItemSets.push_back(duplicateList);
    }

    buildCompressionTable(frequentItemSets);

    double compressionRatio = buildExecutable(itemFrequency, outputFile,inputFile );

    double printratio = compressionRatio * 100;
    cout << "Compression Ratio: " << printratio << endl;

    if (compressionRatio < 0.001) {
        return 1;
    }

    return 0;
}








signed main(int32_t argc, char* argv[]) {
    std::string inputSetName = argv[1];
    std::string finalOutputName = argv[2];
    int iterations = 14;
    std::vector<std::string> compressedFileNames = {"cone.dat", "ctwo.dat"};

    auto startTime = std::chrono::high_resolution_clock::now();
    highValueSoFar = 0;
    double percentage = 1;

    for (int i = 0; i < iterations; i++) {
        std::string currentOutputFile = compressedFileNames[i % 2];
        std::vector<std::set<int>> resultSets;
        int halt= frequentTree(percentage,inputSetName, currentOutputFile,resultSets);

        percentage += 0.1;
        inputSetName = currentOutputFile;
        

        if (halt) percentage = percentage * 1.5;
        

        auto currentTime = std::chrono::high_resolution_clock::now();
        auto  timediff = currentTime - startTime;
        auto overalltime = std::chrono::duration_cast<std::chrono::minutes>(timediff);

        if (overalltime.count() < 58) customCopy(finalOutputName,currentOutputFile );
        
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    std::cout << "Total Time Spent: " << totalTime.count() << " milliseconds" << std::endl;

    return 0;
}
