#include "VocabularyBinary.hpp"
#include <opencv2/core/core.hpp>
using namespace std;

VINSLoop::Vocabulary::Vocabulary()
: nNodes(0), nodes(nullptr), nWords(0), words(nullptr) {
}

VINSLoop::Vocabulary::~Vocabulary() {
    if (nodes != nullptr) {
        delete [] nodes;
        nodes = nullptr;
    }
    
    if (words != nullptr) {
        std::cout<<"~~~~word is not null"<<std::endl;
        delete [] words;
        words = nullptr;
    }
}
    
void VINSLoop::Vocabulary::serialize(ofstream& stream) {
    printf("0\n");
    stream.write((const char *)this, staticDataSize());
    std::cout<<staticDataSize()<<std::endl;
    printf("1\n");
    std::cout<<sizeof(Node)<<std::endl;
    std::cout<<nNodes<<std::endl;
    std::cout<<sizeof(Word)<<std::endl;
    std::cout<<nWords<<std::endl;
    stream.write((const char *)nodes, sizeof(Node) * nNodes);
    std::cout<<"em..."<<std::endl;
    stream.write((const char *)words, sizeof(Word) * nWords);
    std::cout<<"em2..."<<std::endl;
}
    
void VINSLoop::Vocabulary::deserialize(ifstream& stream) {
    printf("0\n");
    stream.read((char *)this, staticDataSize());
    std::cout<<staticDataSize()<<std::endl;
    printf("1\n");
    std::cout<<sizeof(Node)<<std::endl;
    std::cout<<nNodes<<std::endl;
    nodes = new Node[nNodes];
    
    stream.read((char *)nodes, sizeof(Node) * nNodes);
    printf("2\n");
    words = new Word[nWords];
    stream.read((char *)words, sizeof(Word) * nWords);
    printf("3\n");
}

void VINSLoop::Vocabulary::init(){
        nodes = new Node[nNodes];
        words = new Word[nWords];
        std::cout<<sizeof(nodes)<<std::endl;
        std::cout<<sizeof(words)<<std::endl;
    }
