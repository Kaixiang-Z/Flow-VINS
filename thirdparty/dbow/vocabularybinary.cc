/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-10-28 10:26:21
 * @Description:
 */
#include "vocabularybinary.hpp"
#include <opencv2/core/core.hpp>
using namespace std;

VINSLoop::Vocabulary::Vocabulary()
    : nNodes(0)
    , nodes(nullptr)
    , nWords(0)
    , words(nullptr) {}

VINSLoop::Vocabulary::~Vocabulary() {
	if (nodes != nullptr) {
		delete[] nodes;
		nodes = nullptr;
	}

	if (words != nullptr) {
		delete[] words;
		words = nullptr;
	}
}

void VINSLoop::Vocabulary::serialize(ofstream& stream) {
	stream.write((const char*)this, staticDataSize());
	stream.write((const char*)nodes, sizeof(Node) * nNodes);
	stream.write((const char*)words, sizeof(Word) * nWords);
}

void VINSLoop::Vocabulary::deserialize(ifstream& stream) {
	stream.read((char*)this, staticDataSize());

	nodes = new Node[nNodes];
	stream.read((char*)nodes, sizeof(Node) * nNodes);

	words = new Word[nWords];
	stream.read((char*)words, sizeof(Word) * nWords);
}