/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-10-28 10:26:21
 * @Description:
 */
#pragma once

#include "bowvector.h"
#include <iostream>
#include <map>
#include <vector>

namespace DBoW2 {

/// Vector of nodes with indexes of local features
class FeatureVector : public std::map<NodeId, std::vector<unsigned int> > {
public:
	/**
	 * Constructor
	 */
	FeatureVector(void);

	/**
	 * Destructor
	 */
	~FeatureVector(void);

	/**
	 * Adds a feature to an existing node, or adds a new node with an initial
	 * feature
	 * @param id node id to add or to modify
	 * @param i_feature index of feature to add to the given node
	 */
	void addFeature(NodeId id, unsigned int i_feature);

	/**
	 * Sends a string versions of the feature vector through the stream
	 * @param out stream
	 * @param v feature vector
	 */
	friend std::ostream& operator<<(std::ostream& out, const FeatureVector& v);
};

} // namespace DBoW2
