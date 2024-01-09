/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-10-28 10:26:21
 * @Description:
 */

#include "featurevector.h"
#include <iostream>
#include <map>
#include <vector>

namespace DBoW2 {

// ---------------------------------------------------------------------------

FeatureVector::FeatureVector(void) {}

// ---------------------------------------------------------------------------

FeatureVector::~FeatureVector(void) {}

// ---------------------------------------------------------------------------

void FeatureVector::addFeature(NodeId id, unsigned int i_feature) {
	FeatureVector::iterator vit = this->lower_bound(id);

	if (vit != this->end() && vit->first == id) {
		vit->second.push_back(i_feature);
	} else {
		vit = this->insert(vit, FeatureVector::value_type(id, std::vector<unsigned int>()));
		vit->second.push_back(i_feature);
	}
}

// ---------------------------------------------------------------------------

std::ostream& operator<<(std::ostream& out, const FeatureVector& v) {
	if (!v.empty()) {
		FeatureVector::const_iterator vit = v.begin();

		const std::vector<unsigned int>* f = &vit->second;

		out << "<" << vit->first << ": [";
		if (!f->empty())
			out << (*f)[0];
		for (unsigned int i = 1; i < f->size(); ++i) {
			out << ", " << (*f)[i];
		}
		out << "]>";

		for (++vit; vit != v.end(); ++vit) {
			f = &vit->second;

			out << ", <" << vit->first << ": [";
			if (!f->empty())
				out << (*f)[0];
			for (unsigned int i = 1; i < f->size(); ++i) {
				out << ", " << (*f)[i];
			}
			out << "]>";
		}
	}

	return out;
}

// ---------------------------------------------------------------------------

} // namespace DBoW2
