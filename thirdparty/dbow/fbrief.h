/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-10-28 10:26:21
 * @Description:
 */

#pragma once
#include "../dvision/dvision.h"
#include "fclass.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace DBoW2 {

/// Functions to manipulate BRIEF descriptors
class FBrief : protected FClass {
public:
	typedef DVision::BRIEF::bitset TDescriptor;
	typedef const TDescriptor* pDescriptor;

	/**
	 * Calculates the mean value of a set of descriptors
	 * @param descriptors
	 * @param mean mean descriptor
	 */
	static void meanValue(const std::vector<pDescriptor>& descriptors, TDescriptor& mean);

	/**
	 * Calculates the distance between two descriptors
	 * @param a
	 * @param b
	 * @return distance
	 */
	static double distance(const TDescriptor& a, const TDescriptor& b);

	/**
	 * Returns a string version of the descriptor
	 * @param a descriptor
	 * @return string version
	 */
	static std::string toString(const TDescriptor& a);

	/**
	 * Returns a descriptor from a string
	 * @param a descriptor
	 * @param s string version
	 */
	static void fromString(TDescriptor& a, const std::string& s);

	/**
	 * Returns a mat with the descriptors in float format
	 * @param descriptors
	 * @param mat (out) NxL 32F matrix
	 */
	static void toMat32F(const std::vector<TDescriptor>& descriptors, cv::Mat& mat);
};

} // namespace DBoW2
