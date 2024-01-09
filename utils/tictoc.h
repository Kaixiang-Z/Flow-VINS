/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-10-31 08:28:11
 * @Description:
 */
#pragma once

#include <chrono>

/**
 * @brief: Class which is used to timing, unit is ms
 */
class TicToc {
public:
	TicToc() { tic(); }

	void tic() { start = std::chrono::system_clock::now(); }

	double toc() {
		end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;
		return elapsed_seconds.count() * 1000;
	}

private:
	std::chrono::time_point<std::chrono::system_clock> start, end;
};
