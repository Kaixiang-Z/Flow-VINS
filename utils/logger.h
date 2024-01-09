/*
 * @Author: Kx Zhang
 * @Mailbox: kxzhang@buaa.edu.cn
 * @Date: 2023-10-31 08:27:56
 * @Description:
 */
#pragma once

#include <iostream>

namespace ANSI {

enum class LogLevel { RESET, DEBUG, INFO, WARN, ERROR };
inline constexpr const char* resetCode = "\033[0m";
inline constexpr std::pair<const char*, LogLevel> debugCode = {"\033[92m", LogLevel::DEBUG};
inline constexpr std::pair<const char*, LogLevel> infoCode = {"\033[97m", LogLevel::INFO};
inline constexpr std::pair<const char*, LogLevel> warnCode = {"\033[93m", LogLevel::WARN};
inline constexpr std::pair<const char*, LogLevel> errorCode = {"\033[91m", LogLevel::ERROR};

} // namespace ANSI

class Logger {
public:
	Logger(const std::pair<const char*, ANSI::LogLevel>& code)
	    : level_code(code.first)
	    , level(code.second) {}

	template <typename... Args> Logger& operator()(Args... args) {
		if (this->level < display_level)
			return *this;
		((std::cout << level_code) << ... << args) << ANSI::resetCode << std::endl;
		return *this;
	}

	static inline void setLoggerLevel(ANSI::LogLevel display) { display_level = display; }

private:
	const char* level_code;
	ANSI::LogLevel level;
	static ANSI::LogLevel display_level;
};

inline ANSI::LogLevel Logger::display_level = ANSI::LogLevel::INFO;

#define LOGGER_DEBUG Logger(ANSI::debugCode)
#define LOGGER_INFO Logger(ANSI::infoCode)
#define LOGGER_WARN Logger(ANSI::warnCode)
#define LOGGER_ERROR Logger(ANSI::errorCode)
