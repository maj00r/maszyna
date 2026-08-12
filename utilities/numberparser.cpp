/*
This Source Code Form is subject to the
terms of the Mozilla Public License, v.
2.0. If a copy of the MPL was not
distributed with this file, You can
obtain one at
http://mozilla.org/MPL/2.0/.
*/

// NOTE: kept free of engine dependencies so it can be built and tested on its own

#include "utilities/numberparser.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdlib>
#include <limits>

namespace parser_detail
{
namespace
{
// std::num_get skips leading whitespace and accepts a leading '+', neither of which
// the conversion routines below do on their own
char const *trimSign(std::string const &Token, char const *&End)
{
	char const *begin = Token.data();
	End = begin + Token.size();
	while (begin != End && std::isspace(static_cast<unsigned char>(*begin)))
	{
		++begin;
	}
	if (begin != End && *begin == '+')
	{
		++begin;
	}
	return begin;
}

// std::num_get accepts a decimal digit or a decimal point after the optional sign and
// nothing else, so "inf" and "nan" -- which both from_chars and strtod would take --
// have to keep converting to zero
bool startsNumber(char const *Begin, char const *End)
{
	if (Begin != End && *Begin == '-')
	{
		++Begin;
	}
	return Begin != End && (0 != std::isdigit(static_cast<unsigned char>(*Begin)) || *Begin == '.');
}

bool startsHex(char const *Begin, char const *End)
{
	if (Begin != End && *Begin == '-')
	{
		++Begin;
	}
	return End - Begin >= 2 && Begin[0] == '0' && (Begin[1] == 'x' || Begin[1] == 'X');
}

template <typename Type_> Type_ strtoFloating(char const *Begin, char **Stop)
{
	if constexpr (std::is_same_v<Type_, float>)
	{
		return std::strtof(Begin, Stop);
	}
	else
	{
		return std::strtod(Begin, Stop);
	}
}

template <typename Type_> Type_ toFloating(std::string const &Token)
{
	char const *end = nullptr;
	char const *begin = trimSign(Token, end);
	if (false == startsNumber(begin, end))
	{
		return Type_{0};
	}
	if (true == startsHex(begin, end))
	{
		// hexadecimal notation, which std::num_get reads but from_chars spells without the prefix
		return strtoFloating<Type_>(begin, nullptr);
	}
#if defined(__cpp_lib_to_chars)
	Type_ output{0};
	auto const result{std::from_chars(begin, end, output)};
	if (result.ec == std::errc::result_out_of_range)
	{
		// overflow has to yield infinity and underflow zero, which strtod resolves
		// the same way std::num_get does; both are rare enough to pay for a reparse
		return strtoFloating<Type_>(begin, nullptr);
	}
	if (result.ec != std::errc())
	{
		return Type_{0};
	}
	// std::num_get takes the exponent marker greedily and fails when no digits follow it
	if (result.ptr != end && (*result.ptr == 'e' || *result.ptr == 'E'))
	{
		return Type_{0};
	}
	return output;
#else
	// NOTE: fallback for toolchains without floating point std::from_chars. unlike
	// from_chars this depends on LC_NUMERIC, which the simulator leaves at "C"
	char *stop = nullptr;
	Type_ const output{strtoFloating<Type_>(begin, &stop)};
	if (stop != end && (*stop == 'e' || *stop == 'E'))
	{
		return Type_{0};
	}
	return output;
#endif
}
} // namespace

float toFloat(std::string const &Token)
{
	return toFloating<float>(Token);
}

double toDouble(std::string const &Token)
{
	return toFloating<double>(Token);
}

long long toSigned(std::string const &Token, long long const Min, long long const Max)
{
	char const *end = nullptr;
	char const *begin = trimSign(Token, end);
	long long output{0};
	auto const result{std::from_chars(begin, end, output, 10)};
	if (result.ec == std::errc::result_out_of_range)
	{
		return (begin != end && *begin == '-' ? Min : Max);
	}
	if (result.ec != std::errc())
	{
		return 0;
	}
	return std::clamp(output, Min, Max);
}

unsigned long long toUnsigned(std::string const &Token, unsigned long long const Max)
{
	char const *end = nullptr;
	char const *begin = trimSign(Token, end);
	if (begin != end && *begin == '-')
	{
		// std::num_get reads a negative value and wraps it around the width of the
		// target type, saturating instead once its magnitude no longer fits.
		// Max is always one less than a power of two, so masking performs the wrap
		long long negative{0};
		auto const negativeresult{std::from_chars(begin, end, negative, 10)};
		if (negativeresult.ec == std::errc::result_out_of_range)
		{
			return Max;
		}
		if (negativeresult.ec != std::errc())
		{
			return 0;
		}
		unsigned long long const magnitude{0ull - static_cast<unsigned long long>(negative)};
		return (magnitude > Max ? Max : (0ull - magnitude) & Max);
	}
	unsigned long long output{0};
	auto const result{std::from_chars(begin, end, output, 10)};
	if (result.ec == std::errc::result_out_of_range)
	{
		return Max;
	}
	if (result.ec != std::errc())
	{
		return 0;
	}
	return std::min(output, Max);
}
} // namespace parser_detail
