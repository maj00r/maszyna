/*
This Source Code Form is subject to the
terms of the Mozilla Public License, v.
2.0. If a copy of the MPL was not
distributed with this file, You can
obtain one at
http://mozilla.org/MPL/2.0/.
*/

// parser_detail replaced std::istream extraction, which the scenery format had been
// relying on for two decades, so the primary check is a differential one: every input
// has to convert to exactly what std::stringstream produced. Pass a scenery file (or
// a directory) on the command line to run the same comparison over real data.

#include "utilities/numberparser.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace
{

int g_checks{0};
int g_failures{0};

// the behaviour every conversion below has to reproduce.
// NOTE: the target is zero initialized here, while the code this replaced passed in
// whatever the caller happened to hold. That only differs for input which fails before
// extraction begins -- an empty or whitespace only token -- where the stream left the
// target untouched. cParser never emits such a token, and reporting 0 beats leaving a
// caller's uninitialized variable alone, so that case is pinned down below instead
template <typename Type_> Type_ viaStringstream(std::string const &Token)
{
	std::stringstream converter(Token);
	Type_ output{};
	converter >> output;
	return output;
}

template <typename Type_> std::string toText(Type_ const Value)
{
	std::ostringstream out;
	out.precision(17);
	out << Value;
	return out.str();
}

template <typename Type_> void expectSame(char const *Kind, std::string const &Token, Type_ const Actual, char const *Origin)
{
	Type_ const expected{viaStringstream<Type_>(Token)};
	++g_checks;
	// NaN never compares equal to itself, so it needs an explicit case
	bool const same{expected == Actual || (expected != expected && Actual != Actual)};
	if (false == same)
	{
		if (++g_failures <= 25)
		{
			std::printf("  FAIL %-8s \"%s\" -> expected %s, got %s%s%s\n", Kind, Token.c_str(),
			            toText(expected).c_str(), toText(Actual).c_str(),
			            (Origin ? "   from " : ""), (Origin ? Origin : ""));
		}
	}
}

void checkToken(std::string const &Token, char const *Origin = nullptr)
{
	expectSame("float", Token, parser_detail::toFloat(Token), Origin);
	expectSame("double", Token, parser_detail::toDouble(Token), Origin);
	expectSame("int", Token,
	           static_cast<int>(parser_detail::toSigned(Token, std::numeric_limits<int>::lowest(), std::numeric_limits<int>::max())),
	           Origin);
	expectSame("unsigned", Token,
	           static_cast<unsigned>(parser_detail::toUnsigned(Token, std::numeric_limits<unsigned>::max())), Origin);
	expectSame("longlong", Token,
	           parser_detail::toSigned(Token, std::numeric_limits<long long>::lowest(), std::numeric_limits<long long>::max()),
	           Origin);
}

template <typename Type_> void expectValue(char const *Kind, std::string const &Token, Type_ const Actual, Type_ const Expected)
{
	++g_checks;
	bool const same{Expected == Actual || (Expected != Expected && Actual != Actual)};
	if (false == same)
	{
		++g_failures;
		std::printf("  FAIL %-8s \"%s\" -> expected %s, got %s\n", Kind, Token.c_str(), toText(Expected).c_str(),
		            toText(Actual).c_str());
	}
}

// inputs the scenery format actually produces, plus the malformed ones it survives
void testTokens()
{
	static char const *const tokens[]{
	    "0", "1", "-1", "+1", "1.5", "-1.5", "+1.5", "0.0001", "-0.0001", "-0.000100001",
	    "123.456", "-123.456", "1e10", "1e-10", "-1e10", "1E10", "+1e10", "007", "-0", "-0.0",
	    "3.4028235e38", "3.5e38", "-3.5e38", "1e400", "-1e400", "1e-400", "-1e-400",
	    "1e39", "1e-46", "1.17549e-38", "1e-320", "1e309", "0.", ".5", "-.5", "5.",
	    "2147483647", "2147483648", "-2147483648", "-2147483649", "4294967295", "4294967296",
	    "9223372036854775807", "9223372036854775808", "-9223372036854775808", "-9223372036854775809",
	    "000000000000000000001", "99999999999999999999999999", "-99999999999999999999999999",
	    "12345678901234567890",
	    // malformed or non numeric input, which the format is full of
	    "none", "", " ", "  7  ", "\t42", "abc", "1abc", "abc1", "--1", "++1", "-", "+", ".",
	    "0x10", "0X10", "-0x10", "1,5", "1 5", "inf", "-inf", "nan", "-nan", "INF", "NAN",
	    "infinity", "true", "1.0f", "1#QNAN", "1e", "1e+", "1e-", "1.5e", "1E", "1e10x",
	    ".e5", "1.2.3", "1-2", "5e-"};

	for (auto const *token : tokens)
	{
		checkToken(token);
	}
}

// a sweep over the shape of values terrain and track geometry is made of
void testGeneratedValues()
{
	std::srand(20260812);
	char buffer[64];
	for (int i = 0; i < 100000; ++i)
	{
		double const value{(std::rand() / static_cast<double>(RAND_MAX) - 0.5) * 2e5};
		std::snprintf(buffer, sizeof(buffer), "%.*f", i % 8, value);
		checkToken(buffer);
		std::snprintf(buffer, sizeof(buffer), "%.6e", value);
		checkToken(buffer);
		std::snprintf(buffer, sizeof(buffer), "%d", std::rand() - RAND_MAX / 2);
		checkToken(buffer);
	}
}

// behaviour worth stating outright rather than inferring from the reference
void testDocumentedBehaviour()
{
	// input that cannot be converted reports 0, including the degenerate tokens the
	// stream used to skip over entirely
	expectValue<float>("float", "", parser_detail::toFloat(""), 0.f);
	expectValue<float>("float", " ", parser_detail::toFloat(" "), 0.f);
	expectValue<int>("int", "", static_cast<int>(parser_detail::toSigned("", -2147483647 - 1, 2147483647)), 0);
	expectValue<float>("float", "none", parser_detail::toFloat("none"), 0.f);
	expectValue<float>("float", "nan", parser_detail::toFloat("nan"), 0.f);
	expectValue<float>("float", "inf", parser_detail::toFloat("inf"), 0.f);
	expectValue<float>("float", "1e-400", parser_detail::toFloat("1e-400"), 0.f);
	expectValue<float>("float", "+1.5", parser_detail::toFloat("+1.5"), 1.5f);
	expectValue<double>("double", "-1.5", parser_detail::toDouble("-1.5"), -1.5);
	expectValue<int>("int", "42x", static_cast<int>(parser_detail::toSigned("42x", -2147483647 - 1, 2147483647)), 42);
	// saturation at the width of the target type rather than a wrap around
	expectValue<int>("int", "2147483648", static_cast<int>(parser_detail::toSigned("2147483648", -2147483647 - 1, 2147483647)),
	                 2147483647);
	expectValue<unsigned>("unsigned", "4294967296",
	                      static_cast<unsigned>(parser_detail::toUnsigned("4294967296", 4294967295u)), 4294967295u);
}

// optional: replay real scenery data through both implementations
void testFiles(std::vector<std::string> const &Paths)
{
	std::array<bool, 256> breaks{};
	for (unsigned char c : std::string_view("\n\r\t ;"))
	{
		breaks[c] = true;
	}

	for (auto const &path : Paths)
	{
		std::ifstream file(path, std::ios::binary | std::ios::ate);
		if (false == file.is_open())
		{
			std::printf("  cannot open \"%s\"\n", path.c_str());
			++g_failures;
			continue;
		}
		std::streamoff const size{file.tellg()};
		file.seekg(0);
		std::string data(static_cast<std::size_t>(size), '\0');
		file.read(data.data(), size);

		int const before{g_failures};
		long long count{0};
		char const *position{data.data()};
		char const *const end{position + data.size()};
		std::string token;
		while (position < end)
		{
			while (position < end && breaks[static_cast<unsigned char>(*position)])
			{
				++position;
			}
			char const *const start{position};
			while (position < end && false == breaks[static_cast<unsigned char>(*position)])
			{
				++position;
			}
			if (position == start)
			{
				break;
			}
			token.assign(start, position - start);
			++count;
			checkToken(token, path.c_str());
		}
		std::printf("  %-50s %10lld tokens  %s\n", std::filesystem::path(path).filename().string().c_str(), count,
		            (g_failures == before ? "ok" : "FAILED"));
	}
}

} // namespace

int main(int argc, char **argv)
{
	std::printf("parser_detail conversions vs std::istream extraction\n");

	testTokens();
	testGeneratedValues();
	testDocumentedBehaviour();

	std::vector<std::string> files;
	for (int i = 1; i < argc; ++i)
	{
		std::filesystem::path const path{argv[i]};
		if (true == std::filesystem::is_directory(path))
		{
			for (auto const &entry : std::filesystem::recursive_directory_iterator(path))
			{
				auto const extension{entry.path().extension().string()};
				if (true == entry.is_regular_file() && (extension == ".scn" || extension == ".scm" || extension == ".inc"))
				{
					files.emplace_back(entry.path().string());
				}
			}
		}
		else
		{
			files.emplace_back(path.string());
		}
	}
	if (false == files.empty())
	{
		testFiles(files);
	}

	std::printf("%d checks, %d failures\n", g_checks, g_failures);
	return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
