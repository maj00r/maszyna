/*
This Source Code Form is subject to the
terms of the Mozilla Public License, v.
2.0. If a copy of the MPL was not
distributed with this file, You can
obtain one at
http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <string>
#include <type_traits>

/////////////////////////////////////////////////////////////////////////////////////////////////////
// number conversion used when turning parser tokens into values

namespace parser_detail {

// scenery files consist mostly of numbers, and std::stringstream costs several hundred
// nanoseconds per token, so conversion goes through dedicated routines instead.
// results match what std::istream extraction produced before: unconvertible input
// yields 0, out of range input saturates
float toFloat( std::string const &Token );
double toDouble( std::string const &Token );
long long toSigned( std::string const &Token, long long const Min, long long const Max );
unsigned long long toUnsigned( std::string const &Token, unsigned long long const Max );

// character types are excluded on purpose: std::istream extracts those as single
// characters rather than as numbers, and that behaviour has to be preserved
template <typename Type_>
inline constexpr bool is_number_v =
    std::is_arithmetic_v<Type_>
    && false == std::is_same_v<Type_, bool>
    && false == std::is_same_v<Type_, char>
    && false == std::is_same_v<Type_, signed char>
    && false == std::is_same_v<Type_, unsigned char>
    && false == std::is_same_v<Type_, wchar_t>
    && false == std::is_same_v<Type_, char8_t>
    && false == std::is_same_v<Type_, char16_t>
    && false == std::is_same_v<Type_, char32_t>;

} // namespace parser_detail
