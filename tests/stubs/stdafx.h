// Stand-in for the project wide precompiled header, used only when building cParser
// into a test binary. The real one reaches the renderer interfaces and drags the whole
// engine in with it, which would make the parser untestable.
//
// Everything the parser actually touches outside itself is declared here, and the
// definitions in parsersupport.cpp mirror the engine ones. Keep both in step: a stub
// that drifts from utilities.cpp turns these tests into a fiction.

#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <glm/glm.hpp>

class cParser;

// utilities.h
std::string ToLower( std::string const &text );
void replace_slashes( std::string &Filename );
bool contains( std::string_view const String, std::string_view Substring );
bool contains( std::string_view const String, char Character );
std::string deserialize_random_set( cParser &Input, char const *Break = "\n\r\t ;" );

// Globals.h, narrowed to the two settings the parser reads
struct global_settings_stub {
    bool ParserLogIncludes { false };
    bool file_binary_terrain_state { false };
    std::mt19937 random_engine { 1234 };
};
extern global_settings_stub Global;
