// Definitions behind tests/stubs/stdafx.h. The string helpers are copies of the
// utilities.cpp originals; logging and node grouping have no effect on tokenizing and
// are recorded or ignored.

#include "stdafx.h"
#include "utilities/Logs.h"
#include "scene/scenenodegroups.h"
#include "scene/sourcemanifest.h"
#include "utilities/parser.h"

#include <vector>

global_settings_stub Global;

namespace {
std::vector<std::string> g_errors;
}

std::vector<std::string> const & stub_errors() {

    return g_errors;
}

void stub_clear_errors() {

    g_errors.clear();
}

void WriteLog( std::string const & ) {}

void ErrorLog( std::string const &Message ) {

    g_errors.emplace_back( Message );
}

namespace scene {

node_groups_stub Groups;

void node_groups_stub::create() {}
void node_groups_stub::close() {}

} // namespace scene

std::string ToLower( std::string const &text ) {

    auto lowercase { text };
    std::transform( std::begin( text ), std::end( text ), std::begin( lowercase ), []( unsigned char c ) { return std::tolower( c ); } );
    return lowercase;
}

void replace_slashes( std::string &Filename ) {

    std::replace( Filename.begin(), Filename.end(), '\\', '/' );
}

bool contains( std::string_view const String, std::string_view Substring ) {

    return String.find( Substring ) != std::string_view::npos;
}

bool contains( std::string_view const String, char Character ) {

    return String.find( Character ) != std::string_view::npos;
}

std::string deserialize_random_set( cParser &Input, char const *Break ) {

    auto token { Input.getToken<std::string>( true, Break ) };
    std::replace( token.begin(), token.end(), '\\', '/' );
    if( token != "[" ) {
        return token;
    }
    std::vector<std::string> tokens;
    while( ( token = deserialize_random_set( Input, Break ) ) != "" && token != "]" ) {
        tokens.emplace_back( token );
    }
    if( false == tokens.empty() ) {
        std::shuffle( std::begin( tokens ), std::end( tokens ), Global.random_engine );
        return tokens.front();
    }
    return "";
}
