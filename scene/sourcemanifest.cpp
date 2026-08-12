/*
This Source Code Form is subject to the
terms of the Mozilla Public License, v.
2.0. If a copy of the MPL was not
distributed with this file, You can
obtain one at
http://mozilla.org/MPL/2.0/.
*/

#include "stdafx.h"
#include "scene/sourcemanifest.h"

#include "utilities/Globals.h"
#include "utilities/Logs.h"
#include "utilities/utilities.h"

#include <filesystem>

namespace scene {

source_manifest Sources;

namespace {

std::string const EU07_FILEEXTENSION_MANIFEST { ".sbt.sources" };

} // namespace

void source_manifest::clear() {

    m_entries.clear();
}

// only the scenery text shapes the twin. models, textures, sounds and vehicle physics are
// referenced by it but contribute nothing to the terrain, and treating them as sources
// would rebuild twins over edits that cannot possibly have changed them
bool source_manifest::is_scenery_text( std::string const &Path ) {

    auto const lowercase { ToLower( Path ) };
    for( auto const *extension : { ".scn", ".inc", ".scm", ".ctr" } ) {
        auto const length { std::strlen( extension ) };
        if( lowercase.size() >= length
         && lowercase.compare( lowercase.size() - length, length, extension ) == 0 ) {
            return true;
        }
    }
    return false;
}

bool source_manifest::stat_file( std::string const &Path, std::uint64_t &Size, std::int64_t &Modified ) {

    std::error_code error;
    auto const size { std::filesystem::file_size( Path, error ) };
    if( error ) { return false; }
    auto const written { std::filesystem::last_write_time( Path, error ) };
    if( error ) { return false; }

    Size = static_cast<std::uint64_t>( size );
    Modified = static_cast<std::int64_t>( written.time_since_epoch().count() );
    return true;
}

void source_manifest::record( std::string const &Path ) {

    if( false == is_scenery_text( Path ) ) { return; }

    // the same include can be pulled in by several files
    for( auto const &recorded : m_entries ) {
        if( recorded.path == Path ) { return; }
    }

    entry item;
    item.path = Path;
    if( false == stat_file( Path, item.size, item.modified ) ) {
        // a file the scenery names but which cannot be read is worth remembering all the
        // same: should it appear later, the twin has to be rebuilt
        item.size = 0;
        item.modified = 0;
    }
    m_entries.emplace_back( std::move( item ) );
}

std::string source_manifest::control_file( std::string const &Scenariofile ) {

    auto filename { Scenariofile };
    while( false == filename.empty() && filename[ 0 ] == '$' ) {
        // trim leading $ char rainsted utility may add to the base name for modified .scn files
        filename.erase( 0, 1 );
    }
    erase_extension( filename );
    return Global.asCurrentSceneryPath + filename + EU07_FILEEXTENSION_MANIFEST;
}

void source_manifest::write( std::string const &Scenariofile ) const {

    auto const filename { control_file( Scenariofile ) };
    std::ofstream output { filename, std::ios::trunc };
    if( false == output.is_open() ) {
        ErrorLog( "Failed to write scenery source list \"" + filename + "\"" );
        return;
    }

    output << "# sources of the binary terrain beside this file; delete either to force a rebuild\n";
    for( auto const &item : m_entries ) {
        output << item.size << ' ' << item.modified << ' ' << item.path << '\n';
    }

    WriteLog( "Recorded " + std::to_string( m_entries.size() ) + " scenery sources in \"" + filename + "\"" );
}

bool source_manifest::is_current( std::string const &Scenariofile ) const {

    auto const filename { control_file( Scenariofile ) };
    std::ifstream input { filename };
    if( false == input.is_open() ) {
        WriteLog( "No scenery source list beside the binary terrain, rebuilding it" );
        return false;
    }

    std::string line;
    std::size_t count { 0 };
    while( std::getline( input, line ) ) {

        if( line.empty() || line[ 0 ] == '#' ) { continue; }

        std::istringstream parser { line };
        std::uint64_t size { 0 };
        std::int64_t modified { 0 };
        if( false == static_cast<bool>( parser >> size >> modified ) ) {
            WriteLog( "Damaged scenery source list \"" + filename + "\", rebuilding the binary terrain" );
            return false;
        }
        // the rest of the line is the path, which may hold spaces
        std::string path;
        std::getline( parser, path );
        if( false == path.empty() && path[ 0 ] == ' ' ) { path.erase( 0, 1 ); }

        std::uint64_t currentsize { 0 };
        std::int64_t currentmodified { 0 };
        if( false == stat_file( path, currentsize, currentmodified ) ) {
            if( size != 0 || modified != 0 ) {
                WriteLog( "Scenery source \"" + path + "\" is gone, rebuilding the binary terrain" );
                return false;
            }
            // it was missing when the twin was built and it is missing still
            ++count;
            continue;
        }

        if( currentsize != size || currentmodified != modified ) {
            WriteLog( "Scenery source \"" + path + "\" changed, rebuilding the binary terrain" );
            return false;
        }
        ++count;
    }

    if( count == 0 ) {
        WriteLog( "Empty scenery source list \"" + filename + "\", rebuilding the binary terrain" );
        return false;
    }

    return true;
}

void record_source_file( std::string const &Path ) {

    Sources.record( Path );
}

} // namespace scene
