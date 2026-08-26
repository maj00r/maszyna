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

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <format>
#include <string_view>

namespace scene {

namespace {

std::string const EU07_FILEEXTENSION_MANIFEST { ".sbt.sources" };

} // namespace

void source_manifest::clear() {

    m_entries.clear();
}

// only the scenery text shapes the terrain. models, textures, sounds and vehicle physics
// are referenced by a scenery but put nothing into the region, and treating them as
// sources would force rebuilds over edits that cannot have changed it
bool source_manifest::is_scenery_text( std::string const &Path ) {

    std::filesystem::path const lowercase { ToLower( Path ) };
    auto const normalised { lowercase.lexically_normal().generic_string() };
    if( auto const extensions = { ".scn", ".inc", ".scm" };
        std::ranges::none_of(
            extensions,
            [&normalised]( auto const *extension ) { return normalised.ends_with( extension ); } ) ) {
        return false;
    }

    return normalised.starts_with( ToLower( Global.asCurrentSceneryPath ) );
}

bool source_manifest::stat_file( std::filesystem::path const &Path, std::uint64_t &Size, std::int64_t &Modified ) {

    std::error_code error;
    auto const size { std::filesystem::file_size( Path, error ) };
    if( error ) { return false; }
    auto const written { std::filesystem::last_write_time( Path, error ) };
    if( error ) { return false; }

    Size = static_cast<std::uint64_t>( size );
    auto const written_utc { std::chrono::clock_cast<std::chrono::system_clock>( written ) };
    Modified = std::chrono::duration_cast<std::chrono::seconds>( written_utc.time_since_epoch() ).count();
    return true;
}

std::uint64_t source_manifest::hash_file( std::string const &Path ) {

    std::ifstream input { Path, std::ios::binary };
    if( false == input.is_open() ) { return 0; }

    // fnv-1a over 64 bit words. not a cryptographic digest, and it does not need to be:
    // it only has to tell an edited file from an untouched one
    std::uint64_t digest { 14695981039346656037ULL };
    std::vector<char> buffer( 64 * 1024 );

    while( input ) {
        input.read( buffer.data(), static_cast<std::streamsize>( buffer.size() ) );
        auto const read { static_cast<std::size_t>( input.gcount() ) };
        std::size_t offset { 0 };
        for( ; offset + sizeof( std::uint64_t ) <= read; offset += sizeof( std::uint64_t ) ) {
            std::uint64_t word;
            std::memcpy( &word, buffer.data() + offset, sizeof( word ) );
            digest = ( digest ^ word ) * 1099511628211ULL;
        }
        for( ; offset < read; ++offset ) {
            digest = ( digest ^ static_cast<unsigned char>( buffer[ offset ] ) ) * 1099511628211ULL;
        }
    }
    // a file which happens to digest to zero would look like one that could not be read
    return ( digest == 0 ? 1 : digest );
}

void source_manifest::record( std::string const &Filepath ) {

    if( false == is_scenery_text( Filepath ) ) { return; }

    auto const Path { original( Filepath ) };

    // the same include can be pulled in by several files
    for( auto const &recorded : m_entries ) {
        if( recorded.path == Path ) { return; }
    }

    entry item;
    item.path = Path;
    if( false == stat_file( Path, item.size, item.modified ) ) {
        // a file the scenery names but cannot read is worth remembering all the same:
        // should it appear later, the terrain has to be rebuilt
        item.size = 0;
        item.modified = 0;
    }
    else {
        item.content = hash_file( Path );
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

std::string source_manifest::original( std::string const &Path ) {

    std::filesystem::path const path { Path };
    auto const filename { path.filename().string() };

    auto const trimmed { filename.find_first_not_of( '$' ) };
    if( trimmed == 0 ) { return Path; }

    auto const name { trimmed == std::string::npos ? std::string_view {} : std::string_view( filename ).substr( trimmed ) };
    auto const stripped { ( path.parent_path() / name ).generic_string() };
    return ( FileExists( stripped ) ? stripped : Path );
}

void source_manifest::write_entries( std::string const &Filename, std::vector<entry> const &Entries ) {

    std::ofstream output { Filename, std::ios::trunc };
    if( false == output.is_open() ) {
        ErrorLog( "Failed to write scenery source list \"" + Filename + "\"" );
        return;
    }

    output << "# sources of the binary terrain beside this file; delete either to force a rebuild\n";
    output << "# size modified contenthash path\n";
    for( auto const &item : Entries ) {
        output << item.size << ' ' << item.modified << ' ' << item.content << ' ' << item.path << '\n';
    }
}

void source_manifest::write( std::string const &Scenariofile ) const {

    auto const filename { control_file( Scenariofile ) };
    write_entries( filename, m_entries );
    WriteLog( std::format( "Recorded {} scenery sources in \"{}\"", m_entries.size(), filename ) );
}

bool source_manifest::is_current( std::string const &Scenariofile ) const {

    auto const filename { control_file( Scenariofile ) };
    std::ifstream input { filename };
    if( false == input.is_open() ) {
        WriteLog( "No source list beside binary terrain \"" + filename + "\", treating it as out of date" );
        return false;
    }

    std::vector<entry> recorded;
    std::string line;
    while( std::getline( input, line ) ) {

        if( line.empty() || line[ 0 ] == '#' ) { continue; }

        std::istringstream parser { line };
        entry item;
        if( false == static_cast<bool>( parser >> item.size >> item.modified >> item.content ) ) {
            WriteLog( "Damaged source list \"" + filename + "\", treating the binary terrain as out of date" );
            return false;
        }
        // the rest of the line is the path, which may hold spaces
        std::getline( parser, item.path );
        if( false == item.path.empty() && item.path[ 0 ] == ' ' ) { item.path.erase( 0, 1 ); }
        recorded.emplace_back( std::move( item ) );
    }

    if( true == recorded.empty() ) {
        WriteLog( "Empty source list \"" + filename + "\", treating the binary terrain as out of date" );
        return false;
    }

    auto refreshed { false };
    for( auto &item : recorded ) {

        std::uint64_t currentsize { 0 };
        std::int64_t currentmodified { 0 };
        if( false == stat_file( item.path, currentsize, currentmodified ) ) {
            if( item.size != 0 || item.modified != 0 ) {
                WriteLog( "Scenery source \"" + item.path + "\" is gone, the binary terrain is out of date" );
                return false;
            }
            // it was missing when the terrain was built and it is missing still
            continue;
        }

        if( currentsize == item.size && currentmodified == item.modified ) {
            // untouched, and worth nothing more than the two numbers just compared
            continue;
        }

        if( currentsize != item.size || hash_file( item.path ) != item.content ) {
            WriteLog( "Scenery source \"" + item.path + "\" changed, the binary terrain is out of date" );
            return false;
        }

        // same contents under a new timestamp, which a checkout or a copied installation
        // leaves behind. note the new timestamp so the file is not read again next time
        item.modified = currentmodified;
        refreshed = true;
    }

    if( true == refreshed ) {
        WriteLog( "Scenery sources carry new timestamps but unchanged contents, keeping the binary terrain" );
        write_entries( filename, recorded );
    }

    return true;
}

source_manifest &sources() {

    static source_manifest manifest;
    return manifest;
}

} // namespace scene
