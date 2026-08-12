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

std::uint64_t source_manifest::hash_file( std::string const &Path ) {

    std::ifstream input { Path, std::ios::binary };
    if( false == input.is_open() ) { return 0; }

    // fnv-1a over 64 bit words. not a cryptographic digest, and it does not need to be:
    // it only has to tell an edited file from an untouched one
    std::uint64_t digest { 14695981039346656037ull };
    std::vector<char> buffer( 64 * 1024 );

    while( input ) {
        input.read( buffer.data(), static_cast<std::streamsize>( buffer.size() ) );
        auto const read { static_cast<std::size_t>( input.gcount() ) };
        std::size_t offset { 0 };
        for( ; offset + sizeof( std::uint64_t ) <= read; offset += sizeof( std::uint64_t ) ) {
            std::uint64_t word;
            std::memcpy( &word, buffer.data() + offset, sizeof( word ) );
            digest = ( digest ^ word ) * 1099511628211ull;
        }
        for( ; offset < read; ++offset ) {
            digest = ( digest ^ static_cast<unsigned char>( buffer[ offset ] ) ) * 1099511628211ull;
        }
    }
    // a file which happens to digest to zero would look like one that could not be read
    return ( digest == 0 ? 1 : digest );
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
    WriteLog( "Recorded " + std::to_string( m_entries.size() ) + " scenery sources in \"" + filename + "\"" );
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
            // it was missing when the twin was built and it is missing still
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

void record_source_file( std::string const &Path ) {

    Sources.record( Path );
}

} // namespace scene
