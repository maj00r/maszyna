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
#include <vector>

/////////////////////////////////////////////////////////////////////////////////////////////////////
// keeps track of the text files a binary terrain twin was built from

// a twin is only as good as the scenery text it came from, and nothing about the twin
// itself says when that text was last touched. every scenery file opened while loading is
// noted down, and the list is written beside the twin as a plain text control file. on the
// next load the recorded sizes and timestamps are held against what is on disk, so an
// edited scenery rebuilds its twin instead of quietly loading the stale one.
//
// the control file sits next to the twin rather than inside it, which keeps the binary
// format untouched and leaves something readable to look at when a twin rebuilds and it is
// not obvious why

namespace scene {

class source_manifest {

public:
// methods
    // forgets the files gathered so far. called when a scenery starts loading
    void clear();
    // notes a file the scenery was read from. ignores anything that is not scenery text
    void record( std::string const &Path );
    // writes the gathered list beside the twin of specified scenery
    void write( std::string const &Scenariofile ) const;
    // true if the control file of specified scenery matches what is on disk now. a scenery
    // with no control file counts as out of date: its twin predates this bookkeeping and
    // there is no telling what it was built from
    bool is_current( std::string const &Scenariofile ) const;

private:
// types
    struct entry {
        std::string path;
        std::uint64_t size { 0 };
        std::int64_t modified { 0 };
        std::uint64_t content { 0 };
    };

// methods
    static std::string control_file( std::string const &Scenariofile );
    static bool is_scenery_text( std::string const &Path );
    static bool stat_file( std::string const &Path, std::uint64_t &Size, std::int64_t &Modified );
    // digest of the file contents. a checkout or a copied installation rewrites
    // modification times without touching content, and rebuilding a twin costs a hundred
    // times more than reading the sources back, so a file is only given up on once its
    // contents really differ
    static std::uint64_t hash_file( std::string const &Path );
    static void write_entries( std::string const &Filename, std::vector<entry> const &Entries );

// members
    std::vector<entry> m_entries;
};

extern source_manifest Sources;

// records a file the current scenery is being read from. called by the parser
void record_source_file( std::string const &Path );

} // namespace scene
