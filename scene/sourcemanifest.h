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
// source_manifest -- the scenery text files a .sbt was built from

// nothing in a .sbt says when the scenery text behind it was last touched, so an edited
// scenery would go on loading stale binary terrain. every scenery file opened while
// loading is noted down, and the list is written beside the .sbt as a plain text control
// file which the next load holds against what is on disk.
//
// the list sits next to the .sbt rather than inside it, which leaves the binary format
// untouched and something readable to look at when a rebuild happens for no obvious reason

namespace scene {

class source_manifest {

public:
// methods
    void clear();
    // ignores anything that is not scenery text
    void record( std::string const &Path );
    void write( std::string const &Scenariofile ) const;
    // a scenery with no control file counts as out of date: its .sbt predates this
    // bookkeeping and there is no telling what it was built from
    bool is_current( std::string const &Scenariofile ) const;

private:
// types
    struct entry {
        std::string path;
        std::uint64_t size { 0 };
        std::int64_t modified { 0 };
    };

// methods
    static std::string control_file( std::string const &Scenariofile );
    static bool is_scenery_text( std::string const &Path );
    static bool stat_file( std::string const &Path, std::uint64_t &Size, std::int64_t &Modified );

// members
    std::vector<entry> m_entries;
};

// the manifest of the scenery currently being read
extern source_manifest Sources;

} // namespace scene
