// see tests/stubs/stdafx.h
#pragma once

#include <string>

namespace scene {

// the parser notes scenery text files for the binary terrain bookkeeping; irrelevant here
struct source_manifest {
    void record( std::string const & ) {}
};

inline source_manifest Sources;

} // namespace scene
