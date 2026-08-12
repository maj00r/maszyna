// see tests/stubs/stdafx.h
#pragma once

namespace scene {

// the parser opens a node group per *.inc file; grouping has no bearing on tokenizing
struct node_groups_stub {
    void create();
    void close();
};

extern node_groups_stub Groups;

} // namespace scene
