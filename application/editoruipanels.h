/*
This Source Code Form is subject to the
terms of the Mozilla Public License, v.
2.0. If a copy of the MPL was not
distributed with this file, You can
obtain one at
http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include "application/uilayer.h"
#include "utilities/Classes.h"
/*
// helper, associated bool is set when the primary value was changed and expects processing at the observer's leisure
template<typename Type_>
using changeable = std::pair<Type_, bool>;

// helper, holds a set of changeable properties for a scene node
struct item_properties {

    scene::basic_node const *node { nullptr }; // properties' owner

    changeable<std::string> name {};
    changeable<glm::dvec3> location {};
    changeable<glm::vec3> rotation {};
};
*/
class itemproperties_panel : public ui_panel
{

  public:
	itemproperties_panel(std::string const &Name, bool const Isopen) : ui_panel(Name, Isopen) {}

	// non-const node pointer so the transform editor in render() can mutate
	// position/rotation/scale of TAnimModel nodes in place. Other node types
	// are still treated as read-only.
	void update(scene::basic_node *Node);
	void render() override;

  private:
	// methods
	void update_group();
	bool render_group();
	// renders DragFloat3/DragScalarN widgets for position, rotation and scale
	// of the currently bound TAnimModel; no-op for other node subclasses.
	void render_transform_editor();

	// members
	scene::basic_node *m_node{nullptr}; // scene node bound to the panel
	scene::group_handle m_grouphandle{null_handle}; // scene group bound to the panel
	std::string m_groupprefix;
	std::vector<text_line> m_grouplines;
};

class brush_object_list : public ui_panel
{
  private:
	int idx;
	static bool VectorGetter(void *data, int idx, const char **out_text);
	std::string Template;

  public:
	brush_object_list(std::string const &Name, bool const Isopen);
	void render() override;
	void update(std::string nodeTemplate);

	// class use
	std::vector<std::string> Objects;
	std::string *GetRandomObject();
	bool useRandom = {false};
	float spacing{1.0f};
};

class nodebank_panel : public ui_panel
{

  public:
	enum edit_mode
	{
		MODIFY,
		COPY,
		ADD,
		BRUSH
	};
	edit_mode mode = MODIFY;

	nodebank_panel(std::string const &Name, bool const Isopen);
	void nodebank_reload();
	void render() override;
	void add_template(const std::string &desc);
	const std::string *get_active_template();

  private:
	// methods:
	std::string generate_node_label(std::string Input) const;
	// members:
	std::vector<std::pair<std::string, std::shared_ptr<std::string>>> m_nodebank;
	char m_nodesearch[128];
	std::shared_ptr<std::string> m_selectedtemplate;
};

// track-laying tool: pick a track type and parameters, then place tracks by clicking in the scene
class track_panel : public ui_panel
{

  public:
	enum track_type
	{
		STRAIGHT,
		ARC,
		TRANSITION,
		SWITCH
	};
	track_type type = STRAIGHT;
	float length = 50.0f;   // [m]
	float radius = 300.0f;  // [m], used by ARC / SWITCH
	// TRANSITION (clothoid): curvature changes linearly from 1/radius_start to 1/radius_end.
	// a radius of 0 means "straight" (infinite radius, zero curvature) at that end.
	float radius_start = 0.0f;   // [m], 0 = straight
	float radius_end = 300.0f;   // [m], 0 = straight
	bool curve_left = true; // ARC/TRANSITION/SWITCH curve direction
	int cuts = 0; // STRAIGHT: mid-element cuts (element emits cuts+1 track pieces)
	float cant = 0.0f; // superelevation [mm] applied to new elements / the selected track
	bool place_active = false; // when set, a left click in the scene lays a track
	bool finish_chain = false; // one-shot: user asked to end the current niweleta
	bool save_alignments = false; // one-shot: user asked to save the niweleta + scene
	bool apply_cant = false; // one-shot: user asked to set cant on the selected track
	bool apply_geometry = false; // one-shot: apply length/radius fields to the selected niweleta element
	bool diag = false; // diagnostics overlay for the niweleta (continuity angles, gaps, element info)
	bool dump = false; // one-shot: user asked to dump the niweleta state to the log

	track_panel(std::string const &Name, bool const Isopen) : ui_panel(Name, Isopen) {}
	void render() override;
};

class functions_panel : public ui_panel
{

  public:
	enum rotation_mode
	{
		RANDOM,
		FIXED,
		DEFAULT
	};
	rotation_mode rot_mode = DEFAULT;

	float rot_value = 0.0f;
	bool rot_from_last = false;

	functions_panel(std::string const &Name, bool const Isopen) : ui_panel(Name, Isopen) {}

	void update(scene::basic_node const *Node);
	void render() override;

  private:
	// methods

	// members
	scene::basic_node const *m_node{nullptr}; // scene node bound to the panel
	scene::group_handle m_grouphandle{null_handle}; // scene group bound to the panel
	std::string m_groupprefix;
	std::vector<text_line> m_grouplines;
};