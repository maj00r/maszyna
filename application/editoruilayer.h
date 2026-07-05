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
#include "application/editoruipanels.h"

namespace scene
{

class basic_node;

}

class editor_ui : public ui_layer
{

  public:
	// constructors
	editor_ui();
	// methods
	// updates state of UI elements
	void update() override;
	void set_node(scene::basic_node *Node);
	void add_node_template(const std::string &desc);
	float rot_val();
	bool rot_from_last();
	functions_panel::rotation_mode rot_mode();
	const std::string *get_active_node_template(bool bypassRandom = false);
	nodebank_panel::edit_mode mode();
	float getSpacing();
	void toggleBrushSettings(bool isVisible);
	// track-laying tool state (see track_panel)
	bool track_place_active() const;
	int track_type() const;
	float track_length() const;
	float track_radius() const;
	float track_radius_start() const;
	float track_radius_end() const;
	bool track_curve_left() const;
	int track_cuts() const;
	// one-shot: true when the user clicked "finish niweleta"; clears the flag
	bool consume_track_finish();
	// one-shot: true when the user clicked "save niweleta + scene"; clears the flag
	bool consume_track_save();

  private:
	// members
	itemproperties_panel m_itempropertiespanel{"Node Properties", true};
	functions_panel m_functionspanel{"Functions", true};
	nodebank_panel m_nodebankpanel{"Node Bank", true};
	brush_object_list m_brushobjects{"Brush properties", false};
	track_panel m_trackpanel{"Tory", true};
	scene::basic_node *m_node{nullptr}; // currently bound scene node, if any
};
