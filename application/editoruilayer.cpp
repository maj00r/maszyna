/*
This Source Code Form is subject to the
terms of the Mozilla Public License, v.
2.0. If a copy of the MPL was not
distributed with this file, You can
obtain one at
http://mozilla.org/MPL/2.0/.
*/

#include "stdafx.h"
#include "application/editoruilayer.h"

#include "utilities/Globals.h"
#include "utilities/utilities.h"
#include "scene/scenenode.h"
#include "rendering/renderer.h"

editor_ui::editor_ui()
{

	clear_panels();
	// bind the panels with ui object. maybe not the best place for this but, eh

	add_external_panel(&m_itempropertiespanel);
	add_external_panel(&m_nodebankpanel);
	add_external_panel(&m_functionspanel);
	add_external_panel(&m_brushobjects);
	add_external_panel(&m_trackpanel);
}

bool editor_ui::track_place_active() const { return m_trackpanel.place_active; }
bool editor_ui::consume_track_finish() { bool const v = m_trackpanel.finish_chain; m_trackpanel.finish_chain = false; return v; }
bool editor_ui::consume_track_save() { bool const v = m_trackpanel.save_alignments; m_trackpanel.save_alignments = false; return v; }
int editor_ui::track_type() const { return m_trackpanel.type; }
float editor_ui::track_length() const { return m_trackpanel.length; }
float editor_ui::track_radius() const { return m_trackpanel.radius; }
float editor_ui::track_radius_start() const { return m_trackpanel.radius_start; }
float editor_ui::track_radius_end() const { return m_trackpanel.radius_end; }
bool editor_ui::track_curve_left() const { return m_trackpanel.curve_left; }
int editor_ui::track_cuts() const { return m_trackpanel.cuts; }
float editor_ui::track_cant() const { return m_trackpanel.cant; }
bool editor_ui::consume_track_apply_cant() { bool const v = m_trackpanel.apply_cant; m_trackpanel.apply_cant = false; return v; }
bool editor_ui::consume_track_apply_geometry() { bool const v = m_trackpanel.apply_geometry; m_trackpanel.apply_geometry = false; return v; }
bool editor_ui::track_diag() const { return m_trackpanel.diag; }
bool editor_ui::consume_track_dump() { bool const v = m_trackpanel.dump; m_trackpanel.dump = false; return v; }

// updates state of UI elements
void editor_ui::update()
{

	set_tooltip("");

	if (Global.ControlPicking && DebugModeFlag)
	{
		const auto sceneryNode = GfxRenderer->Pick_Node();
		const std::string content = sceneryNode ? sceneryNode->tooltip() : "";
		set_tooltip(content);
	}

	ui_layer::update();
	m_itempropertiespanel.update(m_node);
	m_functionspanel.update(m_node);

	auto ptr = get_active_node_template(true);
	if (ptr)
		m_brushobjects.update(*ptr);
}

void editor_ui::toggleBrushSettings(bool isVisible)
{
	if (m_brushobjects.is_open != isVisible)
		m_brushobjects.is_open = isVisible;
}

void editor_ui::set_node(scene::basic_node *Node)
{
	m_node = Node;
}

void editor_ui::add_node_template(const std::string &desc)
{
	m_nodebankpanel.add_template(desc);
}

std::string const *editor_ui::get_active_node_template(bool bypassRandom)
{
	if (!bypassRandom && m_brushobjects.is_open && m_brushobjects.useRandom && m_brushobjects.Objects.size() > 0)
	{
		return m_brushobjects.GetRandomObject();
	}
	return m_nodebankpanel.get_active_template();
}

nodebank_panel::edit_mode editor_ui::mode()
{
	return m_nodebankpanel.mode;
}
float editor_ui::getSpacing()
{
	return m_brushobjects.spacing;
}

functions_panel::rotation_mode editor_ui::rot_mode()
{
	return m_functionspanel.rot_mode;
}
float editor_ui::rot_val()
{
	return m_functionspanel.rot_value;
}
bool editor_ui::rot_from_last()
{
	return m_functionspanel.rot_from_last;
}