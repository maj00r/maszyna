/*
This Source Code Form is subject to the
terms of the Mozilla Public License, v.
2.0. If a copy of the MPL was not
distributed with this file, You can
obtain one at
http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <unordered_map>
#include <string>

#include "application/applicationmode.h"
#include "input/editormouseinput.h"
#include "input/editorkeyboardinput.h"
#include "vehicle/Camera.h"
#include "scene/sceneeditor.h"
#include "scene/scenenode.h"
#include "editor/editorTerrain.hpp"
#include "editor/editorTerrainStreamer.hpp"

#include <memory>

class editor_mode : public application_mode
{

  public:
	// constructors
	editor_mode();
	// methods
	// initializes internal data structures of the mode. returns: true on success, false otherwise
	bool init() override;
	// mode-specific update of simulation data. returns: false on error, true otherwise
	bool update() override;
	// maintenance method, called when the mode is activated
	void enter() override;
	// maintenance method, called when the mode is deactivated
	void exit() override;
	// input handlers
	void on_key(int Key, int Scancode, int Action, int Mods) override;
	void on_cursor_pos(double Horizontal, double Vertical) override;
	void on_mouse_button(int Button, int Action, int Mods) override;
	void on_scroll(double const Xoffset, double const Yoffset) override
	{
		;
	}
	void on_window_resize(int w, int h) override
	{
		;
	}
	void on_event_poll() override;
	bool is_command_processor() const override;
	void undo_last();
	static bool focus_active();
	static void  set_focus_active(bool isActive);
	static TCamera& get_camera() { return Camera; }
	static bool change_history() { return m_change_history; }
	static void set_change_history(bool enabled) { m_change_history = enabled; }
	static bool settings_open() { return m_settings_open; }
	static void set_settings_open(bool enabled) { m_settings_open = enabled; }
  private:
	// types
	struct editormode_input
	{

		editormouse_input mouse;
		editorkeyboard_input keyboard;

		bool init();
		void poll();
	};

	struct state_backup
	{

		TCamera camera;
		bool freefly;
		bool picking;
	};

	struct EditorSnapshot
	{
		enum class Action { Move, Rotate, Scale, Add, Delete, Other };

		Action action{Action::Other};
		std::string node_name;          // node identifier (basic_node::name())
		// direct pointer to node when available; used for in-memory undo/redo lookup
		scene::basic_node *node_ptr{nullptr};
		std::string serialized;         // full text for recreate (used for Add/Delete)
		glm::dvec3 position{0.0, 0.0, 0.0};
		glm::vec3 rotation{0.0f, 0.0f, 0.0f};
		glm::vec3 scale{1.0f, 1.0f, 1.0f};
		UID uuid; // node UUID for reference, used as fallback lookup for deleted/recreated nodes

	};
	void push_snapshot(scene::basic_node *node, EditorSnapshot::Action Action = EditorSnapshot::Action::Move, std::string const &Serialized = std::string());

	std::vector<EditorSnapshot> m_history; // history of changes to nodes, used for undo functionality
	std::vector<EditorSnapshot> g_redo;
	// methods
	void update_camera(double const Deltatime);

	editor_ui *ui() const;
	void redo_last();
	void handle_brush_mouse_hold(int Action, int Button);
	void apply_rotation_for_new_node(scene::basic_node *node, int rotation_mode, float fixed_rotation_value);
	// track tool: create a track (type/params from the track panel) starting at a point, along a direction
	void create_track_at(glm::dvec3 const &start, glm::dvec3 const &dir);
	// snaps start point+direction to the nearest existing track endpoint within reach (for connecting tracks)
	bool snap_track_start(glm::dvec3 &start, glm::dvec3 &dir);
	// builds a track node from bezier control offsets and commits it (create_track + select)
	TTrack *commit_track(glm::dvec3 const &p1, glm::dvec3 const &cv1, glm::dvec3 const &cv2, glm::dvec3 const &p2, double radius, double length);
	// builds a switch node (straight main path + diverging arc path sharing the entry point)
	TTrack *commit_switch(glm::dvec3 const &entry, glm::dvec3 const &straightend, glm::dvec3 const &divcv1, glm::dvec3 const &divcv2, glm::dvec3 const &divend, double radius, double length);
	// removes a track from the scene (cell + baked section geometry + path table + labels)
	void delete_track(TTrack *track);

	// --- rainsted-style endpoint editing: tracks are typed segments (straight/arc/transition),
	// new tracks extend from an existing endpoint (incl. a switch's P4), and geometry is edited
	// by moving endpoints; tracks sharing the moved point are re-fitted to keep continuity ---
	// finds a track endpoint close to the given screen position
	bool pick_track_endpoint(float screenx, float screeny, bool allowswitch, TTrack *&track, int &endindex, glm::dvec3 &point, glm::dvec3 &outward);

	// --- niweleta: an ordered chain of typed elements (straight / arc / transition curve),
	// marched from an origin+direction; tracks are the generated result layer (P1->P2 always
	// follows the march, preserving track directionality). A switch is a branch point: new
	// chains anchor at its outlets and re-march when the switch is moved. ---
	struct chain_element {
		int type { 0 };           // track_panel::track_type
		double length { 50.0 };   // [m] element arc length
		double radius { 300.0 };  // ARC radius / TRANSITION end radius (0 = straight)
		double radius0 { 0.0 };   // TRANSITION start radius (0 = straight)
		bool left { true };
		int cuts { 0 }; // straight only: number of mid-element cuts (emits cuts+1 track pieces)
		std::vector<TTrack *> tracks; // generated result (straights may emit several pieces)
	};
	struct track_chain {
		glm::dvec3 origin { 0.0 };               // start point (node base level)
		glm::dvec3 direction { 0.0, 0.0, -1.0 }; // unit start tangent
		TTrack *anchor { nullptr };  // optional switch feeding this chain
		int anchor_end { -1 };       // switch endpoint index (1 = main end, 3 = diverging end)
		std::vector<chain_element> elements;
		std::vector<glm::dvec3> joints; // cached: origin + each element end (railhead-free base points)
		glm::dvec3 endtangent { 0.0, 0.0, -1.0 }; // cached march exit tangent (for branching / extending)
	};
	struct switch_meta { glm::dvec3 entry, straightend, divcv1, divcv2, divend; double radius, length; };
	// marches one element from (pos, dir), optionally emitting the result track
	TTrack *march_element(chain_element &el, glm::dvec3 &pos, glm::dvec3 &dir, bool emit);
	// regenerates a chain: re-marches all elements from the (possibly anchored) origin
	void regenerate_chain(int index);
	// finds a chain joint close to the given screen position; joint 0 = chain origin
	bool pick_chain_joint(float screenx, float screeny, int &chain, int &joint);
	// lays a transition curve (clothoid) as a chain of short arc segments; curvature goes linearly
	// from kappa_start to kappa_end (1/radius; 0 == straight)
	void create_transition(glm::dvec3 const &start, glm::dvec3 const &tangent, double length, double kappa_start, double kappa_end, bool left);
	// convenience: track ahead of the editor camera (hotkey)
	void create_straight_track_ahead(double length = 50.0);
	// world point where the cursor ray meets the ground plane (y=0); deterministic, unlike a
	// geometry raycast, so track placement/snap doesn't depend on what's under the cursor
	glm::dvec3 cursor_ground_point() const;
	// members
	state_backup m_statebackup; // helper, cached variables to be restored on mode exit
	editormode_input m_input;
	static TCamera Camera;

	// focus (smooth camera fly-to) state
	static bool m_focus_active;
	glm::dvec3 m_focus_start_pos{0.0,0.0,0.0};
	glm::dvec3 m_focus_target_pos{0.0,0.0,0.0};
	glm::vec3 m_focus_start_angle{0.0f};   // camera pitch/yaw/roll at focus start
	glm::vec3 m_focus_target_angle{0.0f};  // camera pitch/yaw/roll facing the focused object
	double m_focus_time{0.0};
	double m_focus_duration{0.6};

	double fTime50Hz{0.0}; // bufor czasu dla komunikacji z PoKeys
	scene::basic_editor m_editor;
	scene::basic_node *m_node{nullptr}; // currently selected scene node
	// per-track editor labels (e.g. "KP 0->300", "Luk R=300"), shown by the track overlay
	std::unordered_map<scene::basic_node const *, std::string> m_track_labels;
	std::string m_pending_track_label; // label applied to the next track created by commit_track/commit_switch
	// niweleta state
	std::vector<track_chain> m_chains;
	int m_active_chain { -1 }; // chain receiving new elements in place mode
	std::unordered_map<TTrack *, switch_meta> m_switch_meta; // creation parameters of editor switches
	// drag state: either a chain joint or a whole switch
	bool m_dragactive { false };
	int m_chaindrag_chain { -1 };
	int m_chaindrag_joint { -1 };
	TTrack *m_switchdrag { nullptr };
	glm::dvec3 m_drag_from { 0.0 };
	glm::dvec3 m_dragpos { 0.0 };
	bool m_takesnapshot{true}; // helper, hints whether snapshot of selected node(s) should be taken before modification
	bool m_dragging = false;
	glm::dvec3 oldPos;
	bool mouseHold{false};
	float kMaxPlacementDistance = 200.0f;
	static bool m_change_history;
	static bool m_settings_open;

	// camera fly-mode (right mouse button held); used to flush motion when it's released
	command_relay m_camera_relay;
	bool m_camera_flying{false};

	// UI/history settings
	int m_max_history_size{200};
	int m_selected_history_idx{-1};
	glm::dvec3 clamp_mouse_offset_to_max(const glm::dvec3 &offset);

	// focus camera smoothly on specified node
	void start_focus(scene::basic_node *node, double duration = 0.6);

	// drops the node straight down onto the nearest surface below (terrain or another object)
	void snap_to_ground(scene::basic_node *node);

	// editable terrain patches created in the editor
	void render_terrain_ui();
	// creates a large terrain as a grid of adjacent chunks (each its own editable patch)
	void create_chunked_terrain();
	// manual grid-aligned chunks: add/remove single chunks for fine control
	float chunk_grid_size() const { return m_terrain_cells * m_terrain_cellsize; }
	void add_grid_chunk(int Cx, int Cz);
	void remove_grid_chunk(int Cx, int Cz);
	// handles a click in chunk-edit mode (add a neighbour, or Shift = delete the clicked chunk)
	void handle_chunk_edit_click(bool DeleteMode);
	// commits authored terrain to disk, enables streaming, and exports the scenery (Ctrl+S)
	void save_scene_with_terrain();
	// raises/lowers terrain under the cursor while the left mouse button is held in sculpt mode
	void handle_terrain_sculpt(double Deltatime);
	// returns the terrain patch (if any) whose footprint covers the given world point
	editor_terrain *terrain_at(double X, double Z);
	// gathers every active terrain patch: manually-created ones plus streamed chunks
	std::vector<editor_terrain *> active_terrains();
	// samples the selected model instance's geometry into a new editable terrain patch, then removes it
	void capture_terrain();
	std::vector<std::unique_ptr<editor_terrain>> m_terrains;
	// grid-aligned manual chunks, keyed by (cx,cz) on the global chunk grid
	std::map<std::pair<int, int>, std::unique_ptr<editor_terrain>> m_grid_chunks;
	bool m_terrain_sculpt{false};     // when true, LMB sculpts terrain instead of picking
	bool m_chunk_edit{false};         // when true, LMB adds/removes whole chunks
	int m_terrain_cells{32};          // grid resolution (quads per side)
	int m_terrain_chunks{4};          // chunks per side for a chunked terrain
	float m_terrain_cellsize{2.0f};   // metres per quad
	float m_terrain_baseheight{0.0f}; // flat starting height
	float m_terrain_brush_radius{12.0f};
	float m_terrain_brush_strength{4.0f}; // metres per second while held (one-shot for the buttons)
	float m_terrain_simplify_error{0.5f}; // flatness tolerance (m) for mesh simplification
	bool m_terrain_auto_optimize{false};  // auto-simplify edited chunks after sculpting settles
	double m_terrain_idle{0.0};           // seconds since the last sculpt edit (debounce timer)
	char m_terrain_texture[128]{""};  // optional ground texture name

	// streaming terrain that follows the camera (open-world); the editor shares the single
	// simulation-level instance so authored terrain also renders in the driver / other modes
	terrain_streamer &m_streamer{EditorTerrain};
	int m_stream_radius{2};
	bool m_stream_persist{true}; // save edited chunks to disk and load them back

	// hierarchy management
	void add_to_hierarchy(scene::basic_node *node);
	void remove_from_hierarchy(scene::basic_node *node);
	scene::basic_node* find_in_hierarchy(const std::string &uuid_str);
	scene::basic_node* find_node_by_any(scene::basic_node *node_ptr, const std::string &uuid_str, const std::string &name);

	// clear history/redo pointers that reference the given node (prevent dangling pointers)
	void nullify_history_pointers(scene::basic_node *node);
	void render_change_history();
	void render_settings();

	// ImGuizmo-based transform gizmo for the selected node
	enum class gizmo_operation { translate, rotate, scale };
	void render_gizmo();
	// draws the selected track's bezier control handles and a type/radius label as a 2D overlay
	void render_track_overlay();
	bool m_gizmo_enabled{true};                                  // master switch for the in-viewport gizmo
	bool m_gizmo_using{false};                                   // tracks an ongoing drag, so a single undo snapshot is taken per drag
	bool m_gizmo_local{false};                                   // manipulate in the object's local space instead of world space
	gizmo_operation m_gizmo_op{gizmo_operation::translate};      // current transform mode (translate/rotate/scale)
	float m_gizmo_snap{1.0f};                                    // translation snap step (metres) applied while Ctrl is held
};
