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
#include <memory>
#include <unordered_map>

#include <glm/glm.hpp>

class editor_mode;
class editor_ui;
class TTrack;
class TSegment;
namespace scene { class basic_node; }

// Niweleta (track alignment) editor. A niweleta is an ordered chain of typed elements
// (straight / arc / transition curve) marched from an origin; the tracks are the generated
// result layer. Switches are branch points that anchor new chains. This subsystem was extracted
// out of editor_mode, which now owns a track_editor and drives it from its input handlers.
//
// The state is public: editor_mode's mouse/key/update dispatch reads and mutates it directly
// (the input dispatch stays in editor_mode because it is interleaved with the gizmo / terrain /
// brush tools). The logic methods live here. A back-pointer to the owner provides the panel
// parameters (ui()), the selected-node slot (m_node) and the editor camera.
class track_editor
{
  public:
	explicit track_editor(editor_mode *owner) : m_owner(owner) {}

	// one element of a niweleta chain
	struct chain_element
	{
		int type{0};           // track_panel::track_type
		double length{50.0};   // [m] element arc length
		double radius{300.0};  // ARC radius / TRANSITION end radius (0 = straight)
		double radius0{0.0};   // TRANSITION start radius (0 = straight)
		bool left{true};
		int cuts{0};                  // straight only: mid-element cuts (emits cuts+1 track pieces)
		std::vector<TTrack *> tracks; // generated result (straights may emit several pieces)
	};
	struct track_chain
	{
		int id{0};                             // stable id (persisted; drives generated track names)
		glm::dvec3 origin{0.0};                // start point (node base level)
		glm::dvec3 direction{0.0, 0.0, -1.0};  // unit start tangent
		TTrack *anchor{nullptr};               // optional switch feeding this chain
		int anchor_end{-1};                    // switch endpoint index (1 = main end, 3 = diverging end)
		std::vector<chain_element> elements;
		std::vector<glm::dvec3> joints; // cached: origin + each element end (railhead-free base points)
		// superelevation [mm] at each joint (signed; same size as joints). elements interpolate
		// between their two joint values, so cant is continuous across junctions by construction
		std::vector<double> joint_cant;
		glm::dvec3 endtangent{0.0, 0.0, -1.0}; // cached march exit tangent (for branching / extending)
	};
	struct switch_meta
	{
		glm::dvec3 entry, straightend, divcv1, divcv2, divend;
		double radius, length;
	};

	// --- track construction ---
	// track tool: create a track (type/params from the track panel) starting at a point, along a direction
	void create_track_at(glm::dvec3 const &start, glm::dvec3 const &dir);
	// snaps start point+direction to the nearest existing track endpoint within reach (for connecting tracks)
	bool snap_track_start(glm::dvec3 &start, glm::dvec3 &dir);
	// builds a track node from bezier control offsets and commits it (create_track + select).
	// roll1/roll2 are the superelevation angles [deg] at p1/p2 (written into the node's roll fields)
	TTrack *commit_track(glm::dvec3 const &p1, glm::dvec3 const &cv1, glm::dvec3 const &cv2, glm::dvec3 const &p2, double radius, double length, double roll1 = 0.0, double roll2 = 0.0);
	// builds a switch node (straight main path + diverging arc path sharing the entry point)
	TTrack *commit_switch(glm::dvec3 const &entry, glm::dvec3 const &straightend, glm::dvec3 const &divcv1, glm::dvec3 const &divcv2, glm::dvec3 const &divend, double radius, double length);
	// removes a track from the scene (network detach + mesh hide + path table + labels)
	void delete_track(TTrack *track);
	// lays a transition curve (clothoid) as one bezier; curvature goes linearly from kappa_start to kappa_end
	void create_transition(glm::dvec3 const &start, glm::dvec3 const &tangent, double length, double kappa_start, double kappa_end, bool left);
	// convenience: track ahead of the editor camera (hotkey)
	void create_straight_track_ahead(double length = 50.0);

	// --- niweleta chain model ---
	// marches one element from (pos, dir), optionally emitting the result track; namebase makes the
	// generated names deterministic (reload re-link); cant_start/cant_end [mm] bank the emitted track
	TTrack *march_element(chain_element &el, glm::dvec3 &pos, glm::dvec3 &dir, bool emit, std::string const &namebase = "", double cant_start = 0.0, double cant_end = 0.0);
	// regenerates a chain: re-marches all elements from the (possibly anchored) origin
	void regenerate_chain(int index);
	// sets the cant [mm] of the element owning the given track and regenerates; false if not in a chain
	bool set_track_cant(TTrack *track, double cant_mm);
	// applies the panel's geometry fields to the element owning the given track; false if not in a chain
	bool set_track_geometry(TTrack *track);

	// --- editing (drag / fit) ---
	// finds a track endpoint close to the given screen position
	bool pick_track_endpoint(float screenx, float screeny, bool allowswitch, TTrack *&track, int &endindex, glm::dvec3 &point, glm::dvec3 &outward);
	// finds a draggable chain joint (free origin or the far end of a straight) near a screen position
	bool pick_chain_joint(float screenx, float screeny, int &chain, int &joint);
	// fits a curve group [gfirst..glast] between two fixed straight lines, keeping KP length/radii and
	// the arc radius (only the arc sweep varies); outputs the group's placement. false if it can't fit.
	bool fit_group_between_lines(int chain, int gfirst, int glast, glm::dvec3 const &P1, glm::dvec3 const &d1, glm::dvec3 const &P2, glm::dvec3 const &d2, glm::dvec3 &groupstart, glm::dvec3 &groupend);
	// drags a STRAIGHT element: rotates about its fixed start; side curve groups re-fit against the
	// unchanged neighbour lines (neighbours only change length). false if the fit is infeasible.
	bool fit_straight_drag(int chain, int element, glm::dvec3 const &target);
	// applies the active chain-joint drag toward target; preview only re-marches joints (live), else
	// regenerates the tracks. false if the fit was infeasible.
	bool apply_chain_drag(glm::dvec3 const &target, bool preview);

	// --- persistence / diagnostics ---
	std::string alignment_filepath() const; // <scenery>.niw sidecar path
	void save_alignments();                 // writes the chain source to the sidecar
	void load_alignments();                 // reads the sidecar and re-links to loaded tracks by name
	void dump_alignment_diag();             // full diagnostic dump of every chain to the log

	// --- view ---
	// world point where the cursor ray meets the ground plane (y=0); deterministic
	glm::dvec3 cursor_ground_point() const;
	// draws track endpoints, control vectors, labels, chain joints and diagnostics over the scene
	void render_track_overlay();

	// collects the drawable segments of a track (both nitki of a switch); shared with editor_mode's
	// track-tool input handlers
	static void collect_track_segments(TTrack *track, std::vector<std::shared_ptr<TSegment>> &out);

	// --- state (public: driven by editor_mode's input dispatch) ---
	editor_mode *m_owner{nullptr};
	// per-track editor labels (e.g. "KP 0->300", "Luk R=300"), shown by the overlay
	std::unordered_map<scene::basic_node const *, std::string> m_track_labels;
	std::string m_pending_track_label; // label applied to the next track created by commit_track/commit_switch
	std::string m_pending_track_name;  // overrides the auto name in commit_track/commit_switch (stable niweleta names)
	std::vector<track_chain> m_chains;
	int m_active_chain{-1};  // chain receiving new elements in place mode
	int m_next_chain_id{0};  // running id assigned to new chains (persisted via the sidecar)
	std::unordered_map<TTrack *, switch_meta> m_switch_meta; // creation parameters of editor switches
	// drag state: either a chain joint or a whole switch
	bool m_dragactive{false};
	int m_chaindrag_chain{-1};
	int m_chaindrag_joint{-1};
	TTrack *m_switchdrag{nullptr};
	glm::dvec3 m_drag_from{0.0};
	glm::dvec3 m_dragpos{0.0};
	glm::dvec3 m_dragapplied{0.0};  // cursor position at which the live fit was last applied
	glm::dvec3 m_draglastgood{0.0}; // last cursor position where the fit was feasible (drag holds here)
	bool m_dragfeasible{true};      // false while the cursor is in an infeasible region (shows a warning)
	// snapshot of the dragged chain's element params (taken at drag start) so a live drag always fits
	// relative to the original geometry, not the already-moved result. .tracks left empty.
	std::vector<chain_element> m_dragsnap;
	glm::dvec3 m_dragsnap_origin{0.0};
	glm::dvec3 m_dragsnap_dir{0.0, 0.0, -1.0};
};
