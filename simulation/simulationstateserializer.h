/*
This Source Code Form is subject to the
terms of the Mozilla Public License, v.
2.0. If a copy of the MPL was not
distributed with this file, You can
obtain one at
http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include "utilities/parser.h"
#include "scene/scene.h"

namespace simulation {

struct deserializer_state {
	std::string scenariofile;
	cParser input;
	scene::scratch_data scratchpad;
	using deserializefunctionbind = std::function<void()>;
	std::unordered_map<
	    std::string,
	    deserializefunctionbind> functionmap;

	deserializer_state(std::string const &File, cParser::buffertype const Type, const std::string &Path, bool const Loadtraction)
	    : scenariofile(File), input(File, Type, Path, Loadtraction) { }
};

class state_serializer {

public:

// methods
	// starts deserialization from specified file, returns context pointer on success, throws otherwise
	std::shared_ptr<deserializer_state>
	    deserialize_begin(std::string const &Scenariofile);
	// continues deserialization for given context, amount limited by time, returns true if needs to be called again
	bool
	    deserialize_continue(std::shared_ptr<deserializer_state> state);
    // stores class data in specified file, in legacy (text) format
    void
        export_as_text( std::string const &Scenariofile ) const;
	// create new model from node stirng
	TAnimModel * create_model(std::string const &src, std::string const &name, const glm::dvec3 &position);
	// create new eventlauncher from node stirng
	TEventLauncher * create_eventlauncher(std::string const &src, std::string const &name, const glm::dvec3 &position);

	// Faza 4c (model streaming): write each populated section's model instances to a per-section text
	// file (generate-if-missing) at load end, so they can be paged in/out around the camera.
	void bake_section_models();
	// page section model instances around the camera: destroy those in sections that leave Radius
	// (freeing their smoke sources / Hierarchy / instance-table entries) and recreate those that come
	// back from the section's model file. only ungrouped instances are paged (grouped stay resident).
	void stream_section_models( glm::dvec3 const &Camera, int Radius );

private:
	// Faza 4c: model-paging bookkeeping (section index -> baked/resident), mirrors the geometry pager.
	std::unordered_set<std::size_t> m_baked_model_sections;
	std::unordered_set<std::size_t> m_resident_model_sections;
	// per-model instantiation queue: (section index, node source string) from sections that paged in,
	// drained a few ms per frame so a fast-moving camera causes pop-in instead of multi-second stalls
	// (a section can hold hundreds of models; re-parsing them all in one frame froze the frame loop).
	// entries whose section left the radius before they surfaced are dropped at drain time.
	std::deque<std::pair<std::size_t, std::string>> m_pending_model_creates;
	// per-model destruction queue, drained under the same per-frame budget. destroying a large
	// scenery's out-of-range sections at once (~660k models on tomaszewo) stalled for minutes: the
	// teardown ran O(N) table/group scans per model. queued models are torn down in batches through
	// the bulk paths (basic_table::purge_batch, node_groups::detach_many). a section that re-enters
	// the radius while its models are still queued simply cancels those entries (models never died).
	std::deque<std::pair<std::size_t, TAnimModel *>> m_pending_model_destroys;
	std::unordered_map<std::size_t, std::uint32_t> m_destroy_backlog; // section -> entries still queued
	// tears down up to BudgetMs worth of queued models (everything when BudgetMs <= 0)
	void drain_model_destroys( double BudgetMs );
// methods
    // restores class data from provided stream
    void deserialize_area( cParser &Input, scene::scratch_data &Scratchpad );
    void deserialize_isolated( cParser &Input, scene::scratch_data &Scratchpad );
    void deserialize_assignment( cParser &Input, scene::scratch_data &Scratchpad );
    void deserialize_atmo( cParser &Input, scene::scratch_data &Scratchpad );
    void deserialize_camera( cParser &Input, scene::scratch_data &Scratchpad );
    void deserialize_config( cParser &Input, scene::scratch_data &Scratchpad );
    void deserialize_description( cParser &Input, scene::scratch_data &Scratchpad );
    void deserialize_event( cParser &Input, scene::scratch_data &Scratchpad );
    void deserialize_lua( cParser &Input, scene::scratch_data &Scratchpad );
    void deserialize_firstinit( cParser &Input, scene::scratch_data &Scratchpad );
    void deserialize_group( cParser &Input, scene::scratch_data &Scratchpad );
    void deserialize_endgroup( cParser &Input, scene::scratch_data &Scratchpad );
    void deserialize_light( cParser &Input, scene::scratch_data &Scratchpad );
	void deserialize_node( cParser &Input, scene::scratch_data &Scratchpad );
    void deserialize_origin( cParser &Input, scene::scratch_data &Scratchpad );
    void deserialize_endorigin( cParser &Input, scene::scratch_data &Scratchpad );
    void deserialize_scale( cParser &Input, scene::scratch_data &Scratchpad );
    void deserialize_endscale( cParser &Input, scene::scratch_data &Scratchpad );
    void deserialize_rotate( cParser &Input, scene::scratch_data &Scratchpad );
    void deserialize_sky( cParser &Input, scene::scratch_data &Scratchpad );
    void deserialize_test( cParser &Input, scene::scratch_data &Scratchpad );
    void deserialize_time( cParser &Input, scene::scratch_data &Scratchpad );
    void deserialize_trainset( cParser &Input, scene::scratch_data &Scratchpad );
    void deserialize_terrain( cParser &Input, scene::scratch_data &Scratchpad );
    void deserialize_editorterrain( cParser &Input, scene::scratch_data &Scratchpad );
    void deserialize_endtrainset( cParser &Input, scene::scratch_data &Scratchpad );
    TTrack * deserialize_path( cParser &Input, scene::scratch_data &Scratchpad, scene::node_data const &Nodedata );
    TTraction * deserialize_traction( cParser &Input, scene::scratch_data &Scratchpad, scene::node_data const &Nodedata );
    TTractionPowerSource * deserialize_tractionpowersource( cParser &Input, scene::scratch_data &Scratchpad, scene::node_data const &Nodedata );
    TMemCell * deserialize_memorycell( cParser &Input, scene::scratch_data &Scratchpad, scene::node_data const &Nodedata );
    TEventLauncher * deserialize_eventlauncher( cParser &Input, scene::scratch_data &Scratchpad, scene::node_data const &Nodedata );
	TAnimModel * deserialize_model( cParser &Input, scene::scratch_data &Scratchpad, scene::node_data const &Nodedata );
    TDynamicObject * deserialize_dynamic( cParser &Input, scene::scratch_data &Scratchpad, scene::node_data const &Nodedata );
    sound_source * deserialize_sound( cParser &Input, scene::scratch_data &Scratchpad, scene::node_data const &Nodedata );
    void init_time();
    // skips content of stream until specified token
    void skip_until( cParser &Input, std::string const &Token );
    // transforms provided location by specifed rotation and offset
    glm::dvec3 transform( glm::dvec3 Location, scene::scratch_data const &Scratchpad );
    void export_nodes_to_stream( std::ostream &, bool Dirty ) const;
};

} // simulation

//---------------------------------------------------------------------------
