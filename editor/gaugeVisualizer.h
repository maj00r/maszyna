/*
This Source Code Form is subject to the
terms of the Mozilla Public License, v.
2.0. If a copy of the MPL was not
distributed with this file, You can
obtain one at
http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <vector>
#include <string>
#include <memory>
#include <glm/glm.hpp>

#include "rendering/geometrybank.h"
#include "utilities/Classes.h"
#include "interfaces/ITexture.h"

class TTrack;
class TSegment;
namespace scene { class basic_cell; }

struct gauge_point
{
	double half_width{0.0};
	double height{0.0};
};

struct gauge_placement
{
	std::shared_ptr<TSegment> segment;
	TTrack *track{nullptr};
	double segment_position_norm{0.0};
};

struct gauge_sample
{
	glm::dvec3 position{0.0};
	double radius{0.0};
};

struct gauge_frame
{
	glm::dvec3 base{0.0};
	glm::dvec3 tangent{0.0};
	glm::dvec3 right{0.0};
	glm::dvec3 up{0.0};
	glm::dvec3 right_flat{0.0};
};

class gauge_overlay
{
  public:
	gauge_overlay();

	void load_profile();
	std::vector<gauge_point> const &profile() const { return m_profile; }
	std::string const &name() const { return m_name; }

	void update(glm::dvec3 const &Cursor, bool Visible, bool Locked, glm::dmat4 const &ViewMatrix);
	void clear();

  private:
	void load_default_profile();
	bool parse_profile_file(std::string const &Path);

	static gauge_placement nearest_track(glm::dvec3 const &Point) ;
	gauge_frame frame_at(gauge_placement const &Placement) const;
	void curve_widening(gauge_placement const &Placement, glm::dvec3 const &RightFlat, double &WidenRight, double &WidenLeft,
	                    std::vector<gauge_sample> *Samples) const;
	std::vector<glm::dvec3> outline(gauge_frame const &Frame, double WidenRight, double WidenLeft) const;
	void render_fill(gauge_frame const &Frame, std::vector<glm::dvec3> const &Outline, glm::dmat4 const &ViewMatrix);
	void render_info(gauge_frame const &Frame, std::vector<gauge_sample> const &Samples, double WidenRight, double WidenLeft, glm::dmat4 const &ViewMatrix) const;

	static glm::dvec3 point_at(std::shared_ptr<TSegment> const &Segment, double Arc) ;
	void local_curvature(std::shared_ptr<TSegment> const &Segment, double Arc, double Baseline, double &Radius, glm::dvec3 &Turn) const;
	bool segment_curvature(std::shared_ptr<TSegment> const &Segment, double &Radius, glm::dvec3 &Turn, glm::dvec3 &Position,
	                       std::vector<gauge_sample> *Samples = nullptr) const;

	std::vector<gauge_point> m_profile;
	std::string m_name;
	gauge_placement m_locked;

	scene::basic_cell *m_cell{nullptr};
	gfx::geometrybank_handle m_bank{0, 0};
	gfx::geometry_handle m_fill{0, 0};
	std::size_t m_fill_count{0};
	material_handle m_material{null_handle};
};
