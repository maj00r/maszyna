/*
This Source Code Form is subject to the
terms of the Mozilla Public License, v.
2.0. If a copy of the MPL was not
distributed with this file, You can
obtain one at
http://mozilla.org/MPL/2.0/.
*/

#include "gaugeVisualizer.h"

#include "world/Segment.h"
#include "world/Track.h"
#include "simulation/simulation.h"
#include "scene/scene.h"
#include "rendering/renderer.h"
#include "utilities/Globals.h"
#include "imgui/imgui.h"

#include <glm/gtc/matrix_transform.hpp>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

namespace
{
constexpr double gauge_pick_radius = 60.0;
constexpr double gauge_curve_range = 26.0;
constexpr double gauge_track_cull = 500.0;
auto const gauge_fill_name = "editor:loading_gauge_fill";
}

gauge_overlay::gauge_overlay()
{
	load_profile();
}

void gauge_overlay::load_profile()
{
	m_profile.clear();
	m_name = "skrajnia.txt";

	if (parse_profile_file(Global.asCurrentSceneryPath + m_name))
	{
		return;
	}

	load_default_profile();
}

bool gauge_overlay::parse_profile_file(std::string const &Path)
{
	std::ifstream in(Path);
	if (!in)
		return false;

	std::string line;
	while (std::getline(in, line))
	{
		auto const start = line.find_first_not_of(" \t");
		if (start == std::string::npos)
			continue;
		if (line[start] == '/' || line[start] == '#')
			continue;

		std::istringstream ls(line);

		if (line.compare(start, 4, "name") == 0)
		{
			std::string tok, rest;
			ls >> tok;
			std::getline(ls, rest);
			auto const s = rest.find_first_not_of(" \t");
			if (s != std::string::npos)
				m_name = rest.substr(s);
			continue;
		}

		double hw, h;
		if (ls >> hw >> h)
			m_profile.push_back({hw, h});
	}

	return m_profile.size() >= 2;
}

void gauge_overlay::load_default_profile()
{
	m_name = "GPL-1";
	m_profile = {{1.585, 0.00},
		{1.675, 0.38},
		{2.000, 1.17},
		{2.000, 3.05},
		{1.900, 3.85},
		{1.800, 4.25},
		{1.600, 4.50},
		{1.450, 4.632},
		{1.260, 4.80},
		{1.150, 4.85},
		{1.150, 6.90}};
}

gauge_placement gauge_overlay::nearest_track(glm::dvec3 const &Point)
{
	gauge_placement result;
	double best = gauge_pick_radius * gauge_pick_radius;

	auto const hdist2 = [](glm::dvec3 const &a, glm::dvec3 const &b)
	{
		double const dx = a.x - b.x, dz = a.z - b.z;
		return dx * dx + dz * dz;
	};

	for (auto *track : simulation::Paths.sequence())
	{
		if (track == nullptr)
			continue;
		if (hdist2(track->location(), Point) > gauge_track_cull * gauge_track_cull)
			continue;

		std::shared_ptr<TSegment> segments[2];
		if (track->SwitchExtension)
		{
			segments[0] = track->SwitchExtension->Segments[0];
			segments[1] = track->SwitchExtension->Segments[1];
		}
		else
			segments[0] = track->CurrentSegment();

		for (auto const &segment : segments)
		{
			if (!segment)
				continue;
			double const t = segment->find_nearest_point(Point);
			double const d2 = hdist2(segment->FastGetPoint(t), Point);
			if (d2 < best)
			{
				best = d2;
				result.track = track;
				result.segment = segment;
				result.segment_position_norm = t;
			}
		}
	}

	return result;
}

glm::dvec3 gauge_overlay::point_at(std::shared_ptr<TSegment> const &Segment, double Arc)
{
	double const length = Segment->GetLength();
	double const t = (length > 1e-9) ? std::clamp(Arc / length, 0.0, 1.0) : 0.0;
	return glm::dvec3(Segment->FastGetPoint(t));
}

void gauge_overlay::local_curvature(std::shared_ptr<TSegment> const &Segment, double Arc, double Baseline, double &Radius, glm::dvec3 &Turn) const
{
	double const length = Segment->GetLength();
	glm::dvec3 back = point_at(Segment, Arc) - point_at(Segment, std::max(0.0, Arc - Baseline));
	glm::dvec3 fwd = point_at(Segment, std::min(length, Arc + Baseline)) - point_at(Segment, Arc);
	back.y = 0.0;
	fwd.y = 0.0;

	Radius = 1.0e9;
	Turn = glm::dvec3(0.0);

	double const lb = glm::length(back), lf = glm::length(fwd);
	if (lb < 1e-6 || lf < 1e-6)
		return;

	back /= lb;
	fwd /= lf;

	double const dtheta = std::acos(std::clamp(glm::dot(back, fwd), -1.0, 1.0));
	double const ds = 0.5 * (lb + lf);
	Radius = (dtheta > 1e-5) ? ds / dtheta : 1.0e9;
	Turn = fwd - back;
}

bool gauge_overlay::segment_curvature(std::shared_ptr<TSegment> const &Segment, double &Radius, glm::dvec3 &Turn, glm::dvec3 &Position, std::vector<gauge_sample> *Samples) const
{
	Radius = 1.0e9;
	Turn = glm::dvec3(0.0);
	Position = glm::dvec3(0.0);

	if (!Segment || !Segment->bCurve)
		return false;

	double const length = Segment->GetLength();
	if (length < 1e-6)
		return false;

	double const step = 1.0;
	bool found = false;

	for (double arc = step; arc <= length - step + 1e-6; arc += 2.0)
	{
		double r;
		glm::dvec3 turn;
		local_curvature(Segment, arc, step, r, turn);
		if (glm::length(turn) < 1e-12)
			continue;

		glm::dvec3 const here = point_at(Segment, arc);
		if (Samples)
			Samples->push_back({here, r});

		if (r < Radius)
		{
			Radius = r;
			Turn = turn;
			Position = here;
			found = true;
		}
	}

	if (!found)
	{
		local_curvature(Segment, length * 0.5, std::min(step, length * 0.49), Radius, Turn);
		Position = point_at(Segment, length * 0.5);
		found = true;
	}

	return found;
}

gauge_frame gauge_overlay::frame_at(gauge_placement const &Placement) const
{
	gauge_frame frame;
	if (!Placement.segment)
		return frame;

	auto const &segment = Placement.segment;
	double const t = Placement.segment_position_norm;

	frame.base = segment->FastGetPoint(t);

	glm::dvec3 const behind = segment->FastGetPoint(std::clamp(t - 0.01, 0.0, 1.0));
	glm::dvec3 const ahead = segment->FastGetPoint(std::clamp(t + 0.01, 0.0, 1.0));
	frame.tangent = ahead - behind;
	frame.tangent.y = 0.0;
	double const tl = glm::length(frame.tangent);
	frame.tangent = (tl > 1e-9) ? frame.tangent / tl : glm::dvec3(0.0, 0.0, 1.0);

	glm::dvec3 const worldup(0.0, 1.0, 0.0);
	frame.right_flat = glm::normalize(glm::cross(worldup, frame.tangent));

	double const roll = segment->GetRoll(t * segment->GetLength());
	double const c = std::cos(roll), s = std::sin(roll);
	frame.right = frame.right_flat * c + worldup * s;
	frame.up = worldup * c - frame.right_flat * s;

	return frame;
}

void gauge_overlay::curve_widening(gauge_placement const &Placement, glm::dvec3 const &RightFlat, double &WidenRight, double &WidenLeft, std::vector<gauge_sample> *Samples) const
{
	WidenRight = 0.0;
	WidenLeft = 0.0;
	if (Samples)
		Samples->clear();
	if (!Placement.segment)
		return;

	auto const scan = [&](int Startdir, double &OutDist, double &OutRadius, glm::dvec3 &OutTurn) -> bool
	{
		struct walkstate
		{
			TTrack *track;
			std::shared_ptr<TSegment> segment;
			double arc;
			int dir;
			double dist;
		};

		std::vector<walkstate> stack;
		double const seglen = Placement.segment->GetLength();
		stack.push_back({Placement.track, Placement.segment, std::clamp(Placement.segment_position_norm, 0.0, 1.0) * seglen, (Startdir >= 0) ? 1 : -1, 0.0});

		double minradius = 1.0e9, mindist = 0.0;
		glm::dvec3 minturn(0.0);
		bool found = false;

		for (int iter = 0; iter < 400 && !stack.empty(); ++iter)
		{
			walkstate const st = stack.back();
			stack.pop_back();
			if (!st.segment)
				continue;

			double radius;
			glm::dvec3 turn, pos;
			segment_curvature(st.segment, radius, turn, pos, Samples);
			if (radius < 2000.0)
			{
				if (!found || radius < minradius - 1e-6 || (std::abs(radius - minradius) < 1e-6 && st.dist < mindist))
				{
					minradius = radius;
					mindist = st.dist;
					minturn = turn;
					found = true;
				}
			}

			double const length = st.segment->GetLength();
			double const avail = (st.dir > 0) ? (length - st.arc) : st.arc;
			double const enddist = st.dist + avail;
			if (enddist > gauge_curve_range)
				continue;

			double const endarc = (st.dir > 0) ? length : 0.0;
			glm::dvec3 const endpos = point_at(st.segment, endarc);
			glm::dvec3 heading = point_at(st.segment, endarc + st.dir * 0.1) - point_at(st.segment, endarc - st.dir * 0.1);
			heading.y = 0.0;
			double const hl = glm::length(heading);
			if (hl > 1e-9)
				heading /= hl;

			TTrack *neighbours[4];
			int nneigh = 0;
			if (st.track->SwitchExtension)
			{
				TTrack *const legs[4] = {st.track->SwitchExtension->pNexts[0], st.track->SwitchExtension->pNexts[1], st.track->SwitchExtension->pPrevs[0], st.track->SwitchExtension->pPrevs[1]};
				for (auto *leg : legs)
					if (leg && leg != st.track)
						neighbours[nneigh++] = leg;
			}
			else
			{
				if (st.track->trNext)
					neighbours[nneigh++] = st.track->trNext;
				if (st.track->trPrev)
					neighbours[nneigh++] = st.track->trPrev;
			}

			for (int n = 0; n < nneigh; ++n)
			{
				TTrack *next = neighbours[n];
				std::shared_ptr<TSegment> candidates[2];
				int ncand = 0;
				if (next->SwitchExtension)
				{
					if (next->SwitchExtension->Segments[0])
						candidates[ncand++] = next->SwitchExtension->Segments[0];
					if (next->SwitchExtension->Segments[1])
						candidates[ncand++] = next->SwitchExtension->Segments[1];
				}
				else
					candidates[ncand++] = next->CurrentSegment();

				for (int i = 0; i < ncand; ++i)
				{
					auto const &segment = candidates[i];
					if (!segment || segment == st.segment)
						continue;

					double const length = segment->GetLength();
					glm::dvec3 const p0 = point_at(segment, 0.0);
					glm::dvec3 const p1 = point_at(segment, length);
					double const d0 = glm::length(p0 - endpos);
					double const d1 = glm::length(p1 - endpos);
					if (std::min(d0, d1) > 2.0)
						continue;

					bool const entry_at_start = (d0 <= d1);
					glm::dvec3 intan = entry_at_start ? (point_at(segment, 0.1) - p0) : (point_at(segment, length - 0.1) - p1);
					intan.y = 0.0;
					double const tl = glm::length(intan);
					if (tl > 1e-9 && glm::dot(heading, intan / tl) < 0.2)
						continue;

					stack.push_back({next, segment, entry_at_start ? 0.0 : length, entry_at_start ? 1 : -1, enddist});
				}
			}
		}

		if (!found)
			return false;
		OutDist = mindist;
		OutRadius = minradius;
		OutTurn = minturn;
		return true;
	};

	for (int dir = -1; dir <= 1; dir += 2)
	{
		double dist, radius;
		glm::dvec3 turn;
		if (!scan(dir, dist, radius, turn))
			continue;

		double widen_outer, widen_inner;
		if (radius > 250.0)
			widen_outer = widen_inner = (3750.0 / radius) / 1000.0;
		else
		{
			widen_outer = std::max(0.0, 60000.0 / radius - 225.0) / 1000.0;
			widen_inner = std::max(0.0, 50000.0 / radius - 185.0) / 1000.0;
		}

		double const fade_outer = std::clamp((gauge_curve_range - dist) / 20.0, 0.0, 1.0);
		double const fade_inner = std::clamp((20.0 - dist) / 20.0, 0.0, 1.0);
		bool const inner_is_right = glm::dot(turn, RightFlat) > 0.0;

		WidenRight = std::max(WidenRight, inner_is_right ? widen_inner * fade_inner : widen_outer * fade_outer);
		WidenLeft = std::max(WidenLeft, inner_is_right ? widen_outer * fade_outer : widen_inner * fade_inner);
	}
}

std::vector<glm::dvec3> gauge_overlay::outline(gauge_frame const &Frame, double WidenRight, double WidenLeft) const
{
	std::vector<glm::dvec3> points;
	points.reserve(m_profile.size() * 2);

	for (auto const &pt : m_profile)
	{
		double const hw = pt.half_width + (pt.half_width > 0.05 ? WidenRight : 0.0);
		points.push_back(Frame.base + Frame.right * hw + Frame.up * pt.height);
	}
	for (auto it = m_profile.rbegin(); it != m_profile.rend(); ++it)
	{
		if (it->half_width < 1e-6)
			continue;
		double const hw = it->half_width + (it->half_width > 0.05 ? WidenLeft : 0.0);
		points.push_back(Frame.base - Frame.right * hw + Frame.up * it->height);
	}

	return points;
}

namespace
{
ImU32 constexpr color_dimension = IM_COL32(255, 230, 120, 255);
ImU32 constexpr color_vertex = IM_COL32(210, 240, 255, 255);
ImU32 constexpr color_radius = IM_COL32(255, 255, 255, 255);

std::string meters(double Value)
{
	char buffer[32];
	std::snprintf(buffer, sizeof(buffer), "%.2f", Value);
	return buffer;
}

std::string whole(double Value)
{
	return std::to_string(static_cast<int>(std::lround(Value)));
}

struct gauge_extent
{
	double half_width;
	double height;
};

gauge_extent max_extent(std::vector<gauge_point> const &Profile)
{
	gauge_extent extent{0.0, 0.0};
	for (auto const &point : Profile)
	{
		extent.half_width = std::max(extent.half_width, point.half_width);
		extent.height = std::max(extent.height, point.height);
	}
	return extent;
}

double half_width_with_widening(gauge_point const &Point, double Widening)
{
	bool const is_edge = Point.half_width > 0.05;
	return Point.half_width + (is_edge ? Widening : 0.0);
}

double sharpest_radius(std::vector<gauge_sample> const &Samples)
{
	double sharpest = -1.0;
	for (auto const &sample : Samples)
		if (sharpest < 0.0 || sample.radius < sharpest)
			sharpest = sample.radius;
	return sharpest;
}

struct screen_projector
{
	glm::dmat4 world_to_view;
	glm::mat4 projection;
	ImVec2 viewport;

	bool project(glm::dvec3 const &World, ImVec2 &Screen) const
	{
		glm::dvec4 const eye = world_to_view * glm::dvec4(World, 1.0);
		glm::vec4 const clip = projection * glm::vec4(eye);
		if (clip.w <= 0.001f)
			return false;
		Screen.x = (clip.x / clip.w * 0.5f + 0.5f) * viewport.x;
		Screen.y = (1.0f - (clip.y / clip.w * 0.5f + 0.5f)) * viewport.y;
		return true;
	}
};

screen_projector make_projector(glm::dmat4 const &ViewMatrix)
{
	ImGuiIO const &io = ImGui::GetIO();
	float const field_of_view = glm::radians(Global.FieldOfView / Global.ZoomFactor);
	float const aspect = io.DisplaySize.y > 0.0f ? io.DisplaySize.x / io.DisplaySize.y : 1.0f;
	glm::mat4 const projection = glm::perspective(field_of_view, aspect, 0.1f, 10000.0f);
	return {ViewMatrix, projection, io.DisplaySize};
}

void draw_label(ImDrawList *DrawList, ImVec2 const &Position, ImU32 Color, std::string const &Text)
{
	ImVec2 const size = ImGui::CalcTextSize(Text.c_str());
	DrawList->AddRectFilled(ImVec2(Position.x - 2.0f, Position.y - 1.0f), ImVec2(Position.x + size.x + 2.0f, Position.y + size.y + 1.0f), IM_COL32(0, 0, 0, 170), 2.0f);
	DrawList->AddText(Position, Color, Text.c_str());
}

std::vector<world_vertex> build_two_sided_fan(gauge_frame const &Frame, std::vector<glm::dvec3> const &Outline)
{
	glm::dvec3 centroid(0.0);
	for (auto const &point : Outline)
		centroid += point;
	centroid /= static_cast<double>(Outline.size());
	glm::vec3 const facing(Frame.tangent);

	std::vector<world_vertex> mesh;
	mesh.reserve(Outline.size() * 6);
	auto add = [&](glm::dvec3 const &Position, glm::vec3 const &Normal)
	{
		world_vertex vertex;
		vertex.position = Position;
		vertex.normal = Normal;
		vertex.texture = glm::vec2(0.0f);
		mesh.push_back(vertex);
	};
	for (std::size_t i = 0; i < Outline.size(); ++i)
	{
		glm::dvec3 const &a = Outline[i];
		glm::dvec3 const &b = Outline[(i + 1) % Outline.size()];
		add(centroid, facing);
		add(a, facing);
		add(b, facing);
		add(centroid, -facing);
		add(b, -facing);
		add(a, -facing);
	}
	return mesh;
}

void draw_dimension(ImDrawList *DrawList, screen_projector const &Projector, glm::dvec3 const &From, glm::dvec3 const &To, glm::dvec3 const &TickOffset, std::string const &Label)
{
	ImVec2 from, to, tick;
	if (!Projector.project(From, from) || !Projector.project(To, to))
		return;
	DrawList->AddLine(from, to, color_dimension, 1.5f);
	if (Projector.project(From + TickOffset, tick))
		DrawList->AddLine(from, tick, color_dimension, 1.5f);
	if (Projector.project(To + TickOffset, tick))
		DrawList->AddLine(to, tick, color_dimension, 1.5f);
	draw_label(DrawList, ImVec2((from.x + to.x) * 0.5f + 4.0f, (from.y + to.y) * 0.5f - 6.0f), color_dimension, Label);
}

void draw_curve_radius(ImDrawList *DrawList, screen_projector const &Projector, std::vector<gauge_sample> const &Samples)
{
	for (auto const &sample : Samples)
	{
		ImVec2 point;
		if (!Projector.project(sample.position, point))
			continue;
		DrawList->AddCircleFilled(point, 3.0f, color_radius);
		draw_label(DrawList, ImVec2(point.x + 5.0f, point.y - 6.0f), color_radius, "R=" + whole(sample.radius));
	}
}

void draw_extent_dimensions(ImDrawList *DrawList, screen_projector const &Projector, gauge_frame const &Frame, std::vector<gauge_point> const &Profile, double WidenRight, double WidenLeft)
{
	gauge_extent const extent = max_extent(Profile);
	double const right_edge = extent.half_width + WidenRight;
	double const left_edge = extent.half_width + WidenLeft;

	glm::dvec3 const width_row = Frame.base - Frame.up * 0.5;
	draw_dimension(DrawList, Projector, width_row - Frame.right * left_edge, width_row + Frame.right * right_edge, -Frame.up * 0.25, "szer " + meters(left_edge + right_edge) + " m");

	glm::dvec3 const height_column = Frame.right * (right_edge + 0.6);
	draw_dimension(DrawList, Projector, Frame.base + height_column, Frame.base + height_column + Frame.up * extent.height, -Frame.right * 0.25, "wys " + meters(extent.height) + " m");
}

void draw_vertex_dimensions(ImDrawList *DrawList, screen_projector const &Projector, gauge_frame const &Frame, std::vector<gauge_point> const &Profile, double WidenRight, double WidenLeft)
{
	for (auto const &point : Profile)
	{
		double const right = half_width_with_widening(point, WidenRight);
		double const left = half_width_with_widening(point, WidenLeft);
		std::string const height = meters(point.height);
		ImVec2 screen;
		if (Projector.project(Frame.base + Frame.right * right + Frame.up * point.height, screen))
		{
			DrawList->AddCircleFilled(screen, 2.5f, color_vertex);
			draw_label(DrawList, ImVec2(screen.x + 4.0f, screen.y - 6.0f), color_vertex, meters(right) + " / " + height);
		}
		if (point.half_width > 1e-6 && Projector.project(Frame.base - Frame.right * left + Frame.up * point.height, screen))
		{
			DrawList->AddCircleFilled(screen, 2.5f, color_vertex);
			draw_label(DrawList, ImVec2(screen.x - 66.0f, screen.y - 6.0f), color_vertex, meters(left) + " / " + height);
		}
	}
}

void draw_summary(ImDrawList *DrawList, screen_projector const &Projector, gauge_frame const &Frame, std::vector<gauge_point> const &Profile, std::string const &Name, std::vector<gauge_sample> const &Samples, double WidenRight, double WidenLeft)
{
	gauge_extent const extent = max_extent(Profile);
	double const full_width = 2.0 * extent.half_width + WidenLeft + WidenRight;

	ImVec2 point;
	if (Projector.project(Frame.base + Frame.up * (extent.height + 0.7), point))
		draw_label(DrawList, ImVec2(point.x + 4.0f, point.y), color_dimension,
		           Name + "  " + std::to_string(Profile.size()) + " pkt  " + meters(full_width) + " x " + meters(extent.height) + " m  (wierzch: polszer/wys)");

	double const radius = sharpest_radius(Samples);
	if (Projector.project(Frame.base + glm::dvec3(0.0, 5.0, 0.0), point))
	{
		std::string const arc = radius > 0.0 ? "R=" + whole(radius) + "m" : "prosto";
		draw_label(DrawList, ImVec2(point.x + 6.0f, point.y), color_dimension, "luk: " + arc + "  posz. P=" + whole(WidenRight * 1000.0) + "mm L=" + whole(WidenLeft * 1000.0) + "mm");
	}
}
}

void gauge_overlay::update(glm::dvec3 const &Cursor, bool Visible, bool Locked, glm::dmat4 const &ViewMatrix)
{
	if (!Visible)
	{
		clear();
		return;
	}

	gauge_placement placement;
	if (Locked && m_locked.segment)
		placement = m_locked;
	else
	{
		placement = nearest_track(Cursor);
		if (placement.segment)
			m_locked = placement;
	}

	if (!placement.segment)
	{
		clear();
		return;
	}

	gauge_frame const frame = frame_at(placement);
	double widen_right, widen_left;
	std::vector<gauge_sample> samples;
	curve_widening(placement, frame.right_flat, widen_right, widen_left, &samples);

	std::vector<glm::dvec3> const shape = outline(frame, widen_right, widen_left);
	if (shape.size() < 3)
	{
		clear();
		return;
	}

	render_info(frame, samples, widen_right, widen_left, ViewMatrix);
	render_fill(frame, shape, ViewMatrix);
}

void gauge_overlay::render_fill(gauge_frame const &Frame, std::vector<glm::dvec3> const &Outline, glm::dmat4 const &ViewMatrix)
{
	(void)ViewMatrix;

	std::vector<world_vertex> mesh = build_two_sided_fan(Frame, Outline);

	scene::basic_cell &cell = simulation::Region->section(Frame.base).cell(Frame.base);
	glm::dvec3 const origin = cell.area().center;

	if (m_bank.bank == 0 && m_bank.chunk == 0)
		m_bank = GfxRenderer->Create_Bank();
	if (m_material == null_handle)
		m_material = GfxRenderer->Fetch_Material("white");

	std::size_t const vertex_count = mesh.size();
	gfx::vertex_array vertices;
	vertices.reserve(vertex_count);
	for (auto const &vertex : mesh)
		vertices.emplace_back(gfx::basic_vertex::convert(vertex, origin));
	gfx::userdata_array no_userdata;

	bool const already_uploaded = (m_fill.bank != 0 || m_fill.chunk != 0);
	if (already_uploaded && vertex_count == m_fill_count)
	{
		GfxRenderer->Replace(vertices, no_userdata, m_fill, GL_TRIANGLES);
	}
	else
	{
		m_fill = GfxRenderer->Insert(vertices, no_userdata, m_bank, GL_TRIANGLES);
		m_fill_count = vertex_count;
	}

	bool const crossed_into_new_cell = m_cell != &cell;
	if (crossed_into_new_cell)
	{
		clear();
		scene::shape_node fill;
		fill.make_overlay(std::move(mesh), origin, m_material, glm::vec4(0.25f, 0.6f, 1.0f, 0.35f), true, gauge_fill_name);
		fill.geometry(m_fill);
		cell.push_shape_direct(std::move(fill));
		m_cell = &cell;
		GfxRenderer->Regather_Scene();
	}
}

void gauge_overlay::clear()
{
	if (m_cell == nullptr)
		return;
	m_cell->erase_shapes(gauge_fill_name);
	m_cell = nullptr;
	GfxRenderer->Regather_Scene();
}

void gauge_overlay::render_info(gauge_frame const &Frame, std::vector<gauge_sample> const &Samples, double WidenRight, double WidenLeft, glm::dmat4 const &ViewMatrix) const
{
	screen_projector const projector = make_projector(ViewMatrix);
	ImDrawList *draw_list = ImGui::GetForegroundDrawList();

	draw_curve_radius(draw_list, projector, Samples);
	draw_extent_dimensions(draw_list, projector, Frame, m_profile, WidenRight, WidenLeft);
	draw_vertex_dimensions(draw_list, projector, Frame, m_profile, WidenRight, WidenLeft);
	draw_summary(draw_list, projector, Frame, m_profile, m_name, Samples, WidenRight, WidenLeft);
}
