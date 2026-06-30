/*
This Source Code Form is subject to the
terms of the Mozilla Public License, v.
2.0. If a copy of the MPL was not
distributed with this file, You can
obtain one at
http://mozilla.org/MPL/2.0/.
*/

#include "stdafx.h"
#include "editor/editorTerrain.hpp"

#include "scene/scene.h"
#include "scene/scenenode.h"
#include "simulation/simulation.h"
#include "rendering/renderer.h"
#include "model/vertex.h"
#include "model/AnimModel.h"
#include "model/Model3d.h"
#include "editor/editorTerrainStreamer.hpp"
#include "utilities/Globals.h"
#include "utilities/Logs.h"

#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <limits>
#include <string>
#include <filesystem>
#include <system_error>

namespace
{
	constexpr double kPi = 3.14159265358979323846;
}

bool editor_terrain::create(glm::dvec3 const &Center, int Cells, float CellSize, std::string const &TextureName,
                            height_sampler const &Sampler)
{
	if (Cells < 1 || CellSize <= 0.0f || simulation::Region == nullptr)
		return false;

	m_cells = Cells;
	m_cellsize = CellSize;
	double const half = 0.5 * static_cast<double>(Cells) * CellSize;
	m_x0 = Center.x - half;
	m_z0 = Center.z - half;
	m_heights.assign(static_cast<std::size_t>(Cells + 1) * (Cells + 1), static_cast<float>(Center.y));

	// optionally seed the grid by sampling whatever geometry is already there (terrain capture)
	if (Sampler)
	{
		for (int iz = 0; iz <= Cells; ++iz)
			for (int ix = 0; ix <= Cells; ++ix)
			{
				double const vx = m_x0 + static_cast<double>(ix) * CellSize;
				double const vz = m_z0 + static_cast<double>(iz) * CellSize;
				double y;
				if (Sampler(vx, vz, y))
					m_heights[index(ix, iz)] = static_cast<float>(y);
			}
	}

	m_material = TextureName.empty() ? null_handle : GfxRenderer->Fetch_Material(TextureName);

	// section-level shapes are rendered relative to the section centre, so that is our geometry origin
	scene::basic_section &sec = simulation::Region->section(Center);
	sec.create_geometry(); // ensure existing section geometry is already built (idempotent)
	m_origin = sec.m_area.center;
	m_section = &sec;

	std::vector<world_vertex> verts;
	build_vertices(verts, false);
	m_vertexcount = verts.size();

	scene::shape_node shape;
	shape.make_terrain(m_material, std::move(verts), m_origin);

	// upload to a dedicated bank; the renderer resolves draw calls by handle regardless of bank
	m_bank = GfxRenderer->Create_Bank();
	shape.create_geometry(m_bank); // sets the shape's geometry handle, clears its CPU vertices
	m_geometry = shape.data().geometry;

	glm::dvec3 const shapecenter = shape.data().area.center;
	float const shaperadius = shape.radius(); // cached inside make_terrain, vertices already gone

	sec.m_shapes.emplace_back(std::move(shape));
	// extend the section bounds so the new terrain isn't frustum-culled at its edges
	sec.m_area.radius = std::max(
	    sec.m_area.radius,
	    static_cast<float>(glm::length(sec.m_area.center - shapecenter) + shaperadius));

	return true;
}

glm::dvec3 editor_terrain::vertex_position(int Ix, int Iz) const
{
	return glm::dvec3(
	    m_x0 + static_cast<double>(Ix) * m_cellsize,
	    static_cast<double>(m_heights[index(Ix, Iz)]),
	    m_z0 + static_cast<double>(Iz) * m_cellsize);
}

glm::vec3 editor_terrain::vertex_normal(int Ix, int Iz) const
{
	// central differences on the heightfield; clamp to edges
	int const xl = std::max(0, Ix - 1), xr = std::min(m_cells, Ix + 1);
	int const zl = std::max(0, Iz - 1), zr = std::min(m_cells, Iz + 1);
	float const hl = m_heights[index(xl, Iz)], hr = m_heights[index(xr, Iz)];
	float const hd = m_heights[index(Ix, zl)], hu = m_heights[index(Ix, zr)];
	float const dx = static_cast<float>((xr - xl)) * m_cellsize;
	float const dz = static_cast<float>((zr - zl)) * m_cellsize;
	glm::vec3 n(-(hr - hl) / (dx > 0.f ? dx : 1.f), 1.0f, -(hu - hd) / (dz > 0.f ? dz : 1.f));
	return glm::normalize(n);
}

world_vertex editor_terrain::make_vertex(int Ix, int Iz) const
{
	world_vertex v;
	v.position = vertex_position(Ix, Iz);
	v.normal = vertex_normal(Ix, Iz);
	v.texture = glm::vec2(static_cast<float>(Ix), static_cast<float>(Iz));
	return v;
}

// emits one quad (two upward-facing triangles) spanning grid corners (X0,Z0)..(X1,Z1)
void editor_terrain::emit_quad(int X0, int Z0, int X1, int Z1, std::vector<world_vertex> &Out) const
{
	world_vertex const v00 = make_vertex(X0, Z0);
	world_vertex const v10 = make_vertex(X1, Z0);
	world_vertex const v01 = make_vertex(X0, Z1);
	world_vertex const v11 = make_vertex(X1, Z1);

	Out.push_back(v00);
	Out.push_back(v01);
	Out.push_back(v10);

	Out.push_back(v11);
	Out.push_back(v10);
	Out.push_back(v01);
}

// true if every grid vertex inside the block stays within Error of the bilinear plane of its corners
bool editor_terrain::block_flat(int X0, int Z0, int X1, int Z1, float Error) const
{
	float const h00 = m_heights[index(X0, Z0)];
	float const h10 = m_heights[index(X1, Z0)];
	float const h01 = m_heights[index(X0, Z1)];
	float const h11 = m_heights[index(X1, Z1)];
	double const wx = X1 - X0, wz = Z1 - Z0;

	for (int iz = Z0; iz <= Z1; ++iz)
		for (int ix = X0; ix <= X1; ++ix)
		{
			double const tx = (wx > 0.0) ? (ix - X0) / wx : 0.0;
			double const tz = (wz > 0.0) ? (iz - Z0) / wz : 0.0;
			double const top = h00 + tx * (h10 - h00);
			double const bot = h01 + tx * (h11 - h01);
			double const interp = top + tz * (bot - top);
			if (std::abs(static_cast<double>(m_heights[index(ix, iz)]) - interp) > Error)
				return false;
		}
	return true;
}

// adaptive quadtree: collapse flat blocks into a single quad, otherwise split into four
void editor_terrain::emit_block(int X0, int Z0, int X1, int Z1, float Error, std::vector<world_vertex> &Out) const
{
	bool const splitx = (X1 - X0) > 1;
	bool const splitz = (Z1 - Z0) > 1;

	if ((!splitx && !splitz) || block_flat(X0, Z0, X1, Z1, Error))
	{
		emit_quad(X0, Z0, X1, Z1, Out);
		return;
	}

	int const xm = splitx ? (X0 + X1) / 2 : X1;
	int const zm = splitz ? (Z0 + Z1) / 2 : Z1;

	emit_block(X0, Z0, xm, zm, Error, Out);
	if (splitx)
		emit_block(xm, Z0, X1, zm, Error, Out);
	if (splitz)
		emit_block(X0, zm, xm, Z1, Error, Out);
	if (splitx && splitz)
		emit_block(xm, zm, X1, Z1, Error, Out);
}

void editor_terrain::build_vertices(std::vector<world_vertex> &Out, bool Simplify) const
{
	Out.clear();
	Out.reserve(static_cast<std::size_t>(m_cells) * m_cells * 6);

	if (Simplify)
	{
		emit_block(0, 0, m_cells, m_cells, m_simplify_error, Out);
		return;
	}

	for (int iz = 0; iz < m_cells; ++iz)
		for (int ix = 0; ix < m_cells; ++ix)
			emit_quad(ix, iz, ix + 1, iz + 1, Out);
}

void editor_terrain::regenerate(bool Simplify)
{
	if (!valid())
		return;

	std::vector<world_vertex> verts;
	build_vertices(verts, Simplify);

	gfx::vertex_array gpuverts;
	gpuverts.reserve(verts.size());
	for (auto const &v : verts)
		gpuverts.emplace_back(gfx::basic_vertex::convert(v, m_origin));
	gfx::userdata_array nouserdata;

	// fast path: same vertex count -> in-place swap into the existing chunk
	if (gpuverts.size() == m_vertexcount && (m_geometry.bank != 0 || m_geometry.chunk != 0))
	{
		GfxRenderer->Replace(gpuverts, nouserdata, m_geometry, GL_TRIANGLES);
		return;
	}

	// count changed (optimize / un-optimize): upload a fresh chunk and point the shape at it
	gfx::geometry_handle const newhandle = GfxRenderer->Insert(gpuverts, nouserdata, m_bank, GL_TRIANGLES);
	if (m_section != nullptr)
	{
		for (auto &shape : m_section->m_shapes)
		{
			auto const h = shape.data().geometry;
			if (h.bank == m_geometry.bank && h.chunk == m_geometry.chunk)
			{
				shape.geometry(newhandle);
				break;
			}
		}
	}
	m_geometry = newhandle;
	m_vertexcount = gpuverts.size();
}

void editor_terrain::optimize(float ErrorMetres)
{
	m_simplify = true;
	m_simplify_error = (ErrorMetres > 0.0f ? ErrorMetres : 0.01f);
	m_dirty = false;
	regenerate(true);
}

void editor_terrain::unoptimize()
{
	m_simplify = false;
	regenerate(false);
}

void editor_terrain::destroy()
{
	if (m_section != nullptr)
	{
		// erase the shape whose geometry handle is ours; other shapes keep their handles (so they
		// keep rendering), and the geometry GC reclaims our now-undrawn chunk's GPU memory
		for (auto it = m_section->m_shapes.begin(); it != m_section->m_shapes.end(); ++it)
		{
			auto const h = it->data().geometry;
			if (h.bank == m_geometry.bank && h.chunk == m_geometry.chunk)
			{
				m_section->m_shapes.erase(it);
				break;
			}
		}
	}
	// release this chunk's dedicated GPU bank (deferred past frames-in-flight on Vulkan). the bank is
	// never drawn again - a chunk re-entering the radius builds a fresh one - so nothing re-uploads.
	if (m_bank.bank != 0 && GfxRenderer != nullptr)
		GfxRenderer->Release_Bank(m_bank);

	m_section = nullptr;
	m_geometry = gfx::geometry_handle{0, 0};
	m_bank = gfx::geometrybank_handle{0, 0};
	m_cells = 0; // mark invalid
	m_heights.clear();
}

bool editor_terrain::contains(double X, double Z) const
{
	double const x1 = m_x0 + static_cast<double>(m_cells) * m_cellsize;
	double const z1 = m_z0 + static_cast<double>(m_cells) * m_cellsize;
	return (X >= m_x0 && X <= x1 && Z >= m_z0 && Z <= z1);
}

double editor_terrain::height_at(double X, double Z) const
{
	double const fx = (X - m_x0) / m_cellsize;
	double const fz = (Z - m_z0) / m_cellsize;
	int ix = static_cast<int>(std::floor(fx));
	int iz = static_cast<int>(std::floor(fz));
	ix = std::clamp(ix, 0, m_cells - 1);
	iz = std::clamp(iz, 0, m_cells - 1);
	double const tx = std::clamp(fx - ix, 0.0, 1.0);
	double const tz = std::clamp(fz - iz, 0.0, 1.0);

	double const h00 = m_heights[index(ix, iz)];
	double const h10 = m_heights[index(ix + 1, iz)];
	double const h01 = m_heights[index(ix, iz + 1)];
	double const h11 = m_heights[index(ix + 1, iz + 1)];

	// matches the triangulation in build_vertices
	if (tx + tz <= 1.0)
		return h00 + tx * (h10 - h00) + tz * (h01 - h00);
	return h11 + (1.0 - tx) * (h01 - h11) + (1.0 - tz) * (h10 - h11);
}

bool editor_terrain::sculpt(double X, double Z, double Radius, double Strength)
{
	if (!valid() || Radius <= 0.0)
		return false;

	bool changed = false;
	for (int iz = 0; iz <= m_cells; ++iz)
		for (int ix = 0; ix <= m_cells; ++ix)
		{
			double const vx = m_x0 + static_cast<double>(ix) * m_cellsize;
			double const vz = m_z0 + static_cast<double>(iz) * m_cellsize;
			double const d = std::sqrt((vx - X) * (vx - X) + (vz - Z) * (vz - Z));
			if (d > Radius)
				continue;
			// smooth cosine falloff: full strength at the centre, zero at the rim
			double const falloff = 0.5 * (std::cos(kPi * d / Radius) + 1.0);
			m_heights[index(ix, iz)] += static_cast<float>(Strength * falloff);
			changed = true;
		}

	if (changed)
	{
		// sculpting edits the full-resolution mesh (fixed vertex count => fast in-place update);
		// mark dirty so it can be auto-simplified once the stroke finishes, and modified for saving
		m_simplify = false;
		m_dirty = true;
		m_modified = true;
		regenerate(false);
	}
	return changed;
}

glm::dvec3 editor_terrain::centre() const
{
	double const c = 0.5 * static_cast<double>(m_cells) * m_cellsize;
	double y = 0.0;
	if (!m_heights.empty())
		y = m_heights[index(m_cells / 2, m_cells / 2)];
	return glm::dvec3(m_x0 + c, y, m_z0 + c);
}

// ---------------------------------------------------------------------------
// shared terrain geometry helpers (relocated from editormode.cpp) + startup bake
// ---------------------------------------------------------------------------

bool triangle_height_at(glm::dvec3 const &a, glm::dvec3 const &b, glm::dvec3 const &c,
                        double const Px, double const Pz, double &OutY)
{
	double const ux = b.x - a.x, uz = b.z - a.z;
	double const vx = c.x - a.x, vz = c.z - a.z;
	double const wx = Px - a.x, wz = Pz - a.z;
	double const den = ux * vz - vx * uz;
	if (std::abs(den) < 1e-9)
		return false; // degenerate or vertical triangle, no defined height
	double const s = (wx * vz - vx * wz) / den;
	double const t = (ux * wz - wx * uz) / den;
	if (s < 0.0 || t < 0.0 || (s + t) > 1.0)
		return false;
	OutY = a.y + s * (b.y - a.y) + t * (c.y - a.y);
	return true;
}

void gather_submodel_triangles(TSubModel *Submodel, glm::dmat4 const &M, std::vector<world_triangle> &Out)
{
	for (TSubModel *sub = Submodel; sub != nullptr; sub = sub->Next)
	{
		glm::dmat4 mlocal = M;
		if ((sub->iFlags & 0xC000) && (sub->GetMatrix() != nullptr))
			mlocal = M * glm::dmat4(glm::make_mat4(sub->GetMatrix()->readArray()));

		if (sub->eType < TP_ROTATOR) // a drawable mesh, not a rotator/light/etc.
		{
			auto const handle = sub->m_geometry.handle;
			if (handle.bank != 0 || handle.chunk != 0)
			{
				auto const &verts = GfxRenderer->Vertices(handle);
				auto const &indices = GfxRenderer->Indices(handle);
				auto const to_world = [&](gfx::basic_vertex const &v) {
					return glm::dvec3(mlocal * glm::dvec4(glm::dvec3(v.position), 1.0));
				};
				if (false == indices.empty())
				{
					for (std::size_t i = 0; i + 2 < indices.size(); i += 3)
						Out.push_back({to_world(verts[indices[i]]), to_world(verts[indices[i + 1]]), to_world(verts[indices[i + 2]])});
				}
				else
				{
					for (std::size_t i = 0; i + 2 < verts.size(); i += 3)
						Out.push_back({to_world(verts[i]), to_world(verts[i + 1]), to_world(verts[i + 2])});
				}
			}
		}

		if (sub->Child != nullptr)
			gather_submodel_triangles(sub->Child, mlocal, Out); // children inherit this matrix
	}
}

namespace
{
	// streamed terrain chunk grid: one chunk per 250 m scene cell, at this vertex resolution.
	// 32 -> 2048 tris/chunk; a uniform full-res grid tiles seamlessly (no T-junction cracks) while
	// keeping ~4x fewer triangles than 64. bump if a scenery needs finer terrain relief.
	constexpr double kChunkSize = static_cast<double>(scene::EU07_CELLSIZE); // 250 m
	constexpr int kChunkCells = 32;                                         // (kChunkCells+1)^2 vertices

	// per-scenery chunk folder so different maps don't collide on shared chunk_X_Z world coordinates
	std::string chunk_dir()
	{
		std::string name = Global.SceneryFile;
		auto const slash = name.find_last_of("/\\");
		if (slash != std::string::npos)
			name.erase(0, slash + 1);
		while (!name.empty() && name.front() == '$')
			name.erase(0, 1); // rainsted override prefix
		auto const dot = name.find_last_of('.');
		if (dot != std::string::npos)
			name.erase(dot);
		if (name.empty())
			name = "default";
		return "terrain_chunks/" + name;
	}

	int floor_div(double V, double Size) { return static_cast<int>(std::floor(V / Size)); }

	// a source terrain triangle plus the material it was drawn with, so each chunk can record its
	// dominant material name (.etc v2) and streamed terrain keeps its texture.
	struct bake_tri
	{
		world_triangle t;
		material_handle mat;
	};

	// terrain triangles accumulated across the whole scenario load, baked once at the end so a 250 m
	// cell overlapped by several source nodes gets all of them (a per-node bake would miss the rest).
	std::vector<bake_tri> g_bake_tris;

	// world transform of a terrain model instance (matches the renderer / capture_terrain)
	glm::dmat4 model_world_matrix(TAnimModel *Terrain)
	{
		glm::dmat4 M(1.0);
		M = glm::translate(M, Terrain->location());
		glm::vec3 const angles = Terrain->Angles();
		if (angles.y != 0.0f) M = glm::rotate(M, glm::radians(static_cast<double>(angles.y)), glm::dvec3(0.0, 1.0, 0.0));
		if (angles.x != 0.0f) M = glm::rotate(M, glm::radians(static_cast<double>(angles.x)), glm::dvec3(1.0, 0.0, 0.0));
		if (angles.z != 0.0f) M = glm::rotate(M, glm::radians(static_cast<double>(angles.z)), glm::dvec3(0.0, 0.0, 1.0));
		glm::vec3 const scale = Terrain->Scale();
		M = glm::scale(M, glm::dvec3(scale));
		return M;
	}
}

void bake_reset()
{
	g_bake_tris.clear();
}

void bake_collect_shape(scene::shape_node const &Shape)
{
	material_handle const mat = Shape.data().material;
	auto const &verts = Shape.data().vertices; // world-space GL_TRIANGLES
	for (std::size_t i = 0; i + 2 < verts.size(); i += 3)
		g_bake_tris.push_back({world_triangle{glm::dvec3(verts[i].position),
		                                      glm::dvec3(verts[i + 1].position),
		                                      glm::dvec3(verts[i + 2].position)},
		                       mat});
}

void bake_collect_model(TAnimModel *Terrain)
{
	if (Terrain == nullptr || Terrain->pModel == nullptr)
		return;
	std::vector<world_triangle> tris;
	gather_submodel_triangles(Terrain->pModel->Root, model_world_matrix(Terrain), tris);
	material_handle const mat = Terrain->pModel->Root ? Terrain->pModel->Root->GetMaterial() : null_handle;
	for (auto const &t : tris)
		g_bake_tris.push_back({t, mat});
}

void bake_finalize_chunks()
{
	if (g_bake_tris.empty())
		return;
	auto const &tris = g_bake_tris;
	std::string const dir = chunk_dir();
	auto const bake_start = std::chrono::steady_clock::now();

	// bucket triangles by the 250 m chunk(s) their bounding box overlaps
	std::map<std::pair<int, int>, std::vector<bake_tri const *>> buckets;
	for (auto const &bt : tris)
	{
		auto const &t = bt.t;
		double const minx = std::min({t[0].x, t[1].x, t[2].x});
		double const maxx = std::max({t[0].x, t[1].x, t[2].x});
		double const minz = std::min({t[0].z, t[1].z, t[2].z});
		double const maxz = std::max({t[0].z, t[1].z, t[2].z});
		for (int cx = floor_div(minx, kChunkSize); cx <= floor_div(maxx, kChunkSize); ++cx)
			for (int cz = floor_div(minz, kChunkSize); cz <= floor_div(maxz, kChunkSize); ++cz)
				buckets[{cx, cz}].push_back(&bt);
	}

	int generated = 0, skipped = 0;
	double const step = kChunkSize / static_cast<double>(kChunkCells);
	std::size_t const vcount = static_cast<std::size_t>(kChunkCells + 1) * (kChunkCells + 1);

	for (auto const &entry : buckets)
	{
		int const cx = entry.first.first;
		int const cz = entry.first.second;
		if (terrain_streamer::chunk_file_exists(dir, cx, cz))
		{
			++skipped; // generate-if-missing: leave existing chunk untouched
			continue;
		}

		double const x0 = cx * kChunkSize;
		double const z0 = cz * kChunkSize;
		std::vector<float> heights(vcount, 0.0f);
		std::vector<char> found(vcount, 0);
		double sum = 0.0;
		std::size_t hits = 0;

		// spatial acceleration: bin the chunk's triangles into a GxG grid by bounding box, so each
		// sampled vertex only tests the few triangles in its bin instead of the whole bucket. without
		// this the bake is O(verts * bucket-size) and freezes the first load of large, finely
		// triangulated sceneries. a triangle covering a vertex always overlaps that vertex's bin, so
		// binning by bbox preserves correctness.
		constexpr int kBinGrid = 16;
		double const binsize = kChunkSize / static_cast<double>(kBinGrid);
		auto const binx = [&](double X) {
			return std::clamp(static_cast<int>(std::floor((X - x0) / binsize)), 0, kBinGrid - 1);
		};
		auto const binz = [&](double Z) {
			return std::clamp(static_cast<int>(std::floor((Z - z0) / binsize)), 0, kBinGrid - 1);
		};
		std::vector<std::vector<bake_tri const *>> bins(kBinGrid * kBinGrid);
		for (auto const *tp : entry.second)
		{
			auto const &t = tp->t;
			int const bx0 = binx(std::min({t[0].x, t[1].x, t[2].x}));
			int const bx1 = binx(std::max({t[0].x, t[1].x, t[2].x}));
			int const bz0 = binz(std::min({t[0].z, t[1].z, t[2].z}));
			int const bz1 = binz(std::max({t[0].z, t[1].z, t[2].z}));
			for (int bz = bz0; bz <= bz1; ++bz)
				for (int bx = bx0; bx <= bx1; ++bx)
					bins[bz * kBinGrid + bx].push_back(tp);
		}

		for (int iz = 0; iz <= kChunkCells; ++iz)
			for (int ix = 0; ix <= kChunkCells; ++ix)
			{
				double const X = x0 + ix * step;
				double const Z = z0 + iz * step;
				double best = -std::numeric_limits<double>::max();
				bool any = false;
				for (auto const *tp : bins[binz(Z) * kBinGrid + binx(X)])
				{
					auto const &t = tp->t;
					double const minx = std::min({t[0].x, t[1].x, t[2].x});
					double const maxx = std::max({t[0].x, t[1].x, t[2].x});
					double const minz = std::min({t[0].z, t[1].z, t[2].z});
					double const maxz = std::max({t[0].z, t[1].z, t[2].z});
					if (X < minx || X > maxx || Z < minz || Z > maxz)
						continue;
					double y;
					if (triangle_height_at(t[0], t[1], t[2], X, Z, y) && (!any || y > best))
					{
						best = y;
						any = true;
					}
				}
				std::size_t const vi = static_cast<std::size_t>(iz) * (kChunkCells + 1) + ix;
				if (any)
				{
					heights[vi] = static_cast<float>(best);
					found[vi] = 1;
					sum += best;
					++hits;
				}
			}

		if (hits == 0)
			continue; // chunk's triangles didn't cover any grid vertex; nothing to save

		// fill gaps (vertices outside the covered footprint) with the chunk mean to avoid spikes
		float const mean = static_cast<float>(sum / static_cast<double>(hits));
		for (std::size_t i = 0; i < vcount; ++i)
			if (!found[i])
				heights[i] = mean;

		// dominant material in this chunk (most frequent across its triangles) -> texture name
		std::map<material_handle, int> matcount;
		for (auto const *tp : entry.second)
			++matcount[tp->mat];
		material_handle dominant = null_handle;
		int domn = 0;
		for (auto const &mc : matcount)
			if (mc.second > domn) { domn = mc.second; dominant = mc.first; }
		std::string matname;
		if (dominant > null_handle && GfxRenderer != nullptr)
			if (auto const *m = GfxRenderer->Material(dominant))
				matname = m->GetName();

		terrain_streamer::save_height_grid(dir, cx, cz, heights, kChunkCells, matname);
		++generated;
	}

	auto const bake_ms = std::chrono::duration<double, std::milli>(
	                         std::chrono::steady_clock::now() - bake_start)
	                         .count();
	WriteLog("Terrain chunk bake: " + std::to_string(tris.size()) + " triangles -> generated "
	         + std::to_string(generated) + ", skipped " + std::to_string(skipped)
	         + " existing in " + std::to_string(static_cast<int>(bake_ms)) + " ms (dir " + dir + ")");

	g_bake_tris.clear();
	g_bake_tris.shrink_to_fit();
}

void bake_activate_streaming(int Radius)
{
	if (EditorTerrain.active())
		return; // an editorterrain scenery directive already owns the streamer
	std::string const dir = chunk_dir();
	std::error_code ec;
	if (!std::filesystem::exists(dir, ec))
		return; // no baked terrain for this scenery

	EditorTerrain.directory(dir);
	EditorTerrain.configure(kChunkCells, static_cast<float>(kChunkSize / kChunkCells), Radius, 0.0f, std::string());
	// no adaptive simplification: merging flat blocks into bigger quads leaves T-junction cracks
	// (sub-metre slivers) where a large quad meets a neighbour's smaller quads. full-res chunks share
	// identical edge vertices, so they tile seamlessly.
	EditorTerrain.simplify(false, 0.0f);
	EditorTerrain.active(true);
	WriteLog("Terrain streaming activated from baked chunks (dir " + dir
	         + ", radius " + std::to_string(Radius) + ")");
}
