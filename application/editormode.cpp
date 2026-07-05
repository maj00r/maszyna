/*
This Source Code Form is subject to the
terms of the Mozilla Public License, v.
2.0. If a copy of the MPL was not
distributed with this file, You can
obtain one at
http://mozilla.org/MPL/2.0/.
*/

#include "stdafx.h"
#include "application/editormode.h"
#include "application/editoruilayer.h"

#include "application/application.h"
#include "editor/editorSettings.hpp"
#include "utilities/Globals.h"
#include "simulation/simulation.h"
#include "simulation/simulationtime.h"
#include "simulation/simulationenvironment.h"
#include "utilities/Timer.h"
#include "Console.h"
#include "rendering/renderer.h"
#include "model/AnimModel.h"
#include "model/Model3d.h"
#include "world/Track.h"
#include <sstream>
#include <iomanip>
#include "utilities/Float3d.h"
#include "scene/scene.h"


#include "imgui/imgui.h"
#include "imgui/ImGuizmo.h"
#include "utilities/Logs.h"
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>
#include <memory>

// collects a track's segment(s): one for normal tracks, both paths for a switch/crossing (defined below)
static void collect_track_segments( TTrack *track, std::vector<std::shared_ptr<TSegment>> &out );

// Static member initialization
TCamera editor_mode::Camera;
bool editor_mode::m_focus_active = false;
bool editor_mode::m_change_history = false;
bool editor_mode::m_settings_open = false;

namespace
{
    using vec3 = glm::vec3;
    using dvec2 = glm::dvec2;

    inline bool is_release(int state)
    {
        return state == GLFW_RELEASE;
    }

    inline bool is_press(int state)
    {
        return state == GLFW_PRESS;
    }

    // tests whether the vertical line through (Px,Pz) passes over triangle abc; if so returns the
    // surface height at that point through OutY. used by the "snap to ground" (END) feature.
    inline bool triangle_height_at(glm::dvec3 const &a, glm::dvec3 const &b, glm::dvec3 const &c,
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
        if (s < 0.0 || t < 0.0 || s + t > 1.0)
            return false;
        OutY = a.y + s * (b.y - a.y) + t * (c.y - a.y);
        return true;
    }

    using world_triangle = std::array<glm::dvec3, 3>;

    // walks a model's submodel tree (mirroring the renderer's transform chain) and appends every
    // mesh triangle, in world space, to Out. siblings are iterated to avoid deep recursion.
    void gather_submodel_triangles(TSubModel *Submodel, glm::dmat4 const &M, std::vector<world_triangle> &Out)
    {
        for (TSubModel *sub = Submodel; sub != nullptr; sub = sub->Next)
        {
            glm::dmat4 mlocal = M;
            if (sub->iFlags & 0xC000 && sub->GetMatrix() != nullptr)
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

} 

bool editor_mode::editormode_input::init()
{
    return mouse.init() && keyboard.init();
}

void editor_mode::editormode_input::poll()
{
    keyboard.poll();
}

editor_mode::editor_mode() {
	m_userinterface = std::make_shared<editor_ui>();
 }

editor_ui *editor_mode::ui() const
{
    return static_cast<editor_ui *>(m_userinterface.get());
}

bool editor_mode::init()
{
    EditorSettings.load();
    Camera.Init({0, 15, 0}, {glm::radians(-30.0), glm::radians(180.0), 0}, nullptr);
    return m_input.init();
}

void editor_mode::apply_rotation_for_new_node(scene::basic_node *node, int rotation_mode, float fixed_rotation_value)
{
    if (!node)
        return;

    if (rotation_mode == functions_panel::RANDOM)
    {
        const vec3 rotation{0.0f, LocalRandom(0.0, 360.0), 0.0f};
        m_editor.rotate(node, rotation, 1);
    }
    else if (rotation_mode == functions_panel::FIXED)
    {
        const vec3 rotation{0.0f, fixed_rotation_value, 0.0f};
        m_editor.rotate(node, rotation, 0);
    }
}

TTrack *editor_mode::commit_track(glm::dvec3 const &p1, glm::dvec3 const &cv1, glm::dvec3 const &cv2, glm::dvec3 const &p2, double radius, double length)
{
    static int counter = 0;
    std::string const name = "editor_track_" + std::to_string( counter++ );

    // track node; geometry is the bezier p1 -> p1+cv1 -> p2+cv2 -> p2 (cv are offsets), radius only
    // drives mesh density / physics (see TTrack::Load)
    std::ostringstream src;
    src << std::fixed << std::setprecision(3)
        << "node -1 0 " << name << " track normal " << length
        << " 1.435 0.25 20.0 20 0 flat vis Rail_screw_used1 4.0 TpBpS-new2 0.2 0.5 1.1\n"
        << p1.x << ' ' << p1.y << ' ' << p1.z << " 0.0\n"
        << cv1.x << ' ' << cv1.y << ' ' << cv1.z << '\n'
        << cv2.x << ' ' << cv2.y << ' ' << cv2.z << '\n'
        << p2.x << ' ' << p2.y << ' ' << p2.z << " 0.0\n"
        << radius << '\n'
        << "endtrack\n";

    TTrack *track = simulation::State.create_track( src.str(), name );
    if( track == nullptr ) {
        ErrorLog( "editor: failed to create track" );
        return nullptr;
    }
    m_node = track;
    ui()->set_node( track );
    if( !m_pending_track_label.empty() ) { m_track_labels[track] = m_pending_track_label; }
    WriteLog( "editor: created track \"" + name + "\"" );
    return track;
}

void editor_mode::create_track_at(glm::dvec3 const &start, glm::dvec3 const &dir)
{
    // horizontal unit tangent
    glm::dvec3 T{ dir.x, 0.0, dir.z };
    double const tl = glm::length( T );
    T = ( tl > 1e-6 ) ? T / tl : glm::dvec3{ 0.0, 0.0, -1.0 };

    double const length = (double)ui()->track_length();
    int const type = ui()->track_type();
    bool const left = ui()->track_curve_left();

    {   // label shown by the track overlay
        std::ostringstream lbl;
        lbl << std::fixed << std::setprecision(0);
        if( type == track_panel::STRAIGHT )        { lbl << "Prosty L=" << length; }
        else if( type == track_panel::TRANSITION ) { lbl << "KP " << ui()->track_radius_start() << "->" << ui()->track_radius_end() << " L=" << length; }
        else if( type == track_panel::SWITCH )     { lbl << "Rozjazd R=" << ui()->track_radius(); }
        else                                       { lbl << "Luk R=" << ui()->track_radius() << " L=" << length; }
        m_pending_track_label = lbl.str();
    }

    if( type == track_panel::STRAIGHT ) {
        commit_track( start, glm::dvec3{ 0.0 }, glm::dvec3{ 0.0 }, start + T * length, 0.0, length );
        return;
    }

    if( type == track_panel::TRANSITION ) {
        // clothoid: curvature varies linearly from 1/radius_start to 1/radius_end (0 == straight)
        double const rs = (double)ui()->track_radius_start();
        double const re = (double)ui()->track_radius_end();
        double const ks = ( rs <= 1e-6 ) ? 0.0 : 1.0 / rs;
        double const ke = ( re <= 1e-6 ) ? 0.0 : 1.0 / re;
        create_transition( start, T, length, ks, ke, left );
        return;
    }

    // ARC / SWITCH: single circular arc approximated by a cubic bezier
    double const R = std::max( 1.0, (double)ui()->track_radius() );
    double theta = length / R;              // sweep angle
    theta = std::min( theta, M_PI * 0.9 );  // keep a single bezier accurate (< 180 deg)

    // rotation about the vertical axis
    auto const rotY = []( glm::dvec3 const &v, double a ) {
        double const c = std::cos( a ), s = std::sin( a );
        return glm::dvec3{ v.x * c + v.z * s, v.y, -v.x * s + v.z * c };
    };
    // derive the arc centre from the SAME rotation used for the sweep, so endpoint and its
    // tangent stay consistent (otherwise the bezier control points don't match the endpoints)
    double const turn = left ? 1.0 : -1.0;
    double const ang = turn * theta;
    glm::dvec3 const N = rotY( T, turn * M_PI_2 ); // unit normal toward the arc centre
    glm::dvec3 const center = start + N * R;
    glm::dvec3 const p2 = center + rotY( start - center, ang );
    glm::dvec3 const Tend = rotY( T, ang );
    double const h = R * ( 4.0 / 3.0 ) * std::tan( theta / 4.0 );

    if( type == track_panel::SWITCH ) {
        // switch: straight main route + diverging arc route, both sharing the entry point
        commit_switch( start, start + T * length, T * h, -Tend * h, p2, R, length );
        return;
    }
    // ARC
    commit_track( start, T * h, -Tend * h, p2, R, length );
}

void editor_mode::create_transition(glm::dvec3 const &start, glm::dvec3 const &tangent, double length, double kappa_start, double kappa_end, bool left)
{
    // A single track segment is one cubic bezier, so we represent the whole transition as one
    // bezier node: integrate the clothoid (curvature linear in arc length) only to find the true
    // end position and end tangent, then fit a single bezier through start/end with those tangents.
    int const N = 240; // integration steps (endpoint accuracy only; nothing is emitted per step)
    double const ds = length / N;
    double const turn = left ? 1.0 : -1.0;
    auto const rotY = []( glm::dvec3 const &v, double a ) {
        double const c = std::cos( a ), s = std::sin( a );
        return glm::dvec3{ v.x * c + v.z * s, v.y, -v.x * s + v.z * c };
    };

    glm::dvec3 pos = start;
    glm::dvec3 d{ tangent.x, 0.0, tangent.z };
    double const dl = glm::length( d );
    d = ( dl > 1e-6 ) ? d / dl : glm::dvec3{ 0.0, 0.0, -1.0 };
    glm::dvec3 const T = d;
    double phi_total = 0.0;

    for( int i = 0; i < N; ++i ) {
        double const s_mid = ( i + 0.5 ) * ds;
        double const kappa = kappa_start + ( kappa_end - kappa_start ) * ( s_mid / length );
        double const dphi = turn * kappa * ds;
        pos += rotY( d, dphi * 0.5 ) * ds; // advance using the mid-step heading
        d = rotY( d, dphi );
        phi_total += dphi;
    }
    glm::dvec3 const p2 = pos;      // end position
    glm::dvec3 const Tend = d;      // end tangent

    // symmetric handle sized as the equivalent circular arc of the total turn through the chord;
    // keeps the endpoint tangents exact (curvature is approximate, but it's a single bezier)
    double const chord = glm::length( p2 - start );
    double const Phi = std::abs( phi_total );
    double const h = ( Phi < 1e-4 )
        ? chord / 3.0
        : ( chord / ( 2.0 * std::sin( Phi * 0.5 ) ) ) * ( 4.0 / 3.0 ) * std::tan( Phi * 0.25 );

    // radius field: tightest of the two ends (drives mesh density / physics)
    double const r_end = ( std::abs( kappa_end ) > 1e-9 ) ? 1.0 / std::abs( kappa_end ) : 0.0;
    double const r_start = ( std::abs( kappa_start ) > 1e-9 ) ? 1.0 / std::abs( kappa_start ) : 0.0;
    double const radius = ( r_start > 0.0 && r_end > 0.0 ) ? std::min( r_start, r_end ) : std::max( r_start, r_end );

    commit_track( start, T * h, -Tend * h, p2, radius, length );
}

TTrack *editor_mode::commit_switch(glm::dvec3 const &entry, glm::dvec3 const &straightend, glm::dvec3 const &divcv1, glm::dvec3 const &divcv2, glm::dvec3 const &divend, double radius, double length)
{
    static int counter = 0;
    std::string const name = "editor_switch_" + std::to_string( counter++ );

    // switch node: two paths sharing the entry point. path 1 = straight main route (control
    // vectors and radius zero), path 2 = diverging arc (bezier control offsets + radius).
    std::ostringstream src;
    src << std::fixed << std::setprecision(3)
        << "node -1 0 " << name << " track switch 34.0"
        // normal ballast profile width (texwidth/slope): a standalone switch has no connected track
        // to take the profile from, so it uses its own fTexWidth/fTexSlope - keep them track-sized
        << " 1.435 0.24 15.0 20 2 flat vis Rail_screw_used1 4.0 TpBpS-new2 0.2 0.5 1.1\n"
        << entry.x << ' ' << entry.y << ' ' << entry.z << " 0.0\n"
        << "0.0 0.0 0.0\n"
        << "0.0 0.0 0.0\n"
        << straightend.x << ' ' << straightend.y << ' ' << straightend.z << " 0.0\n"
        << "0.0\n"
        << entry.x << ' ' << entry.y << ' ' << entry.z << " 0.0\n"
        << divcv1.x << ' ' << divcv1.y << ' ' << divcv1.z << '\n'
        << divcv2.x << ' ' << divcv2.y << ' ' << divcv2.z << '\n'
        << divend.x << ' ' << divend.y << ' ' << divend.z << " 0.0\n"
        << radius << '\n'
        // explicit trackbed material so the auto-generated switch ballast renders even for a
        // standalone switch (no connected track to copy the ballast texture from)
        << "trackbed TpBpS-new2\n"
        << "endtrack\n";

    TTrack *track = simulation::State.create_track( src.str(), name );
    if( track == nullptr ) {
        ErrorLog( "editor: failed to create switch" );
        return nullptr;
    }
    m_node = track;
    ui()->set_node( track );
    if( !m_pending_track_label.empty() ) { m_track_labels[track] = m_pending_track_label; }
    m_switch_meta[ track ] = switch_meta{ entry, straightend, divcv1, divcv2, divend, radius, length };
    WriteLog( "editor: created switch \"" + name + "\"" );
    return track;
}

// removes a track from the scene: out of its cell, rebuild that section's baked geometry,
// drop from the path lookup. the object itself is left orphaned on purpose (still referenced
// by scene groups) to avoid dangling pointers.
void editor_mode::delete_track(TTrack *track)
{
    if( track == nullptr ) { return; }
    glm::dvec3 const loc = track->location();
    std::string const trackname = track->name();
    simulation::Region->erase( track );
    simulation::Region->section( loc ).rebuild_geometry();
    simulation::Paths.detach( trackname );
    m_track_labels.erase( track );
    if( m_node == track ) {
        m_node = nullptr;
        ui()->set_node( nullptr );
    }
}

// finds a track endpoint close to the given screen position (for extending / dragging)
bool editor_mode::pick_track_endpoint(float screenx, float screeny, bool allowswitch, TTrack *&track, int &endindex, glm::dvec3 &point, glm::dvec3 &outward)
{
    ImGuiIO const &io = ImGui::GetIO();
    glm::mat4 const view = GfxRenderer->Camera_View_Matrix();
    glm::dvec3 const camerapos = GfxRenderer->Camera_Position();
    float const fovy = glm::radians( Global.FieldOfView / Global.ZoomFactor );
    float const aspect = io.DisplaySize.y > 0.0f ? io.DisplaySize.x / io.DisplaySize.y : 1.0f;
    glm::mat4 const projection = glm::perspective( fovy, aspect, 0.1f, 10000.0f );

    auto const hnorm = []( glm::dvec3 v ) {
        v.y = 0.0;
        double const l = glm::length( v );
        return ( l > 1e-9 ) ? v / l : glm::dvec3{ 0.0, 0.0, -1.0 };
    };

    float const pickradius = 14.0f; // [px]
    float best = pickradius * pickradius;
    track = nullptr; endindex = -1;
    std::vector<std::shared_ptr<TSegment>> segs;
    for( auto *path : simulation::Paths.sequence() ) {
        if( path == nullptr ) { continue; }
        if( !allowswitch && path->SwitchExtension ) { continue; } // switch points are locked for dragging
        segs.clear();
        collect_track_segments( path, segs );
        for( size_t s = 0; s < segs.size(); ++s ) {
            if( !segs[ s ] ) { continue; }
            glm::dvec3 const pts[ 2 ] = { glm::dvec3{ segs[ s ]->FastGetPoint_0() }, glm::dvec3{ segs[ s ]->FastGetPoint_1() } };
            for( int e = 0; e < 2; ++e ) {
                glm::vec4 const clip = projection * view * glm::vec4( glm::vec3( pts[ e ] - camerapos ), 1.0f );
                if( clip.w <= 1e-4f ) { continue; }
                glm::vec2 const ndc = glm::vec2( clip ) / clip.w;
                float const sx = ( ndc.x * 0.5f + 0.5f ) * io.DisplaySize.x;
                float const sy = ( 1.0f - ( ndc.y * 0.5f + 0.5f ) ) * io.DisplaySize.y;
                float const dx = sx - screenx, dy = sy - screeny;
                float const d2 = dx * dx + dy * dy;
                if( d2 < best ) {
                    best = d2;
                    track = path;
                    endindex = (int)s * 2 + e;
                    point = pts[ e ];
                    // outward continuation direction beyond this end (same convention as snap)
                    outward = hnorm( e == 0 ? -segs[ s ]->GetDirection1() : -segs[ s ]->GetDirection2() );
                }
            }
        }
    }
    return track != nullptr;
}

// marches a single chain element from (pos, dir); optionally emits the result track
// (P1 = element start, P2 = element end: track directionality always follows the march)
TTrack *editor_mode::march_element( chain_element &el, glm::dvec3 &pos, glm::dvec3 &dir, bool emit )
{
    auto const rotY = []( glm::dvec3 const &v, double a ) {
        double const c = std::cos( a ), s = std::sin( a );
        return glm::dvec3{ v.x * c + v.z * s, v.y, -v.x * s + v.z * c };
    };
    TTrack *made = nullptr;
    std::ostringstream lbl;
    lbl << std::fixed << std::setprecision( 0 );

    if( el.type == track_panel::STRAIGHT ) {
        // one niweleta straight can be cut into several collinear track pieces
        int const pieces = 1 + std::max( 0, el.cuts );
        double const step = el.length / pieces;
        for( int i = 0; i < pieces; ++i ) {
            glm::dvec3 const end = pos + dir * step;
            if( emit ) {
                std::ostringstream plbl;
                plbl << std::fixed << std::setprecision( 0 ) << "Prosty L=" << step;
                m_pending_track_label = plbl.str();
                made = commit_track( pos, glm::dvec3{ 0.0 }, glm::dvec3{ 0.0 }, end, 0.0, step );
                if( made != nullptr ) { el.tracks.push_back( made ); }
            }
            pos = end;
        }
    }
    else if( el.type == track_panel::ARC ) {
        double const R = std::max( 1.0, el.radius );
        double const turn = el.left ? 1.0 : -1.0;
        double const theta = std::min( el.length / R, M_PI * 0.9 );
        glm::dvec3 const N = rotY( dir, turn * M_PI_2 );
        glm::dvec3 const centre = pos + N * R;
        glm::dvec3 const end = centre + rotY( pos - centre, turn * theta );
        glm::dvec3 const Tend = rotY( dir, turn * theta );
        double const h = R * ( 4.0 / 3.0 ) * std::tan( theta / 4.0 );
        if( emit ) {
            lbl << "Luk R=" << R << " L=" << el.length;
            m_pending_track_label = lbl.str();
            made = commit_track( pos, dir * h, -Tend * h, end, R, el.length );
            if( made != nullptr ) { el.tracks.push_back( made ); }
        }
        pos = end;
        dir = Tend;
    }
    else { // TRANSITION: clothoid, parameters (R0 -> R1, L) are never deformed by editing
        double const k0 = ( el.radius0 > 1e-6 ) ? 1.0 / el.radius0 : 0.0;
        double const k1 = ( el.radius > 1e-6 ) ? 1.0 / el.radius : 0.0;
        double const turn = el.left ? 1.0 : -1.0;
        int const N = 240;
        double const ds = el.length / N;
        glm::dvec3 ipos = pos, idir = dir;
        double phi = 0.0;
        for( int i = 0; i < N; ++i ) {
            double const smid = ( i + 0.5 ) * ds;
            double const kappa = k0 + ( k1 - k0 ) * ( smid / el.length );
            double const dphi = turn * kappa * ds;
            ipos += rotY( idir, dphi * 0.5 ) * ds;
            idir = rotY( idir, dphi );
            phi += dphi;
        }
        double const chord = glm::length( ipos - pos );
        double const Phi = std::abs( phi );
        double const h = ( Phi < 1e-4 )
            ? chord / 3.0
            : ( chord / ( 2.0 * std::sin( Phi * 0.5 ) ) ) * ( 4.0 / 3.0 ) * std::tan( Phi * 0.25 );
        double const rmesh = ( k1 > 1e-9 && k0 > 1e-9 ) ? std::min( 1.0 / k1, 1.0 / k0 ) : std::max( el.radius, el.radius0 );
        if( emit ) {
            lbl << "KP " << el.radius0 << "->" << el.radius << " L=" << el.length;
            m_pending_track_label = lbl.str();
            made = commit_track( pos, dir * h, -idir * h, ipos, rmesh, el.length );
            if( made != nullptr ) { el.tracks.push_back( made ); }
        }
        pos = ipos;
        dir = idir;
    }
    return made;
}

// regenerates a chain: resolves the (possibly switch-anchored) origin and re-marches all elements
void editor_mode::regenerate_chain(int index)
{
    if( index < 0 || index >= (int)m_chains.size() ) { return; }
    auto &ch = m_chains[ index ];

    auto const hnorm = []( glm::dvec3 v ) {
        v.y = 0.0;
        double const l = glm::length( v );
        return ( l > 1e-9 ) ? v / l : glm::dvec3{ 0.0, 0.0, -1.0 };
    };

    // anchored chains take origin/direction from the switch outlet (main = end 1, diverging = end 3)
    if( ch.anchor != nullptr && ch.anchor->SwitchExtension ) {
        auto const seg = ch.anchor->SwitchExtension->Segments[ ch.anchor_end == 3 ? 1 : 0 ];
        if( seg ) {
            glm::dvec3 p{ seg->FastGetPoint_1() };
            p.y -= 0.18;
            ch.origin = p;
            ch.direction = hnorm( -seg->GetDirection2() );
        }
    }

    for( auto &el : ch.elements ) {
        for( auto *t : el.tracks ) { delete_track( t ); }
        el.tracks.clear();
    }
    ch.joints.clear();

    glm::dvec3 pos = ch.origin;
    glm::dvec3 dir = hnorm( ch.direction );
    ch.joints.push_back( pos );
    for( auto &el : ch.elements ) {
        march_element( el, pos, dir, true );
        ch.joints.push_back( pos );
    }
    ch.endtangent = dir;
    m_pending_track_label.clear();
}

// finds a chain joint close to the given screen position; transition-curve end joints are
// locked (their parameters are never deformed), the origin of an anchored chain likewise
bool editor_mode::pick_chain_joint(float screenx, float screeny, int &chain, int &joint)
{
    ImGuiIO const &io = ImGui::GetIO();
    glm::mat4 const view = GfxRenderer->Camera_View_Matrix();
    glm::dvec3 const camerapos = GfxRenderer->Camera_Position();
    float const fovy = glm::radians( Global.FieldOfView / Global.ZoomFactor );
    float const aspect = io.DisplaySize.y > 0.0f ? io.DisplaySize.x / io.DisplaySize.y : 1.0f;
    glm::mat4 const projection = glm::perspective( fovy, aspect, 0.1f, 10000.0f );

    float const pickradius = 14.0f;
    float best = pickradius * pickradius;
    chain = -1; joint = -1;
    for( size_t c = 0; c < m_chains.size(); ++c ) {
        auto const &ch = m_chains[ c ];
        for( size_t j = 0; j < ch.joints.size(); ++j ) {
            if( j == 0 && ch.anchor != nullptr ) { continue; } // anchored origin is owned by the switch
            if( j > 0 && ch.elements[ j - 1 ].type == track_panel::TRANSITION ) { continue; } // KP locked
            glm::vec4 const clip = projection * view * glm::vec4( glm::vec3( ch.joints[ j ] - camerapos ), 1.0f );
            if( clip.w <= 1e-4f ) { continue; }
            glm::vec2 const ndc = glm::vec2( clip ) / clip.w;
            float const sx = ( ndc.x * 0.5f + 0.5f ) * io.DisplaySize.x;
            float const sy = ( 1.0f - ( ndc.y * 0.5f + 0.5f ) ) * io.DisplaySize.y;
            float const dx = sx - screenx, dy = sy - screeny;
            if( dx * dx + dy * dy < best ) {
                best = dx * dx + dy * dy;
                chain = (int)c;
                joint = (int)j;
            }
        }
    }
    return chain != -1;
}

bool editor_mode::snap_track_start(glm::dvec3 &start, glm::dvec3 &dir)
{
    double const snapradius = 8.0; // [m]
    double best = snapradius * snapradius;
    bool found = false;
    // compare horizontally (ignore the railhead height offset) so proximity is stable
    auto const hdist2 = []( glm::dvec3 const &a, glm::dvec3 const &b ) {
        double const dx = a.x - b.x, dz = a.z - b.z;
        return dx * dx + dz * dz;
    };
    glm::dvec3 const click = start; // snap against the fixed click point, not a moving target
    std::vector<std::shared_ptr<TSegment>> segs;
    for( auto *path : simulation::Paths.sequence() ) {
        if( path == nullptr ) { continue; }
        segs.clear();
        collect_track_segments( path, segs );
        for( auto const &seg : segs ) {
            if( !seg ) { continue; }
            glm::dvec3 const p1{ seg->FastGetPoint_0() };
            glm::dvec3 const p2{ seg->FastGetPoint_1() };
            double const d1 = hdist2( p1, click );
            if( d1 < best ) { best = d1; start = p1; dir = -seg->GetDirection1(); found = true; }
            double const d2 = hdist2( p2, click );
            if( d2 < best ) { best = d2; start = p2; dir = -seg->GetDirection2(); found = true; }
        }
    }
    // snapped endpoints are geometry points at railhead height; TTrack::Load re-adds the 0.18 m
    // railhead offset when the new track is built, so bring the start back down to the node base level
    if( found ) { start.y -= 0.18; }
    return found;
}

glm::dvec3 editor_mode::cursor_ground_point() const
{
    ImGuiIO const &io = ImGui::GetIO();
    glm::mat4 const view = GfxRenderer->Camera_View_Matrix();          // camera-relative (rotation only)
    glm::dvec3 const camerapos = GfxRenderer->Camera_Position();
    float const fovy = glm::radians( Global.FieldOfView / Global.ZoomFactor );
    float const aspect = io.DisplaySize.y > 0.0f ? io.DisplaySize.x / io.DisplaySize.y : 1.0f;
    glm::mat4 const projection = glm::perspective( fovy, aspect, 0.1f, 10000.0f );

    float const nx = ( io.MousePos.x / io.DisplaySize.x ) * 2.0f - 1.0f;
    float const ny = 1.0f - ( io.MousePos.y / io.DisplaySize.y ) * 2.0f;
    glm::mat4 const invvp = glm::inverse( projection * view );
    glm::vec4 pn = invvp * glm::vec4( nx, ny, -1.0f, 1.0f );
    glm::vec4 pf = invvp * glm::vec4( nx, ny,  1.0f, 1.0f );
    glm::dvec3 const nearp = camerapos + glm::dvec3( glm::vec3( pn ) / pn.w );
    glm::dvec3 const farp  = camerapos + glm::dvec3( glm::vec3( pf ) / pf.w );

    glm::dvec3 const rd = farp - nearp;
    if( std::abs( rd.y ) < 1e-9 ) { return glm::dvec3{ nearp.x, 0.0, nearp.z }; } // ray ~parallel to ground
    double const t = -nearp.y / rd.y;
    glm::dvec3 hit = nearp + rd * t;
    hit.y = 0.0;
    return hit;
}

void editor_mode::create_straight_track_ahead(double length)
{
    // horizontal camera forward from yaw (TCamera::RaLook: Angle.y = atan2(-x, -z))
    glm::dvec3 const forward{ -std::sin( (double)Camera.Angle.y ), 0.0, -std::cos( (double)Camera.Angle.y ) };
    glm::dvec3 const start{ Camera.Pos.x + forward.x * 20.0, 0.0, Camera.Pos.z + forward.z * 20.0 };
    create_track_at( start, forward );
}

void editor_mode::start_focus(scene::basic_node *node, double duration)
{
    if (!node)
        return;

    glm::dvec3 const center = node->location();

    // distance that frames the object's bounding sphere within the vertical FOV, with some margin
    double const radius = std::max(1.0, static_cast<double>(node->radius()));
    double const fovy = glm::radians(static_cast<double>(Global.FieldOfView) / std::max(0.01, static_cast<double>(Global.ZoomFactor)));
    double distance = radius / std::tan(fovy * 0.5) * 1.6;
    distance = std::clamp(distance, radius * 1.5, static_cast<double>(kMaxPlacementDistance));

    // keep the camera on the side it currently views from, so the move turns toward the object
    // rather than flying around it; fall back to a pleasant 3/4 direction when sitting on top of it
    glm::dvec3 dir = Camera.Pos - center;
    double const len = glm::length(dir);
    dir = len > 1e-3 ? dir / len : glm::normalize(glm::dvec3(1.0, 0.5, 1.0));

    m_focus_start_pos = Camera.Pos;
    m_focus_start_angle = Camera.Angle;
    m_focus_target_pos = center + dir * distance;

    // target orientation looks from the target position straight at the object
    glm::dvec3 look = center - m_focus_target_pos;
    double const looklen = glm::length(look);
    if (looklen > 1e-6)
        look /= looklen;
    m_focus_target_angle = glm::vec3(
        static_cast<float>(std::asin(glm::clamp(look.y, -1.0, 1.0))),   // pitch
        static_cast<float>(std::atan2(-look.x, -look.z)),               // yaw
        0.0f);                                                          // roll

    m_focus_active = true;
    m_focus_time = 0.0;
    m_focus_duration = duration;
}

void editor_mode::snap_to_ground(scene::basic_node *node)
{
    if (!node || !simulation::Region)
        return;

    glm::dvec3 const origin = node->location();
    if (!simulation::Region->point_inside(origin))
        return;

    // small tolerance so a node already resting on a surface still snaps cleanly to it
    double const epsilon = 0.05;
    double bestY = -std::numeric_limits<double>::max();
    bool found = false;

    // record the highest surface that is at or below the node's current position at its (x,z)
    auto consider_triangle = [&](glm::dvec3 const &a, glm::dvec3 const &b, glm::dvec3 const &c) {
        double y;
        if (triangle_height_at(a, b, c, origin.x, origin.z, y) && y <= origin.y + epsilon && y > bestY)
        {
            bestY = y;
            found = true;
        }
    };

    auto consider_shapes = [&](std::vector<scene::shape_node> const &shapes) {
        for (auto const &shape : shapes)
        {
            // quick reject: skip shapes whose bounding circle doesn't cover our (x,z) column
            auto const &sdata = shape.data();
            double const sdx = origin.x - sdata.area.center.x;
            double const sdz = origin.z - sdata.area.center.z;
            if (sdx * sdx + sdz * sdz > static_cast<double>(sdata.area.radius) * sdata.area.radius)
                continue;

            auto const &verts = sdata.vertices;
            for (std::size_t i = 0; i + 2 < verts.size(); i += 3)
                consider_triangle(verts[i].position, verts[i + 1].position, verts[i + 2].position);
        }
    };

    scene::basic_section &sec = simulation::Region->section(origin);
    // section level holds the large opaque geometry, including legacy terrain
    consider_shapes(sec.m_shapes);

    // scan a 3x3 neighbourhood of cells for smaller geometry and other model instances below us
    for (int dz = -1; dz <= 1; ++dz)
        for (int dx = -1; dx <= 1; ++dx)
        {
            scene::basic_cell &cell = sec.cell(origin, glm::ivec2(dx, dz));
            consider_shapes(cell.m_shapesopaque);
            consider_shapes(cell.m_shapestranslucent);

            // other instances are approximated by their bounding sphere, so a node can rest on top of them
            for (auto *inst : cell.m_instancesopaque)
            {
                if (!inst || inst == node)
                    continue;
                glm::dvec3 const ic = inst->location();
                double const r = static_cast<double>(inst->radius());
                double const idx = origin.x - ic.x, idz = origin.z - ic.z;
                double const horiz2 = idx * idx + idz * idz;
                if (horiz2 < r * r)
                {
                    double const ytop = ic.y + std::sqrt(r * r - horiz2);
                    if (ytop <= origin.y + epsilon && ytop > bestY)
                    {
                        bestY = ytop;
                        found = true;
                    }
                }
            }
        }

    // editable terrain patches keep their heightmap on the CPU, so query them directly
    for (editor_terrain *terrain : active_terrains())
    {
        if (!terrain->contains(origin.x, origin.z))
            continue;
        double const y = terrain->height_at(origin.x, origin.z);
        if (y <= origin.y + epsilon && y > bestY)
        {
            bestY = y;
            found = true;
        }
    }

    if (!found)
        return;

    push_snapshot(node, EditorSnapshot::Action::Move);
    glm::dvec3 target = origin;
    target.y = bestY;
    m_editor.translate(node, target, true); // true == apply the computed Y (free vertical move)
}

void editor_mode::handle_brush_mouse_hold(int Action, int Button)
{
    auto const mode = ui()->mode();
    auto const rotation_mode = ui()->rot_mode();
    auto const fixed_rotation_value = ui()->rot_val();

    if(mode != nodebank_panel::BRUSH)
        return;
    
    GfxRenderer->Pick_Node_Callback(
        [this, mode, rotation_mode, fixed_rotation_value, Action, Button](scene::basic_node * /*node*/) {
            const std::string *src = ui()->get_active_node_template();
            if (!src)
                return;

            std::string name = "editor_";

            glm::dvec3 newPos = clamp_mouse_offset_to_max(GfxRenderer->Mouse_Position());
            double distance = glm::distance(newPos, oldPos);
            if (distance < ui()->getSpacing())
                return;

            TAnimModel *cloned = simulation::State.create_model(*src, name, Camera.Pos + newPos);
            oldPos = newPos;
            if (!cloned)
                return;

            std::string new_name = "editor_" + cloned->uuid.to_string();

            cloned->m_name = new_name;
            
            std::string as_text;
            cloned->export_as_text(as_text);
            push_snapshot(cloned, EditorSnapshot::Action::Add, as_text);

            m_node = cloned;
            apply_rotation_for_new_node(m_node, rotation_mode, fixed_rotation_value);
            ui()->set_node(m_node);
        });
}

void editor_mode::add_to_hierarchy(scene::basic_node *node)
{
    if (!node) return;
    scene::Hierarchy[node->uuid.to_string()] = node;
}

void editor_mode::remove_from_hierarchy(scene::basic_node *node)
{
    if (!node) return;
    auto it = scene::Hierarchy.find(node->uuid.to_string());
    if (it != scene::Hierarchy.end())
        scene::Hierarchy.erase(it);
}

scene::basic_node* editor_mode::find_in_hierarchy(const std::string &uuid_str)
{
    if (uuid_str.empty()) return nullptr;
    auto it = scene::Hierarchy.find(uuid_str);
    return it != scene::Hierarchy.end() ? it->second : nullptr;
}

scene::basic_node* editor_mode::find_node_by_any(scene::basic_node *node_ptr, const std::string &uuid_str, const std::string &name)
{
    if (node_ptr) return node_ptr;
    
    if (!uuid_str.empty()) {
        auto *node = find_in_hierarchy(uuid_str);
        if (node) return node;
    }
    
    if (!name.empty()) {
        return simulation::Instances.find(name);
    }
    
    return nullptr;
}

void editor_mode::push_snapshot(scene::basic_node *node, EditorSnapshot::Action Action, std::string const &Serialized)
{
    if (!node)
        return;

    if(m_max_history_size >= 0 && (int)m_history.size() >= m_max_history_size)
    {
        m_history.erase(m_history.begin(), m_history.begin() + ((int)m_history.size() - m_max_history_size + 1));
    }

    EditorSnapshot snap;
    snap.action = Action;
    snap.node_name = node->name();
    snap.position = node->location();
    snap.uuid = node->uuid;
    
    if (auto *model = dynamic_cast<TAnimModel *>(node))
    {
        snap.rotation = model->Angles();
        snap.scale = model->Scale();
    }
    else
    {
        snap.rotation = glm::vec3(0.0f);
        snap.scale = glm::vec3(1.0f);
    }

    if (Action == EditorSnapshot::Action::Delete || Action == EditorSnapshot::Action::Add)
    {
        if (!Serialized.empty())
            snap.serialized = Serialized;
        else
            node->export_as_text(snap.serialized);
    }


    snap.node_ptr = node;

    m_history.push_back(std::move(snap));
    g_redo.clear();
}

glm::dvec3 editor_mode::clamp_mouse_offset_to_max(const glm::dvec3 &offset)
{
    double len = glm::length(offset);
    if (len <= static_cast<double>(kMaxPlacementDistance) || len <= 1e-6)
        return offset;
    return glm::normalize(offset) * static_cast<double>(kMaxPlacementDistance);
}

void editor_mode::nullify_history_pointers(scene::basic_node *node)
{
    if (!node)
        return;

    for (auto &s : m_history)
    {
        if (s.node_ptr == node)
            s.node_ptr = nullptr;
    }

    for (auto &s : g_redo)
    {
        if (s.node_ptr == node)
            s.node_ptr = nullptr;
    }
}

void editor_mode::undo_last()
{
    if (m_history.empty())
        return;

    EditorSnapshot snap = m_history.back();
    m_history.pop_back();

    if (snap.action == EditorSnapshot::Action::Delete)
    {
        // undo delete -> recreate model
        EditorSnapshot redoSnap;
        redoSnap.action = EditorSnapshot::Action::Delete;
        redoSnap.node_name = snap.node_name;
        redoSnap.serialized = snap.serialized;
        redoSnap.position = snap.position;
        redoSnap.node_ptr = nullptr;
        g_redo.push_back(std::move(redoSnap));

        TAnimModel *created = simulation::State.create_model(snap.serialized, snap.node_name, snap.position);
        if (created)
        {
            created->location(snap.position);
            created->Angles(snap.rotation);
            m_node = created;
            m_node->uuid = snap.uuid; // restore original UUID for better tracking (not strictly necessary) 
            add_to_hierarchy(created);
            ui()->set_node(m_node);
        }
        return;
    }

    scene::basic_node *target = find_node_by_any(snap.node_ptr, snap.uuid.to_string(), snap.node_name);
    if (!target)
        return;

    EditorSnapshot current;
    current.action = snap.action;
    current.node_name = snap.node_name;
    current.node_ptr = target;
    current.position = target->location();
    if (auto *model = dynamic_cast<TAnimModel *>(target))
    {
        current.rotation = model->Angles();
        current.scale = model->Scale();
    }
    else
        current.rotation = glm::vec3(0.0f);
    g_redo.push_back(std::move(current));

    if (snap.action == EditorSnapshot::Action::Add)
    {
        // undo add -> delete the instance
        if (auto *model = dynamic_cast<TAnimModel *>(target))
        {
          
            nullify_history_pointers(model);
            remove_from_hierarchy(model);
            simulation::State.delete_model(model);
            m_node = nullptr;
            ui()->set_node(nullptr);
        }
        return;
    }

    target->location(snap.position);

    if (auto *model = dynamic_cast<TAnimModel *>(target))
    {
        glm::vec3 cur = model->Angles();
        glm::vec3 delta = snap.rotation - cur;
        m_editor.rotate(target, delta, 0);
        model->Scale(snap.scale);
    }

    m_node = target;
    ui()->set_node(m_node);
}

void editor_mode::redo_last()
{
    if (g_redo.empty())
        return;

    EditorSnapshot snap = g_redo.back();
    g_redo.pop_back();

    // handle delete redo (re-delete) separately
    if (snap.action == EditorSnapshot::Action::Delete)
    {
        EditorSnapshot hist;
        hist.action = snap.action;
        hist.node_name = snap.node_name;
        hist.serialized = snap.serialized;
        hist.position = snap.position;
        hist.uuid = snap.uuid;
        m_history.push_back(std::move(hist));

        scene::basic_node *target = simulation::Instances.find(snap.node_name);
        if (target)
        {
            if (auto *model = dynamic_cast<TAnimModel *>(target))
            {
                nullify_history_pointers(model);
                remove_from_hierarchy(model);
                simulation::State.delete_model(model);
                m_node = nullptr;
                ui()->set_node(nullptr);
            }
        }
        return;
    }

    scene::basic_node *target = find_node_by_any(snap.node_ptr, snap.uuid.to_string(), snap.node_name);

    EditorSnapshot hist;
    hist.action = snap.action;
    hist.node_name = snap.node_name;
    hist.node_ptr = target;

    if (target)
    {
        hist.position = target->location();
        if (auto *model = dynamic_cast<TAnimModel *>(target))
        {
            hist.rotation = model->Angles();
            hist.scale = model->Scale();
        }
        hist.uuid = snap.uuid;
    }
    m_history.push_back(std::move(hist));

    if (snap.action == EditorSnapshot::Action::Add)
    {
        TAnimModel *created = simulation::State.create_model(snap.serialized, snap.node_name, snap.position);
        if (created)
        {
            created->location(snap.position);
            created->Angles(snap.rotation);
            created->Scale(snap.scale);
            m_node = created;
            m_node->uuid = snap.uuid;
            ui()->set_node(m_node);
            if (!m_history.empty())
                m_history.back().node_ptr = created;
        }
        return;
    }

    if (!target)
        return;

    // apply redo position
    target->location(snap.position);
    if (auto *model = dynamic_cast<TAnimModel *>(target))
    {
        glm::vec3 cur = model->Angles();
        glm::vec3 delta = snap.rotation - cur;
        m_editor.rotate(target, delta, 0);
        model->Scale(snap.scale);
    }

    m_node = target;
    ui()->set_node(m_node);
}

bool editor_mode::update()
{
    Timer::UpdateTimers(true);

    simulation::State.update_clocks();
    simulation::Environment.update();

    auto const deltarealtime = Timer::GetDeltaRenderTime();

    // reconcile camera fly-mode with the real right-button state. ImGui is always fed the button
    // events (even when it captures the mouse), so io.MouseDown[1] is authoritative; if a release
    // was swallowed by an ImGui window while flying, force the editor out of fly-mode here so the
    // camera doesn't get stuck spinning with a hidden cursor.
    if (!ImGui::GetIO().MouseDown[1] && m_input.mouse.button(GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
    {
        m_input.mouse.button(GLFW_MOUSE_BUTTON_RIGHT, GLFW_RELEASE);
        Application.set_cursor(GLFW_CURSOR_NORMAL);
    }

    // fixed step render time routines (50 Hz)
    fTime50Hz += deltarealtime; // accumulate even when paused to keep frame reads stable
    while (fTime50Hz >= 1.0 / 50.0)
    {
#ifdef _WIN32
        Console::Update();
#endif
        m_userinterface->update();

        // update brush settings visibility depending on panel mode
        ui()->toggleBrushSettings(ui()->mode() == nodebank_panel::BRUSH);

        // "finish niweleta" ends the active chain; the next click starts a new one
        if (ui()->consume_track_finish())
            m_active_chain = -1;

        if (mouseHold)
        {
            // process continuous brush placement
            if(ui()->mode() == nodebank_panel::BRUSH)
                handle_brush_mouse_hold(GLFW_REPEAT, GLFW_MOUSE_BUTTON_LEFT);
        }

        // decelerate camera velocity with thresholding
        Camera.Velocity *= 0.65f;
        if (std::abs(Camera.Velocity.x) < 0.01)
            Camera.Velocity.x = 0.0;
        if (std::abs(Camera.Velocity.y) < 0.01)
            Camera.Velocity.y = 0.0;
        if (std::abs(Camera.Velocity.z) < 0.01)
            Camera.Velocity.z = 0.0;

        fTime50Hz -= 1.0 / 50.0;
 
    }

    // variable step routines
    update_camera(deltarealtime);

    // active drag (chain joint or switch) follows the cursor on the ground plane; geometry
    // is re-marched once, on mouse release
    if( m_dragactive ) {
        m_dragpos = cursor_ground_point();
    }

    simulation::Region->update_sounds();
    audio::renderer.update(Global.iPause ? 0.0 : deltarealtime);

    GfxRenderer->Update(deltarealtime);

    simulation::is_ready = true;

    // note: the streamer is advanced centrally in the application main loop (so it runs in every
    // mode), using the published Global.pCamera; nothing to do here

    // continuous terrain sculpting while the left mouse button is held in sculpt mode
    if (m_terrain_sculpt && mouseHold)
        handle_terrain_sculpt(deltarealtime);

    // debounced auto mesh simplification: once sculpting has settled for a short while, simplify
    // any chunk that was edited. holding the brush keeps the timer reset so we don't churn mid-stroke.
    if (m_terrain_auto_optimize)
    {
        auto const terrains = active_terrains();
        bool any_dirty = false;
        for (editor_terrain *terrain : terrains)
            if (terrain->dirty())
            {
                any_dirty = true;
                break;
            }

        if (!any_dirty || (m_terrain_sculpt && mouseHold))
        {
            m_terrain_idle = 0.0; // actively editing (or nothing pending): hold off
        }
        else
        {
            m_terrain_idle += deltarealtime;
            if (m_terrain_idle >= 0.5) // settle time
            {
                for (editor_terrain *terrain : terrains)
                    if (terrain->dirty())
                        terrain->optimize(m_terrain_simplify_error);
                m_terrain_idle = 0.0;
            }
        }
    }

    // --- ImGuizmo: in-viewport transform gizmo for the selected node ---
    render_gizmo();

    // --- 2D overlay: selected track's control-vector handles + type/radius label ---
    render_track_overlay();

    // --- ImGui: Editor Settings & History windows ---
    if(m_settings_open)
        render_settings();

    if(!m_change_history) return true;

    render_change_history();

    return true;
}

void editor_mode::render_settings()
{
    ImGui::Begin("Editor Settings", &m_settings_open, ImGuiWindowFlags_AlwaysAutoResize);

    ImGui::TextUnformatted("Camera movement");

    const char *schemes[] = {"WSAD (new)", "Arrows (legacy)"};
    int current = EditorSettings.movement() == editorSettings::movement_scheme::legacy ? 1 : 0;
    if (ImGui::Combo("##movement_scheme", &current, schemes, IM_ARRAYSIZE(schemes)))
    {
        EditorSettings.movement(current == 1 ? editorSettings::movement_scheme::legacy
                                             : editorSettings::movement_scheme::wsad);
        m_input.keyboard.apply_scheme();
        EditorSettings.save();
    }

    ImGui::Separator();
    ImGui::Checkbox("Transform gizmo (ImGuizmo)", &m_gizmo_enabled);

    render_terrain_ui();

    ImGui::End();
}

void editor_mode::render_terrain_ui()
{
    ImGui::Separator();
    ImGui::TextUnformatted("Terrain");

    ImGui::SetNextItemWidth(120.0f);
    ImGui::InputInt("Grid cells", &m_terrain_cells);
    m_terrain_cells = std::clamp(m_terrain_cells, 1, 512);
    ImGui::SetNextItemWidth(120.0f);
    ImGui::InputFloat("Cell size (m)", &m_terrain_cellsize);
    if (m_terrain_cellsize < 0.1f)
        m_terrain_cellsize = 0.1f;
    ImGui::SetNextItemWidth(120.0f);
    ImGui::InputFloat("Base height (m)", &m_terrain_baseheight);
    ImGui::SetNextItemWidth(200.0f);
    ImGui::InputText("Texture (optional)", m_terrain_texture, IM_ARRAYSIZE(m_terrain_texture));

    if (ImGui::Button("Create flat terrain"))
    {
        // centre the new patch horizontally on the camera, flat at the requested base height
        glm::dvec3 const center(Camera.Pos.x, static_cast<double>(m_terrain_baseheight), Camera.Pos.z);
        auto terrain = std::make_unique<editor_terrain>();
        if (terrain->create(center, m_terrain_cells, m_terrain_cellsize, std::string(m_terrain_texture)))
        {
            if (m_terrain_auto_optimize)
                terrain->optimize(m_terrain_simplify_error);
            m_terrains.push_back(std::move(terrain));
        }
        else
            WriteLog("Editor: failed to create terrain", logtype::generic);
    }

    ImGui::SetNextItemWidth(120.0f);
    ImGui::InputInt("Chunks / side", &m_terrain_chunks);
    m_terrain_chunks = std::clamp(m_terrain_chunks, 1, 32);
    ImGui::SameLine();
    if (ImGui::Button("Create chunked terrain"))
        create_chunked_terrain();
    ImGui::TextDisabled("total %d x %d m, %d chunks",
                        static_cast<int>(m_terrain_chunks * m_terrain_cells * m_terrain_cellsize),
                        static_cast<int>(m_terrain_chunks * m_terrain_cells * m_terrain_cellsize),
                        m_terrain_chunks * m_terrain_chunks);

    if (ImGui::Checkbox("Chunk edit mode (LMB add neighbour / Shift = delete)", &m_chunk_edit))
        if (m_chunk_edit)
            m_terrain_sculpt = false; // mutually exclusive with sculpting
    ImGui::Text("Grid chunks: %zu", m_grid_chunks.size());

    ImGui::Separator();
    ImGui::TextUnformatted("Streaming (open world, follows camera)");
    ImGui::SetNextItemWidth(120.0f);
    ImGui::InputInt("Stream radius", &m_stream_radius);
    m_stream_radius = std::clamp(m_stream_radius, 0, 16);
    ImGui::Checkbox("Persist edits to disk (16-bit)", &m_stream_persist);
    bool streaming = m_streamer.active();
    if (ImGui::Checkbox("Stream terrain", &streaming))
    {
        if (streaming)
        {
            // per-scenery chunk folder so chunks from different sceneries don't collide
            std::string scenery = Global.SceneryFile;
            auto const slash = scenery.find_last_of("/\\");
            if (slash != std::string::npos)
                scenery = scenery.substr(slash + 1);
            auto const dot = scenery.find_last_of('.');
            if (dot != std::string::npos)
                scenery = scenery.substr(0, dot);
            if (scenery.empty())
                scenery = "default";
            m_streamer.directory("editor_terrain/" + scenery);

            m_streamer.configure(m_terrain_cells, m_terrain_cellsize, m_stream_radius,
                                 m_terrain_baseheight, std::string(m_terrain_texture));
            m_streamer.simplify(m_terrain_auto_optimize, m_terrain_simplify_error);
            m_streamer.persist(m_stream_persist);

            // hand the authored grid chunks over to streaming: persist them to disk, then drop the
            // in-memory meshes so the streamer owns residency (it loads them back within the radius)
            for (auto &entry : m_grid_chunks)
                if (entry.second)
                    m_streamer.save_chunk(entry.first.first, entry.first.second, *entry.second);
            for (auto &entry : m_grid_chunks)
                if (entry.second)
                    entry.second->destroy();
            m_grid_chunks.clear();
        }
        else
        {
            m_streamer.clear(); // saves modified chunks before dropping them
        }
        m_streamer.active(streaming);
    }
    if (m_streamer.active())
    {
        // radius / simplify / persist are safe to tweak live; chunk size/base are fixed at toggle
        m_streamer.radius(m_stream_radius);
        m_streamer.simplify(m_terrain_auto_optimize, m_terrain_simplify_error);
        m_streamer.persist(m_stream_persist);
        ImGui::Text("Resident chunks: %zu  (dir: %s)", m_streamer.resident(), m_streamer.directory().c_str());
    }

    ImGui::Text("Patches: %zu", m_terrains.size());

    // capture: sample the selected model's geometry into an editable patch and remove the original
    if (dynamic_cast<TAnimModel *>(m_node) != nullptr)
    {
        if (ImGui::Button("Capture selected model as terrain"))
            capture_terrain();
    }
    else
    {
        ImGui::TextDisabled("Capture: select a model instance first");
    }

    std::vector<editor_terrain *> const terrains = active_terrains();
    if (!terrains.empty())
    {
        ImGui::Separator();
        if (ImGui::Checkbox("Sculpt mode (LMB raise / Shift = lower)", &m_terrain_sculpt))
            if (m_terrain_sculpt)
                m_chunk_edit = false; // mutually exclusive with chunk editing
        ImGui::SetNextItemWidth(120.0f);
        ImGui::InputFloat("Brush radius", &m_terrain_brush_radius);
        if (m_terrain_brush_radius < 0.5f)
            m_terrain_brush_radius = 0.5f;
        ImGui::SetNextItemWidth(120.0f);
        ImGui::InputFloat("Brush strength", &m_terrain_brush_strength);

        // one-shot nudge of the most recent manual patch at its centre, handy for a quick test
        if (!m_terrains.empty())
        {
            auto &terrain = m_terrains.back();
            glm::dvec3 const c = terrain->centre();
            if (ImGui::Button("Raise centre"))
                terrain->sculpt(c.x, c.z, m_terrain_brush_radius, m_terrain_brush_strength);
            ImGui::SameLine();
            if (ImGui::Button("Lower centre"))
                terrain->sculpt(c.x, c.z, m_terrain_brush_radius, -m_terrain_brush_strength);
        }

        ImGui::Separator();
        ImGui::TextUnformatted("Optimize (mesh simplification, all patches)");
        ImGui::SetNextItemWidth(120.0f);
        ImGui::InputFloat("Flatness tol (m)", &m_terrain_simplify_error);
        if (m_terrain_simplify_error < 0.01f)
            m_terrain_simplify_error = 0.01f;
        ImGui::Checkbox("Auto-optimize after sculpt", &m_terrain_auto_optimize);
        if (ImGui::Button("Optimize all"))
            for (editor_terrain *t : terrains)
                t->optimize(m_terrain_simplify_error);
        ImGui::SameLine();
        if (ImGui::Button("Full-res all"))
            for (editor_terrain *t : terrains)
                t->unoptimize();

        std::size_t tris = 0, full = 0;
        for (editor_terrain *t : terrains)
        {
            tris += t->triangles();
            full += t->full_triangles();
        }
        ImGui::Text("Triangles: %zu / %zu", tris, full);
    }
}

editor_terrain *editor_mode::terrain_at(double X, double Z)
{
    for (auto &terrain : m_terrains)
        if (terrain && terrain->contains(X, Z))
            return terrain.get();
    double const size = chunk_grid_size();
    auto const it = m_grid_chunks.find({static_cast<int>(std::floor(X / size)), static_cast<int>(std::floor(Z / size))});
    if (it != m_grid_chunks.end() && it->second && it->second->contains(X, Z))
        return it->second.get();
    return m_streamer.terrain_at(X, Z);
}

std::vector<editor_terrain *> editor_mode::active_terrains()
{
    std::vector<editor_terrain *> out;
    out.reserve(m_terrains.size() + m_grid_chunks.size() + m_streamer.resident());
    for (auto &terrain : m_terrains)
        if (terrain)
            out.push_back(terrain.get());
    for (auto &entry : m_grid_chunks)
        if (entry.second)
            out.push_back(entry.second.get());
    m_streamer.collect(out);
    return out;
}

void editor_mode::add_grid_chunk(int Cx, int Cz)
{
    std::pair<int, int> const key{Cx, Cz};
    if (m_grid_chunks.count(key))
        return; // already occupied

    double const size = chunk_grid_size();
    int const cells = std::clamp(m_terrain_cells, 1, 256);
    glm::dvec3 const center((Cx + 0.5) * size, static_cast<double>(m_terrain_baseheight), (Cz + 0.5) * size);

    auto terrain = std::make_unique<editor_terrain>();
    if (!terrain->create(center, cells, m_terrain_cellsize, std::string(m_terrain_texture)))
        return;
    if (m_terrain_auto_optimize)
        terrain->optimize(m_terrain_simplify_error);
    m_grid_chunks[key] = std::move(terrain);
}

void editor_mode::remove_grid_chunk(int Cx, int Cz)
{
    auto const it = m_grid_chunks.find({Cx, Cz});
    if (it == m_grid_chunks.end())
        return;
    if (it->second)
        it->second->destroy();
    m_grid_chunks.erase(it);
}

void editor_mode::handle_chunk_edit_click(bool DeleteMode)
{
    // world point under the cursor; must land on existing geometry to give a valid depth
    glm::dvec3 const world = Camera.Pos + GfxRenderer->Mouse_Position();
    double const size = chunk_grid_size();
    int const cx = static_cast<int>(std::floor(world.x / size));
    int const cz = static_cast<int>(std::floor(world.z / size));
    bool const streaming = m_streamer.active();

    if (DeleteMode)
    {
        if (streaming)
            m_streamer.remove_chunk(cx, cz);
        else
            remove_grid_chunk(cx, cz);
        return;
    }

    // if the clicked cell holds a chunk, target the neighbour nearest the clicked edge (the empty
    // side); otherwise fill the clicked cell
    bool const occupied = streaming
                              ? m_streamer.terrain_at(world.x, world.z) != nullptr
                              : m_grid_chunks.count({cx, cz}) > 0;
    int tcx = cx, tcz = cz;
    if (occupied)
    {
        double const lx = world.x - cx * size, lz = world.z - cz * size;
        double const dw = lx, de = size - lx, dn = lz, ds = size - lz;
        double const nearest = std::min({dw, de, dn, ds});
        if (nearest == dw)
            tcx = cx - 1;
        else if (nearest == de)
            tcx = cx + 1;
        else if (nearest == dn)
            tcz = cz - 1;
        else
            tcz = cz + 1;
    }

    if (streaming)
        m_streamer.add_chunk(tcx, tcz);
    else
        add_grid_chunk(tcx, tcz);
}

void editor_mode::create_chunked_terrain()
{
    int const chunks = std::clamp(m_terrain_chunks, 1, 32);
    double const size = chunk_grid_size();

    // snap the field to the global chunk grid (so it aligns with manual/streamed chunks), centred
    // on the camera's chunk
    int const ccx = static_cast<int>(std::floor(Camera.Pos.x / size));
    int const ccz = static_cast<int>(std::floor(Camera.Pos.z / size));
    int const half = chunks / 2;

    int created = 0;
    for (int dz = 0; dz < chunks; ++dz)
        for (int dx = 0; dx < chunks; ++dx)
        {
            int const cx = ccx - half + dx, cz = ccz - half + dz;
            if (!m_grid_chunks.count({cx, cz}))
            {
                add_grid_chunk(cx, cz);
                ++created;
            }
        }

    WriteLog("Editor: created chunked terrain with " + std::to_string(created) + " chunks", logtype::generic);
}

void editor_mode::save_scene_with_terrain()
{
    // commit authored terrain so the scenery streams it on load. if not already streaming, hand the
    // manual grid chunks over to the streamer (same as toggling Stream terrain on)
    if (!m_streamer.active())
    {
        std::string scenery = Global.SceneryFile;
        auto const slash = scenery.find_last_of("/\\");
        if (slash != std::string::npos)
            scenery = scenery.substr(slash + 1);
        auto const dot = scenery.find_last_of('.');
        if (dot != std::string::npos)
            scenery = scenery.substr(0, dot);
        if (scenery.empty())
            scenery = "default";

        m_streamer.directory("editor_terrain/" + scenery);
        m_streamer.configure(m_terrain_cells, m_terrain_cellsize, m_stream_radius, m_terrain_baseheight,
                             std::string(m_terrain_texture));
        m_streamer.simplify(m_terrain_auto_optimize, m_terrain_simplify_error);
        m_streamer.persist(true);

        for (auto &entry : m_grid_chunks)
            if (entry.second)
                m_streamer.save_chunk(entry.first.first, entry.first.second, *entry.second);
        for (auto &entry : m_grid_chunks)
            if (entry.second)
                entry.second->destroy();
        m_grid_chunks.clear();
        m_streamer.active(true);
    }

    m_streamer.flush(); // save resident edited chunks to disk

    // export scenery; the exported .scm now carries an `editorterrain` directive (streamer is active)
    simulation::State.export_as_text(Global.SceneryFile);
    WriteLog("Editor: saved scene + terrain", logtype::generic);
}

void editor_mode::handle_terrain_sculpt(double Deltatime)
{
    // world point under the cursor (Mouse_Position is camera-relative, like the brush placement uses)
    glm::dvec3 const world = Camera.Pos + GfxRenderer->Mouse_Position();
    // only sculpt when the cursor is actually over terrain (avoids editing on a stale depth read)
    if (terrain_at(world.x, world.z) == nullptr)
        return;

    double const rate = m_terrain_brush_strength * Deltatime; // metres applied this frame
    double const signedrate = Global.shiftState ? -rate : rate;
    // apply to every chunk the brush touches; each patch clips to its own bounds, so a stroke
    // crossing a chunk boundary edits both and shared-edge vertices stay in sync
    for (editor_terrain *terrain : active_terrains())
        terrain->sculpt(world.x, world.z, m_terrain_brush_radius, signedrate);
}

void editor_mode::capture_terrain()
{
    TAnimModel *model = dynamic_cast<TAnimModel *>(m_node);
    if (model == nullptr || model->pModel == nullptr)
    {
        WriteLog("Editor: select a model instance to capture as terrain", logtype::generic);
        return;
    }

    // instance world transform, matching the renderer: translate * rotateY * rotateX * rotateZ * scale
    glm::dmat4 rootm(1.0);
    rootm = glm::translate(rootm, model->location());
    glm::vec3 const angles = model->Angles();
    if (angles.y != 0.0f) rootm = glm::rotate(rootm, glm::radians(static_cast<double>(angles.y)), glm::dvec3(0.0, 1.0, 0.0));
    if (angles.x != 0.0f) rootm = glm::rotate(rootm, glm::radians(static_cast<double>(angles.x)), glm::dvec3(1.0, 0.0, 0.0));
    if (angles.z != 0.0f) rootm = glm::rotate(rootm, glm::radians(static_cast<double>(angles.z)), glm::dvec3(0.0, 0.0, 1.0));
    glm::vec3 const scale = model->Scale();
    rootm = glm::scale(rootm, glm::dvec3(scale));

    std::vector<world_triangle> tris;
    gather_submodel_triangles(model->pModel->Root, rootm, tris);
    if (tris.empty())
    {
        WriteLog("Editor: selected model has no readable geometry to capture", logtype::generic);
        return;
    }

    // horizontal bounds of the captured geometry
    glm::dvec3 lo(std::numeric_limits<double>::max());
    glm::dvec3 hi(-std::numeric_limits<double>::max());
    for (auto const &t : tris)
        for (auto const &p : t)
        {
            lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
            hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
        }

    glm::dvec3 const center((lo.x + hi.x) * 0.5, lo.y, (lo.z + hi.z) * 0.5);
    double const extent = std::max(hi.x - lo.x, hi.z - lo.z);
    int const cells = std::max(1, m_terrain_cells);
    float const cellsize = static_cast<float>(std::max(0.1, extent / cells));

    // sampler: highest captured triangle at (x,z)
    auto const sampler = [&tris](double X, double Z, double &OutY) -> bool {
        double best = -std::numeric_limits<double>::max();
        bool found = false;
        for (auto const &t : tris)
        {
            double const minx = std::min({t[0].x, t[1].x, t[2].x});
            double const maxx = std::max({t[0].x, t[1].x, t[2].x});
            double const minz = std::min({t[0].z, t[1].z, t[2].z});
            double const maxz = std::max({t[0].z, t[1].z, t[2].z});
            if (X < minx || X > maxx || Z < minz || Z > maxz)
                continue;
            double y;
            if (triangle_height_at(t[0], t[1], t[2], X, Z, y) && (!found || y > best))
            {
                best = y;
                found = true;
            }
        }
        if (found)
            OutY = best;
        return found;
    };

    auto terrain = std::make_unique<editor_terrain>();
    if (!terrain->create(center, cells, cellsize, std::string(m_terrain_texture), sampler))
    {
        WriteLog("Editor: terrain capture failed", logtype::generic);
        return;
    }
    m_terrains.push_back(std::move(terrain));

    // remove the original instance (recorded as a deletion so it can be undone)
    std::string as_text;
    model->export_as_text(as_text);
    push_snapshot(model, EditorSnapshot::Action::Delete, as_text);
    nullify_history_pointers(model);
    remove_from_hierarchy(model);
    m_node = nullptr;
    m_dragging = false;
    ui()->set_node(nullptr);
    simulation::State.delete_model(model);
}

// collects a track's segment(s): one for normal tracks, both paths for a switch/crossing
static void collect_track_segments( TTrack *track, std::vector<std::shared_ptr<TSegment>> &out )
{
    if( track->SwitchExtension ) {
        if( track->SwitchExtension->Segments[ 0 ] ) { out.push_back( track->SwitchExtension->Segments[ 0 ] ); }
        if( track->SwitchExtension->Segments[ 1 ] ) { out.push_back( track->SwitchExtension->Segments[ 1 ] ); }
    }
    else if( track->Segment ) {
        out.push_back( track->Segment );
    }
}

void editor_mode::render_track_overlay()
{
    // same camera-relative view/projection the gizmo uses, so the overlay lines up with the render
    ImGuiIO const &io = ImGui::GetIO();
    glm::mat4 const view = GfxRenderer->Camera_View_Matrix();
    glm::dvec3 const camerapos = GfxRenderer->Camera_Position();
    float const fovy = glm::radians( Global.FieldOfView / Global.ZoomFactor );
    float const aspect = io.DisplaySize.y > 0.0f ? io.DisplaySize.x / io.DisplaySize.y : 1.0f;
    glm::mat4 const projection = glm::perspective( fovy, aspect, 0.1f, 10000.0f );

    auto const project = [&]( glm::dvec3 const &world, ImVec2 &out ) -> bool {
        glm::vec4 const clip = projection * view * glm::vec4( glm::vec3( world - camerapos ), 1.0f );
        if( clip.w <= 1e-4f ) { return false; } // behind the camera
        glm::vec2 const ndc = glm::vec2( clip ) / clip.w;
        out = ImVec2( ( ndc.x * 0.5f + 0.5f ) * io.DisplaySize.x,
                      ( 1.0f - ( ndc.y * 0.5f + 0.5f ) ) * io.DisplaySize.y );
        return true;
    };

    ImDrawList *dl = ImGui::GetForegroundDrawList();
    ImFont *const font = ImGui::GetFont();
    float const fontsize = 22.0f;
    ImU32 const endcol = IM_COL32( 60, 220, 255, 255 );   // track endpoints
    ImU32 const handlecol = IM_COL32( 255, 200, 0, 255 ); // control-vector lines
    ImU32 const dotcol = IM_COL32( 255, 120, 0, 255 );    // control points

    std::vector<std::shared_ptr<TSegment>> segs;
    for( auto *path : simulation::Paths.sequence() ) {
        if( path == nullptr ) { continue; }
        if( glm::length2( path->location() - camerapos ) > sq( 700.0 ) ) { continue; } // cull distant

        segs.clear();
        collect_track_segments( path, segs );
        glm::dvec3 labelpos{ 0.0 };
        bool haslabel = false;

        for( auto const &seg : segs ) {
            if( !seg ) { continue; }
            glm::dvec3 const p1{ seg->FastGetPoint_0() };
            glm::dvec3 const p2{ seg->FastGetPoint_1() };
            ImVec2 a, b;
            // endpoints
            if( project( p1, a ) ) { dl->AddCircleFilled( a, 4.0f, endcol ); }
            if( project( p2, b ) ) { dl->AddCircleFilled( b, 4.0f, endcol ); }
            // control-vector handles (for curved segments)
            if( seg->bCurve ) {
                ImVec2 c;
                if( project( p1, a ) && project( p1 + seg->GetDirection1(), c ) ) { dl->AddLine( a, c, handlecol, 2.0f ); dl->AddCircleFilled( c, 3.5f, dotcol ); }
                if( project( p2, b ) && project( p2 + seg->GetDirection2(), c ) ) { dl->AddLine( b, c, handlecol, 2.0f ); dl->AddCircleFilled( c, 3.5f, dotcol ); }
            }
            if( !haslabel ) { labelpos = ( p1 + p2 ) * 0.5; haslabel = true; }
        }

        if( haslabel ) {
            ImVec2 s;
            if( project( labelpos, s ) ) {
                auto const it = m_track_labels.find( path );
                std::string const label = ( it != m_track_labels.end() ) ? it->second : ( "R=" + std::to_string( (int)path->fRadius ) );
                float const ty = s.y - fontsize * 0.5f;
                dl->AddText( font, fontsize, ImVec2( s.x + 8.0f, ty + 1.0f ), IM_COL32( 0, 0, 0, 220 ), label.c_str() ); // shadow
                dl->AddText( font, fontsize, ImVec2( s.x + 7.0f, ty ), IM_COL32( 255, 240, 60, 255 ), label.c_str() );
            }
        }
    }

    // niweleta joints: editable (green) squares; KP-locked ends drawn grey
    for( auto const &ch : m_chains ) {
        for( size_t j = 0; j < ch.joints.size(); ++j ) {
            ImVec2 s;
            if( !project( ch.joints[ j ] + glm::dvec3{ 0.0, 0.25, 0.0 }, s ) ) { continue; }
            bool const locked = ( j == 0 && ch.anchor != nullptr )
                             || ( j > 0 && ch.elements[ j - 1 ].type == track_panel::TRANSITION );
            ImU32 const col = locked ? IM_COL32( 150, 150, 150, 255 ) : IM_COL32( 40, 255, 40, 255 );
            dl->AddRectFilled( ImVec2( s.x - 5, s.y - 5 ), ImVec2( s.x + 5, s.y + 5 ), col );
        }
    }
    // drag preview: marker at the cursor and a guide line from the grabbed point
    if( m_dragactive ) {
        ImVec2 a, b;
        bool const hasa = project( m_drag_from, a );
        bool const hasb = project( m_dragpos + glm::dvec3{ 0.0, 0.2, 0.0 }, b );
        if( hasa && hasb ) { dl->AddLine( a, b, IM_COL32( 255, 60, 60, 200 ), 2.0f ); }
        if( hasb ) { dl->AddRectFilled( ImVec2( b.x - 6, b.y - 6 ), ImVec2( b.x + 6, b.y + 6 ), IM_COL32( 255, 60, 60, 255 ) ); }
    }
}

void editor_mode::render_gizmo()
{
    // the transform gizmo is suppressed while editing terrain, so the brush/chunk tool owns the mouse
    if (!m_gizmo_enabled || m_terrain_sculpt || m_chunk_edit)
    {
        m_gizmo_using = false;
        return;
    }

    // compact control window: lets the user pick the transform mode without keyboard shortcuts
    ImGui::Begin("Gizmo", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing);
    int op = static_cast<int>(m_gizmo_op);
    ImGui::RadioButton("Translate (Q)", &op, static_cast<int>(gizmo_operation::translate));
    ImGui::SameLine();
    ImGui::RadioButton("Rotate (W)", &op, static_cast<int>(gizmo_operation::rotate));
    ImGui::SameLine();
    ImGui::RadioButton("Scale (E)", &op, static_cast<int>(gizmo_operation::scale));
    m_gizmo_op = static_cast<gizmo_operation>(op);

    if (m_gizmo_op != gizmo_operation::scale) // ImGuizmo always scales in local space
        ImGui::Checkbox("Local space (R)", &m_gizmo_local);
    if (m_gizmo_op == gizmo_operation::translate)
    {
        ImGui::SetNextItemWidth(120.0f);
        ImGui::InputFloat("Snap (hold Ctrl)", &m_gizmo_snap);
        if (m_gizmo_snap < 0.0f)
            m_gizmo_snap = 0.0f;
    }
    if (!m_node)
        ImGui::TextDisabled("No node selected");
    ImGui::End();

    if (!m_node)
    {
        m_gizmo_using = false;
        return;
    }

    ImGuizmo::BeginFrame();
    ImGuizmo::SetOrthographic(false);

    ImGuiIO const &io = ImGui::GetIO();
    ImGuizmo::SetRect(0.0f, 0.0f, io.DisplaySize.x, io.DisplaySize.y);

    // the view matrix comes from the most recent color pass and is camera-relative
    // (rotation only), so the gizmo is positioned relative to the camera as well.
    glm::mat4 const view = GfxRenderer->Camera_View_Matrix();
    glm::dvec3 const camerapos = GfxRenderer->Camera_Position();

    // the engine's own projection bakes in reverse-Z (and screen orientation), which ImGuizmo
    // doesn't expect; rebuild a clean, standard perspective that matches the rendered view.
    // for the main viewport the engine uses a symmetric frustum with this exact fov/aspect.
    float const fovy = glm::radians(Global.FieldOfView / Global.ZoomFactor);
    float const aspect = io.DisplaySize.y > 0.0f ? io.DisplaySize.x / io.DisplaySize.y : 1.0f;
    glm::mat4 const projection = glm::perspective(fovy, aspect, 0.1f, 10000.0f);

    // rotation/scale are only meaningful for instanced models; other node types translate only
    TAnimModel *model = dynamic_cast<TAnimModel *>(m_node);

    glm::vec3 const relativepos = glm::vec3(m_node->location() - camerapos);
    glm::vec3 const angles = model ? model->Angles() : glm::vec3(0.0f);
    glm::vec3 const scalevec = model ? model->Scale() : glm::vec3(1.0f);

    // build the gizmo model matrix from the node's current translation + rotation + scale
    float const translation[3] = {relativepos.x, relativepos.y, relativepos.z};
    float const rotation[3] = {angles.x, angles.y, angles.z};
    float const scale[3] = {scalevec.x, scalevec.y, scalevec.z};
    glm::mat4 matrix(1.0f);
    ImGuizmo::RecomposeMatrixFromComponents(translation, rotation, scale, glm::value_ptr(matrix));

    // map the editor's transform mode onto ImGuizmo; fall back to translate for non-models
    ImGuizmo::OPERATION operation = ImGuizmo::TRANSLATE;
    EditorSnapshot::Action action = EditorSnapshot::Action::Move;
    if (model && m_gizmo_op == gizmo_operation::rotate)
    {
        operation = ImGuizmo::ROTATE;
        action = EditorSnapshot::Action::Rotate;
    }
    else if (model && m_gizmo_op == gizmo_operation::scale)
    {
        operation = ImGuizmo::SCALE;
        action = EditorSnapshot::Action::Scale;
    }
    ImGuizmo::MODE const mode = m_gizmo_local ? ImGuizmo::LOCAL : ImGuizmo::WORLD;

    // optional snapping while Ctrl is held (metres / degrees / scale factor depending on mode)
    glm::vec3 snapvalue(0.0f);
    if (operation == ImGuizmo::TRANSLATE)
        snapvalue = glm::vec3(m_gizmo_snap);
    else if (operation == ImGuizmo::ROTATE)
        snapvalue = glm::vec3(5.0f);
    else
        snapvalue = glm::vec3(0.1f);
    float const *snap = Global.ctrlState && snapvalue.x > 0.0f ? glm::value_ptr(snapvalue) : nullptr;

    ImGuizmo::Manipulate(glm::value_ptr(view), glm::value_ptr(projection),
                         operation, mode, glm::value_ptr(matrix), nullptr, snap);

    if (ImGuizmo::IsUsing())
    {
        // record a single undo snapshot at the start of the drag
        if (!m_gizmo_using)
        {
            push_snapshot(m_node, action);
            m_gizmo_using = true;
        }

        float newtranslation[3], newrotation[3], newscale[3];
        ImGuizmo::DecomposeMatrixToComponents(glm::value_ptr(matrix), newtranslation, newrotation, newscale);

        if (operation == ImGuizmo::ROTATE && model)
        {
            // apply the rotation delta relative to the model's current orientation
            glm::vec3 const newangles(newrotation[0], newrotation[1], newrotation[2]);
            m_editor.rotate(model, newangles - model->Angles(), 0.0f);
        }
        else if (operation == ImGuizmo::SCALE && model)
        {
            model->Scale(glm::vec3(newscale[0], newscale[1], newscale[2]));
        }
        else
        {
            glm::dvec3 const newworldpos = camerapos + glm::dvec3(newtranslation[0], newtranslation[1], newtranslation[2]);
            // pass Snaptoground == true so the gizmo's Y component is applied (free 3D move)
            m_editor.translate(m_node, newworldpos, true);
        }
    }
    else
    {
        m_gizmo_using = false;
    }
}

void editor_mode::update_camera(double const Deltatime)
{
    Camera.Update();

    // focus animation runs after Camera.Update() so it overrides any residual velocity/rotation;
    // it smoothly drives both position and orientation toward the framed object
    if (m_focus_active)
    {
        m_focus_time += Deltatime;
        double t = m_focus_duration > 0.0 ? m_focus_time / m_focus_duration : 1.0;
        if (t >= 1.0)
            t = 1.0;
        // smoothstep easing
        float const s = static_cast<float>(t * t * (3.0 - 2.0 * t));

        Camera.Pos = glm::mix(m_focus_start_pos, m_focus_target_pos, static_cast<double>(s));

        // interpolate angles, taking the shortest path around the yaw wrap-around
        constexpr float TWO_PI = 6.283185307179586f;
        float const dyaw = std::remainder(m_focus_target_angle.y - m_focus_start_angle.y, TWO_PI);
        Camera.Angle.x = m_focus_start_angle.x + (m_focus_target_angle.x - m_focus_start_angle.x) * s;
        Camera.Angle.y = m_focus_start_angle.y + dyaw * s;
        Camera.Angle.z = m_focus_start_angle.z + (m_focus_target_angle.z - m_focus_start_angle.z) * s;

        // suppress any residual fly velocity so it doesn't fight the animation
        Camera.Velocity = glm::dvec3(0.0);

        if (t >= 1.0)
            m_focus_active = false;
    }

    // reset window state (will be set again if UI requires it)
    Global.CabWindowOpen = false;

    // publish camera back to global copy
    Global.pCamera = Camera;
}

void editor_mode::enter()
{
    m_statebackup = {Global.pCamera, FreeFlyModeFlag, Global.ControlPicking};

    Camera = Global.pCamera;

    if (!FreeFlyModeFlag)
    {
        auto const *vehicle = Camera.m_owner;
        if (vehicle)
        {
            const int cab = vehicle->MoverParameters->CabOccupied == 0 ? 1 : vehicle->MoverParameters->CabOccupied;
            const glm::dvec3 left = vehicle->VectorLeft() * (double)cab;
            Camera.Pos = glm::dvec3(Camera.Pos.x, vehicle->GetPosition().y, Camera.Pos.z) + left * vehicle->GetWidth() + glm::dvec3(1.25f * left.x, 1.6f, 1.25f * left.z);
            Camera.m_owner = nullptr;
            Camera.LookAt = vehicle->GetPosition();
            Camera.RaLook(); // single camera reposition
            FreeFlyModeFlag = true;
        }
    }

    Global.ControlPicking = true;
    EditorModeFlag = true;

    Application.set_cursor(GLFW_CURSOR_NORMAL);
}

void editor_mode::exit()
{
    EditorModeFlag = false;
    Global.ControlPicking = m_statebackup.picking;
    FreeFlyModeFlag = m_statebackup.freefly;
    Global.pCamera = m_statebackup.camera;

    g_redo.clear();
    m_history.clear();

    // drop selection so a stale/dangling node pointer isn't used on the next editor session
    m_node = nullptr;
    m_gizmo_using = false;
    ui()->set_node(nullptr);

    Application.set_cursor(Global.ControlPicking ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);

    if (!Global.ControlPicking)
    {
        Application.set_cursor_pos(0, 0);
    }
}

void editor_mode::on_key(int const Key, int const Scancode, int const Action, int const Mods)
{
#ifndef __unix__
    Global.shiftState = Mods & GLFW_MOD_SHIFT ? true : false;
    Global.ctrlState = Mods & GLFW_MOD_CONTROL ? true : false;
    Global.altState = Mods & GLFW_MOD_ALT ? true : false;
#endif
    bool anyModifier = Mods & (GLFW_MOD_SHIFT | GLFW_MOD_CONTROL | GLFW_MOD_ALT);

    // first give UI a chance to handle the key
    if (!anyModifier && m_userinterface->on_key(Key, Action))
        return;

    // gizmo transform shortcuts (Q/W/E/R) — only when the camera isn't being flown (RMB up).
    // handled before the camera keyboard step because Q/W/E are also the fly-mode movement keys,
    // which would otherwise consume them.
    if (!anyModifier && is_press(Action)
        && m_input.mouse.button(GLFW_MOUSE_BUTTON_RIGHT) != GLFW_PRESS)
    {
        bool handled = true;
        switch (Key)
        {
        case GLFW_KEY_Q: m_gizmo_op = gizmo_operation::translate; break;
        case GLFW_KEY_W: m_gizmo_op = gizmo_operation::rotate; break;
        case GLFW_KEY_E: m_gizmo_op = gizmo_operation::scale; break;
        case GLFW_KEY_R: m_gizmo_local = !m_gizmo_local; break;
        case GLFW_KEY_T: create_straight_track_ahead(); break; // initial track tool: straight track ahead of camera
        default: handled = false; break;
        }
        if (handled)
            return;
    }

    // then internal input handling
    if (m_input.keyboard.key(Key, Action))
        return;

    if (Action == GLFW_RELEASE)
        return;

    // shortcuts: undo/redo
    if (Global.ctrlState && Key == GLFW_KEY_Z && is_press(Action))
    {
        undo_last();
        return;
    }
    if (Global.ctrlState && Key == GLFW_KEY_Y && is_press(Action))
    {
        redo_last();
        return;
    }

    // legacy hardcoded keyboard commands
    switch (Key)
    {
    case GLFW_KEY_F11:
        if (Action != GLFW_PRESS)
            break;

        if (!Global.ctrlState && !Global.shiftState)
        {
            Application.pop_mode();
        }
        else if (Global.ctrlState && Global.shiftState)
        {
            simulation::State.export_as_text(Global.SceneryFile);
        }
        break;

    case GLFW_KEY_F12:
        if (Global.ctrlState && Global.shiftState && is_press(Action))
        {
            DebugModeFlag = !DebugModeFlag;
        }
        break;

    case GLFW_KEY_DELETE:
        if (is_press(Action))
        {
            TAnimModel *model = dynamic_cast<TAnimModel *>(m_node);
            if (model)
            {
                // record deletion for undo (serialize full node)
                std::string as_text;
                
                model->export_as_text(as_text);
                std::string debug = "Deleting node: " + as_text + "\nSerialized data:\n";
                push_snapshot(model, EditorSnapshot::Action::Delete, as_text);
                WriteLog(debug, logtype::generic);

                // clear history pointers referencing this model before actually deleting it
                nullify_history_pointers(model);
                remove_from_hierarchy(model);

                m_node = nullptr;
                m_dragging = false;
                ui()->set_node(nullptr);
                simulation::State.delete_model(model);
            }
            else if( TTrack *track = dynamic_cast<TTrack *>( m_node ) )
            {
                // keep the niweleta model in sync: drop the element owning this track, detach
                // chains anchored to a deleted switch
                for( size_t c = 0; c < m_chains.size(); ++c ) {
                    auto &ch = m_chains[ c ];
                    if( ch.anchor == track ) { ch.anchor = nullptr; }
                    for( size_t e = 0; e < ch.elements.size(); ++e ) {
                        auto &trs = ch.elements[ e ].tracks;
                        auto it = std::find( trs.begin(), trs.end(), track );
                        if( it != trs.end() ) {
                            trs.erase( it ); // this piece is going away
                            ch.elements.erase( ch.elements.begin() + e );
                            regenerate_chain( (int)c );
                            break;
                        }
                    }
                }
                m_switch_meta.erase( track );
                std::string const trackname = track->name();
                delete_track( track );
                m_dragging = false;
                WriteLog( "editor: deleted track \"" + trackname + "\"" );
            }
        }
        break;

    case GLFW_KEY_F:
        if (is_press(Action))
        {
            if(!m_node)
                break;

            // start smooth focus camera on selected node
            start_focus(m_node, 0.6);
        }
        break;

    case GLFW_KEY_END:
        if (is_press(Action) && m_node)
        {
            // Unreal-style "snap to floor": drop the selected node onto the surface below it.
            // works against triangle geometry (shape_node terrain / opaque shapes); once a proper
            // editable terrain mesh exists, dropping onto it works without further changes here.
            snap_to_ground(m_node);
        }
        break;

    default:
        break;
    }
}

void editor_mode::on_cursor_pos(double const Horizontal, double const Vertical)
{
    // object transforms are handled by the gizmo now; here we only forward the cursor to the
    // mouse input, which rotates the camera while the right mouse button is held (panning mode)
    m_input.mouse.position(Horizontal, Vertical);
}

void editor_mode::on_mouse_button(int const Button, int const Action, int const Mods)
{
    // UI first
    if (m_userinterface->on_mouse_button(Button, Action))
    {
        m_input.mouse.button(Button, Action);
        return;
    }

    // in chunk-edit mode the left button adds a neighbouring chunk (Shift = delete the clicked one)
    if (m_chunk_edit && Button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (is_press(Action))
            handle_chunk_edit_click(Global.shiftState);
        m_input.mouse.button(Button, Action);
        return;
    }

    // in terrain sculpt mode the left button paints the terrain instead of picking nodes
    if (m_terrain_sculpt && Button == GLFW_MOUSE_BUTTON_LEFT)
    {
        mouseHold = is_press(Action);
        m_input.mouse.button(Button, Action);
        return;
    }

    if (Button == GLFW_MOUSE_BUTTON_LEFT)
    {
        // finish a drag: apply the edit (element parameters / switch position) and re-march
        if( !is_press(Action) && m_dragactive )
        {
            m_dragactive = false;
            glm::dvec3 const target = cursor_ground_point();
            auto const hnorm = []( glm::dvec3 v ) {
                v.y = 0.0;
                double const l = glm::length( v );
                return ( l > 1e-9 ) ? v / l : glm::dvec3{ 0.0, 0.0, -1.0 };
            };

            if( m_switchdrag != nullptr ) {
                // translate the whole switch, then re-march chains anchored to it
                auto const metait = m_switch_meta.find( m_switchdrag );
                if( metait != m_switch_meta.end() ) {
                    switch_meta meta = metait->second;
                    glm::dvec3 const delta{ target.x - m_drag_from.x, 0.0, target.z - m_drag_from.z };
                    meta.entry += delta; meta.straightend += delta; meta.divend += delta;
                    TTrack *old = m_switchdrag;
                    TTrack *fresh = commit_switch( meta.entry, meta.straightend, meta.divcv1, meta.divcv2, meta.divend, meta.radius, meta.length );
                    if( fresh != nullptr ) {
                        m_switch_meta.erase( old );
                        delete_track( old );
                        for( size_t c = 0; c < m_chains.size(); ++c ) {
                            if( m_chains[ c ].anchor == old ) {
                                m_chains[ c ].anchor = fresh;
                                regenerate_chain( (int)c );
                            }
                        }
                    }
                }
                m_switchdrag = nullptr;
            }
            else if( m_chaindrag_chain >= 0 && m_chaindrag_chain < (int)m_chains.size() ) {
                auto &ch = m_chains[ m_chaindrag_chain ];
                int const j = m_chaindrag_joint;
                if( j == 0 ) {
                    // free-chain origin: translate the whole chain
                    if( ch.anchor == nullptr ) {
                        ch.origin = glm::dvec3{ target.x, ch.origin.y, target.z };
                    }
                }
                else if( j - 1 < (int)ch.elements.size() ) {
                    // march up to the element start, then re-parametrize the element to reach
                    // the dragged point while KEEPING ITS TYPE
                    glm::dvec3 pos = ch.origin, dir = hnorm( ch.direction );
                    if( ch.anchor != nullptr ) { pos = ch.joints.front(); }
                    for( int e = 0; e < j - 1; ++e ) {
                        march_element( ch.elements[ e ], pos, dir, false );
                    }
                    auto &el = ch.elements[ j - 1 ];
                    glm::dvec3 const chordv{ target.x - pos.x, 0.0, target.z - pos.z };
                    double const chord = glm::length( chordv );
                    if( chord > 1.0 ) {
                        if( el.type == track_panel::STRAIGHT ) {
                            // a run of consecutive straights behaves like one direction vector:
                            // dragging its end rotates the run; the preceding arc adapts its
                            // sweep to the new line, a KP predecessor locks the direction
                            int const eidx = j - 1;
                            int runstart = eidx;
                            while( runstart > 0 && ch.elements[ runstart - 1 ].type == track_panel::STRAIGHT ) { --runstart; }
                            glm::dvec3 rpos = ch.origin, rdir = hnorm( ch.direction );
                            for( int e = 0; e < runstart; ++e ) { march_element( ch.elements[ e ], rpos, rdir, false ); }
                            glm::dvec3 const newdir = hnorm( glm::dvec3{ target.x - rpos.x, 0.0, target.z - rpos.z } );
                            // when the run is followed by [ARC][straight...], solve the arc as a
                            // fillet between the rotated line and the DOWNSTREAM LINE, which stays
                            // fixed (vector untouched); curve parameters (radius) are preserved
                            int runend = eidx;
                            while( runend + 1 < (int)ch.elements.size() && ch.elements[ runend + 1 ].type == track_panel::STRAIGHT ) { ++runend; }
                            bool const hasfillet =
                                ( runend + 2 < (int)ch.elements.size()
                               && ch.elements[ runend + 1 ].type == track_panel::ARC
                               && ch.elements[ runend + 2 ].type == track_panel::STRAIGHT );
                            glm::dvec3 q{ 0.0 }, d2{ 0.0 }, ffar{ 0.0 };
                            if( hasfillet ) {
                                // downstream line captured from the PRE-EDIT march (it must not move)
                                glm::dvec3 cpos = ch.origin, cdir2 = hnorm( ch.direction );
                                for( int e = 0; e <= runend + 1; ++e ) { march_element( ch.elements[ e ], cpos, cdir2, false ); }
                                q = cpos; d2 = cdir2;
                                ffar = q + d2 * ch.elements[ runend + 2 ].length;
                            }

                            if( runstart == 0 ) {
                                if( ch.anchor == nullptr ) { ch.direction = newdir; }
                            }
                            else if( ch.elements[ runstart - 1 ].type == track_panel::ARC ) {
                                auto &arc = ch.elements[ runstart - 1 ];
                                glm::dvec3 apos = ch.origin, adir = hnorm( ch.direction );
                                for( int e = 0; e < runstart - 1; ++e ) { march_element( ch.elements[ e ], apos, adir, false ); }
                                double const cosang2 = std::clamp( glm::dot( adir, newdir ), -1.0, 1.0 );
                                double const theta2 = std::acos( cosang2 );
                                if( theta2 > 1e-3 ) {
                                    arc.left = ( adir.x * newdir.z - adir.z * newdir.x ) < 0.0;
                                    arc.length = std::max( 1.0, arc.radius ) * theta2;
                                }
                            }
                            // re-march to this element and set its length toward the drag point
                            glm::dvec3 pos2 = ch.origin, dir2 = hnorm( ch.direction );
                            for( int e = 0; e < eidx; ++e ) { march_element( ch.elements[ e ], pos2, dir2, false ); }
                            el.length = std::max( 1.0, glm::dot( glm::dvec3{ target.x - pos2.x, 0.0, target.z - pos2.z }, dir2 ) );

                            if( hasfillet ) {
                                auto &arc = ch.elements[ runend + 1 ];
                                auto &next = ch.elements[ runend + 2 ];
                                // line 1: run start (pivot) + new direction; line 2: fixed q/d2
                                glm::dvec3 apos = ch.origin, adir = hnorm( ch.direction );
                                for( int e = 0; e < runstart; ++e ) { march_element( ch.elements[ e ], apos, adir, false ); }
                                double const det = d2.x * adir.z - adir.x * d2.z;
                                double const cosang3 = std::clamp( glm::dot( adir, d2 ), -1.0, 1.0 );
                                double const theta3 = std::acos( cosang3 );
                                if( std::abs( det ) > 1e-6 && theta3 > 1e-3 ) {
                                    // corner = intersection of the two lines
                                    glm::dvec3 const w = q - apos;
                                    double const denom = adir.x * ( -d2.z ) - ( -d2.x ) * adir.z;
                                    double const a = ( w.x * ( -d2.z ) - ( -d2.x ) * w.z ) / denom;
                                    glm::dvec3 const corner = apos + adir * a;
                                    double const R = std::max( 1.0, arc.radius );
                                    double const t = R * std::tan( theta3 * 0.5 );
                                    // upstream run: total length up to the tangent point
                                    double runothers = 0.0;
                                    for( int e = runstart; e < runend; ++e ) { runothers += ch.elements[ e ].length; }
                                    ch.elements[ runend ].length = std::max( 1.0, a - t - runothers );
                                    // the arc keeps its radius; only the sweep adapts
                                    arc.left = ( adir.x * d2.z - adir.z * d2.x ) < 0.0;
                                    arc.length = R * theta3;
                                    // downstream straight: same line, far end pinned
                                    glm::dvec3 const t2 = corner + d2 * t;
                                    next.length = std::max( 1.0, glm::dot( ffar - t2, d2 ) );
                                }
                            }
                        }
                        else if( el.type == track_panel::ARC ) {
                            // re-fit the arc through the point, tangent to the march direction
                            glm::dvec3 const cdir = hnorm( chordv );
                            double const cosang = std::clamp( glm::dot( cdir, dir ), -1.0, 1.0 );
                            double const ang = std::acos( cosang );
                            if( ang < 1e-3 ) {
                                el.length = chord;
                            }
                            else {
                                double const R = chord / ( 2.0 * std::sin( ang ) );
                                el.radius = R;
                                el.length = R * 2.0 * ang;
                                el.left = ( dir.x * cdir.z - dir.z * cdir.x ) < 0.0;
                            }
                        }
                        // TRANSITION never lands here (locked at pick time)
                    }
                }
                regenerate_chain( m_chaindrag_chain );
                m_chaindrag_chain = -1;
                m_chaindrag_joint = -1;
            }
            m_input.mouse.button(Button, Action);
            return;
        }

        auto const mode = ui()->mode();
        auto const rotation_mode = ui()->rot_mode();
        auto const fixed_rotation_value = ui()->rot_val();

        if (is_press(Action))
        {
            mouseHold = true;
            m_node = nullptr;

            // niweleta building: each click appends a typed element (panel parameters) to the
            // active chain. The first click sets the origin - anchored to a switch outlet when
            // clicked near one, or free (ground point + camera heading). A switch placed at the
            // chain end is a branch point: it ends the chain and new chains can anchor to it.
            if (ui()->track_place_active())
            {
                ImGuiIO const &io = ImGui::GetIO();

                if( ui()->track_type() == track_panel::SWITCH ) {
                    glm::dvec3 start, dir;
                    if( m_active_chain >= 0 && m_active_chain < (int)m_chains.size() && !m_chains[ m_active_chain ].joints.empty() ) {
                        // branch point at the end of the active chain, tangent to it
                        start = m_chains[ m_active_chain ].joints.back();
                        dir = m_chains[ m_active_chain ].endtangent;
                        m_active_chain = -1; // the chain ends at the branch
                    }
                    else {
                        TTrack *seed = nullptr; int seedend = -1;
                        if( pick_track_endpoint( io.MousePos.x, io.MousePos.y, true, seed, seedend, start, dir ) ) {
                            start.y -= 0.18;
                        }
                        else {
                            start = cursor_ground_point();
                            glm::mat3 const rot = glm::mat3( GfxRenderer->Camera_View_Matrix() );
                            glm::vec3 const fwd = glm::transpose( rot ) * glm::vec3( 0.0f, 0.0f, -1.0f );
                            dir = glm::dvec3{ fwd.x, 0.0, fwd.z };
                            double const dl = glm::length( dir );
                            dir = ( dl > 1e-6 ) ? dir / dl : glm::dvec3{ 0.0, 0.0, -1.0 };
                        }
                    }
                    create_track_at( start, dir ); // builds the switch (records its meta)
                    m_input.mouse.button(Button, Action);
                    return;
                }

                if( m_active_chain < 0 || m_active_chain >= (int)m_chains.size() ) {
                    // start a new chain: anchored at a clicked switch outlet, continuing from a
                    // clicked track end, or free at the ground point
                    track_chain ch;
                    TTrack *seed = nullptr; int seedend = -1;
                    glm::dvec3 point, outward;
                    bool picked = pick_track_endpoint( io.MousePos.x, io.MousePos.y, true, seed, seedend, point, outward );
                    if( !picked ) {
                        // world-space fallback: snap to the nearest endpoint within 8 m of the click
                        glm::dvec3 const ground = cursor_ground_point();
                        double best = 8.0 * 8.0;
                        std::vector<std::shared_ptr<TSegment>> fsegs;
                        for( auto *path : simulation::Paths.sequence() ) {
                            if( path == nullptr ) { continue; }
                            fsegs.clear();
                            collect_track_segments( path, fsegs );
                            for( size_t s = 0; s < fsegs.size(); ++s ) {
                                if( !fsegs[ s ] ) { continue; }
                                glm::dvec3 const pts[ 2 ] = { glm::dvec3{ fsegs[ s ]->FastGetPoint_0() }, glm::dvec3{ fsegs[ s ]->FastGetPoint_1() } };
                                for( int e = 0; e < 2; ++e ) {
                                    double const dx = pts[ e ].x - ground.x, dz = pts[ e ].z - ground.z;
                                    if( dx * dx + dz * dz < best ) {
                                        best = dx * dx + dz * dz;
                                        seed = path;
                                        seedend = (int)s * 2 + e;
                                        point = pts[ e ];
                                        glm::dvec3 o = ( e == 0 ? -fsegs[ s ]->GetDirection1() : -fsegs[ s ]->GetDirection2() );
                                        o.y = 0.0;
                                        double const ol = glm::length( o );
                                        outward = ( ol > 1e-9 ) ? o / ol : glm::dvec3{ 0.0, 0.0, -1.0 };
                                        picked = true;
                                    }
                                }
                            }
                        }
                    }
                    if( picked ) {
                        if( seed->SwitchExtension && ( seedend == 1 || seedend == 3 ) ) {
                            ch.anchor = seed;
                            ch.anchor_end = seedend;
                        }
                        else {
                            ch.origin = glm::dvec3{ point.x, point.y - 0.18, point.z };
                            ch.direction = outward;
                        }
                    }
                    else {
                        ch.origin = cursor_ground_point();
                        glm::mat3 const rot = glm::mat3( GfxRenderer->Camera_View_Matrix() );
                        glm::vec3 const fwd = glm::transpose( rot ) * glm::vec3( 0.0f, 0.0f, -1.0f );
                        glm::dvec3 dir{ fwd.x, 0.0, fwd.z };
                        double const dl = glm::length( dir );
                        ch.direction = ( dl > 1e-6 ) ? dir / dl : glm::dvec3{ 0.0, 0.0, -1.0 };
                    }
                    m_chains.push_back( ch );
                    m_active_chain = (int)m_chains.size() - 1;
                }

                // append one element of the panel type/parameters and re-march
                chain_element el;
                el.type = ui()->track_type();
                el.length = (double)ui()->track_length();
                el.radius = ( el.type == track_panel::TRANSITION ) ? (double)ui()->track_radius_end() : (double)ui()->track_radius();
                el.radius0 = (double)ui()->track_radius_start();
                el.left = ui()->track_curve_left();
                el.cuts = ui()->track_cuts();
                m_chains[ m_active_chain ].elements.push_back( el );
                regenerate_chain( m_active_chain );
                m_input.mouse.button(Button, Action);
                return;
            }

            // niweleta editing: grabbing a chain joint (or a switch) starts a drag; on release
            // the element parameters are re-fitted (type preserved) or the switch is moved
            {
                ImGuiIO const &io = ImGui::GetIO();
                int pickchain, pickjoint;
                if( pick_chain_joint( io.MousePos.x, io.MousePos.y, pickchain, pickjoint ) ) {
                    m_dragactive = true;
                    m_chaindrag_chain = pickchain;
                    m_chaindrag_joint = pickjoint;
                    m_switchdrag = nullptr;
                    m_drag_from = m_chains[ pickchain ].joints[ pickjoint ];
                    m_dragpos = m_drag_from;
                    m_input.mouse.button(Button, Action);
                    return;
                }
                TTrack *seed = nullptr; int seedend = -1;
                glm::dvec3 point, outward;
                if( pick_track_endpoint( io.MousePos.x, io.MousePos.y, true, seed, seedend, point, outward )
                 && seed->SwitchExtension
                 && m_switch_meta.find( seed ) != m_switch_meta.end() ) {
                    // grabbing any switch point moves the whole switch (branch point)
                    m_dragactive = true;
                    m_switchdrag = seed;
                    m_chaindrag_chain = -1;
                    m_drag_from = glm::dvec3{ point.x, point.y - 0.18, point.z };
                    m_dragpos = m_drag_from;
                    m_input.mouse.button(Button, Action);
                    return;
                }
            }

            // delegate node picking behaviour depending on current panel mode
            GfxRenderer->Pick_Node_Callback(
                [this, mode, rotation_mode, fixed_rotation_value](scene::basic_node *node) {
                    // ignore picks that are beyond allowed placement distance
                    if (node) {
                        double const dist = glm::distance(node->location(), glm::dvec3{Global.pCamera.Pos});
                        if (dist > static_cast<double>(kMaxPlacementDistance))
                            return;
                    }
                    if (mode == nodebank_panel::MODIFY)
                    {
                        if (!m_dragging)
                            return;

                        m_node = node;
                        ui()->set_node(m_node);
                    }
                    else if (mode == nodebank_panel::COPY)
                    {
                        if (node && typeid(*node) == typeid(TAnimModel))
                        {
                            std::string as_text;
                            node->export_as_text(as_text);
                            ui()->add_node_template(as_text);
                        }

                        m_dragging = false;
                    }
                    else if (mode == nodebank_panel::ADD)
                    {
                        const std::string *src = ui()->get_active_node_template();
                        if (!src)
                            return;

                        std::string name = "editor_";
                        glm::dvec3 mouseOffset = clamp_mouse_offset_to_max(GfxRenderer->Mouse_Position());
                        TAnimModel *cloned = simulation::State.create_model(*src, name, Camera.Pos + mouseOffset);
                        if (!cloned)
                            return;

                        // record addition for undo
                        std::string as_text;
                        std::string new_name = "editor_" + cloned->uuid.to_string();

                        cloned->m_name = new_name;
                        cloned->export_as_text(as_text);
                        push_snapshot(cloned, EditorSnapshot::Action::Add, as_text);

                        if (!m_dragging)
                            return;

                        m_node = cloned;
                        apply_rotation_for_new_node(m_node, rotation_mode, fixed_rotation_value);
                        ui()->set_node(m_node);
                    }
                });

            m_dragging = true;
            m_takesnapshot = true;
        }
        else
        {
            if (is_release(Action))
                mouseHold = false;

            m_dragging = false;
        }
    }
    else if (Button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        // game-engine style look: hide & grab the cursor while flying, restore it on release
        Application.set_cursor(is_press(Action) ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    }

    m_input.mouse.button(Button, Action);
}

void editor_mode::render_change_history(){


    ImGui::Begin("Editor History", &m_change_history, ImGuiWindowFlags_AlwaysAutoResize);
    int maxsize = m_max_history_size;
    if (ImGui::InputInt("Max history size", &maxsize))
    {
        m_max_history_size = std::max(0, maxsize);
        if ((int)m_history.size() > m_max_history_size && m_max_history_size >= 0)
        {
            auto remove_count = (int)m_history.size() - m_max_history_size;
            m_history.erase(m_history.begin(), m_history.begin() + remove_count);
            // adjust selected index
            if (m_selected_history_idx >= (int)m_history.size())
                m_selected_history_idx = (int)m_history.size() - 1;
        }
        
    }  

    float dist = kMaxPlacementDistance;
    if (ImGui::InputFloat("Max placement distance", &dist))
    {
        kMaxPlacementDistance = std::max(0.0f, dist);
    }

    ImGui::Separator();

    ImGui::Text("History (newest at end): %zu entries", m_history.size());
    ImGui::BeginChild("history_list", ImVec2(400, 200), true);
    for (int i = 0; i < (int)m_history.size(); ++i)
    {
        auto &s = m_history[i];
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%3d: %s %s pos=(%.1f,%.1f,%.1f)", i,
                        s.action == EditorSnapshot::Action::Add ? "ADD" :
                        s.action == EditorSnapshot::Action::Delete ? "DEL" :
                        s.action == EditorSnapshot::Action::Move ? "MOV" :
                        s.action == EditorSnapshot::Action::Rotate ? "ROT" :
                        s.action == EditorSnapshot::Action::Scale ? "SCA" : "OTH",
                        s.node_name.empty() ? "(noname)" : s.node_name.c_str(),
                        s.position.x, s.position.y, s.position.z);

        if (ImGui::Selectable(buf, m_selected_history_idx == i))
            m_selected_history_idx = i;
    }
    ImGui::EndChild();

    ImGui::Separator();
    if (ImGui::Button("Clear History"))
    {
        m_history.clear();
        g_redo.clear();
        m_selected_history_idx = -1;
    }
    ImGui::SameLine();
   
    ImGui::SameLine();
    if (ImGui::Button("Undo Selected"))
    {
        if (m_selected_history_idx >= 0 && m_selected_history_idx < (int)m_history.size())
        {
            int target = m_selected_history_idx;
            int undoCount = (int)m_history.size() - 1 - target;
            for (int k = 0; k < undoCount; ++k)
                undo_last();
            m_selected_history_idx = -1;
        }
    }      

    ImGui::End();
}


void editor_mode::on_event_poll()
{
    // game-engine style camera: WSAD/EQ only fly the camera while the right mouse button is held.
    // when it's released the keyboard is free for gizmo shortcuts, and we flush a zero-movement
    // command once so the camera doesn't keep coasting on the last velocity it was given.
    bool const flying = m_input.mouse.button(GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    if (flying)
    {
        m_input.poll();
    }
    else if (m_camera_flying)
    {
        m_camera_relay.post(user_command::movehorizontal, 0.0, 0.0, GLFW_PRESS, 0);
        m_camera_relay.post(user_command::movevertical, 0.0, 0.0, GLFW_PRESS, 0);
    }
    m_camera_flying = flying;
}

bool editor_mode::is_command_processor() const
{
    return false;
}

bool editor_mode::focus_active()
{
    return m_focus_active;
}

void editor_mode::set_focus_active(bool isActive)
{
    m_focus_active = isActive;
}
