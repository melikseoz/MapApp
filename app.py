"""
Road Network Analyzer
Load road vector data, click two points, see the shortest path and distance.
"""

import copy
import os
import tempfile
import zipfile

import folium
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from pyproj import Geod
from streamlit.errors import StreamlitAPIException
from shapely.geometry import LineString, MultiLineString, Point
from shapely.strtree import STRtree
from streamlit_folium import st_folium

MPH_TO_KPH = 1.609344

# ── Live geolocation component ──────────────────────────────────────────────────
# A small inline st.components.v2 component (no npm/React build) that fetches the
# browser's location automatically — once, or repeatedly on a timer — rather than
# requiring a button tap per update like the streamlit-geolocation package did.
#
# `data` carries {enabled, intervalMs}. We don't rely on data changes alone to
# restart the timer with a new interval — the Python side folds both settings into
# the mount `key`, and a changed key is a documented, reliable way to force
# Streamlit to tear down (running the returned cleanup) and remount the frontend
# element, so the old interval is always cleared before a new one is set up.
_LIVE_LOCATION_HTML = """
<div id="live-location-status" style="font-size:13px;color:#555;">📍 Requesting location…</div>
"""

_LIVE_LOCATION_JS = """
export default function(component) {
    const { setStateValue, parentElement, data } = component;
    const statusEl = parentElement.querySelector('#live-location-status');
    const setStatus = (text) => { if (statusEl) statusEl.textContent = text; };

    if (!navigator.geolocation) {
        setStatus('📍 Geolocation is not supported by this browser.');
        setStateValue('location', { latitude: null, longitude: null, accuracy: null, error: 'not_supported' });
        return;
    }

    const fetchOnce = () => {
        navigator.geolocation.getCurrentPosition(
            (pos) => {
                setStatus(data.enabled
                    ? '📍 Live — updating every ' + (data.intervalMs / 1000) + 's'
                    : '📍 Location set (auto-update off)');
                setStateValue('location', {
                    latitude: pos.coords.latitude,
                    longitude: pos.coords.longitude,
                    accuracy: pos.coords.accuracy,
                    error: null,
                });
            },
            (err) => {
                setStatus('📍 ' + err.message);
                setStateValue('location', { latitude: null, longitude: null, accuracy: null, error: err.message });
            },
            // maximumAge: 0 forces a fresh GPS fix each call instead of a cached one —
            // important for polling, otherwise repeated calls inside the cache window
            // would silently return the same stale position.
            { enableHighAccuracy: true, timeout: 15000, maximumAge: 0 }
        );
    };

    setStatus('📍 Requesting location…');
    fetchOnce();

    const intervalId = data.enabled ? setInterval(fetchOnce, data.intervalMs) : null;

    return () => { if (intervalId) clearInterval(intervalId); };
}
"""

# Registered once at import time — re-registering on every call would log warnings
# and is unnecessary, since the mount command below can be invoked repeatedly.
_live_location = st.components.v2.component(
    "live_location", html=_LIVE_LOCATION_HTML, js=_LIVE_LOCATION_JS
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Road Network Analyzer",
    page_icon="🗺️",
    layout="wide",
)

# ── Session state defaults ─────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "gdf": None,
    "graph": None,
    "node_array": None,    # np.ndarray (N, 2) — columns: [lon, lat]
    "coord_to_id": None,   # dict[(round_lon, round_lat)] -> node_id
    "geom_tree": None,     # shapely STRtree over gdf.geometry
    "points": [None, None],       # [start_or_None, end_or_None]  (snap positions)
    "snap_data": [None, None],    # [start_snap_or_None, end_snap_or_None] — aligned with points
    "app_mode": "Plan",    # "Plan" (tap twice) | "Live" (geolocation = Start, tap = End)
    "auto_update_location": True,   # Live mode: keep polling location on a timer vs. fetch once
    "location_update_interval_s": 3.0,  # Live mode: polling interval when auto-update is on
    "path_result": None,   # {"coords", "dist_m", "time_s", "avg_speed_kph", "unknown_edges", "total_route_edges"} | None
    "path_warning": None,  # str | None
    "last_click": None,    # (lat, lng) of last processed click
    "last_geolocation": None,  # (lat, lng) of last processed "use my location" result
    "base_map": None,      # folium.Map — built once per file load, never rebuilt on interaction
    "unit_system": "Metric",  # "Metric" | "Imperial"
    "speed_stats": None,   # {"column", "source_unit", "total_edges", "missing_edges"} | None
    "default_speed_input": 30.0,  # fallback speed, in current unit_system's units
    "walking_speed_mph": 3.0,  # always mph, independent of unit_system and speed-limit data
    "mobile_view": False,  # user-toggled compact layout for small/touch screens
    "base_map_key": None,  # (loaded_file_key, mobile_view) the current base_map was built from
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── File loading ───────────────────────────────────────────────────────────────
def _resolve_vector_path(tmp_dir: str, file_bytes: bytes, file_name: str) -> str:
    dest = os.path.join(tmp_dir, file_name)
    with open(dest, "wb") as fh:
        fh.write(file_bytes)

    if file_name.lower().endswith(".zip"):
        with zipfile.ZipFile(dest) as z:
            z.extractall(tmp_dir)
        for root, _, files in os.walk(tmp_dir):
            for fname in sorted(files):
                if fname.lower().endswith((".shp", ".gpkg", ".geojson", ".json")):
                    return os.path.join(root, fname)
        raise ValueError("No supported vector file (.shp / .gpkg / .geojson) found inside ZIP.")

    return dest


@st.cache_data(show_spinner=False)
def load_and_build(file_bytes: bytes, file_name: str, unit_system: str):
    """Parse the uploaded file, filter to line geometries, build a graph. Cached by file
    content *and* unit_system — a bare "speed_limit" column (no _mph/_kph suffix) is
    interpreted using unit_system, so switching units must rebuild the graph."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _resolve_vector_path(tmp, file_bytes, file_name)
        raw = gpd.read_file(path)

    gdf = (
        raw[raw.geometry.geom_type.isin(["LineString", "MultiLineString"])]
        .copy()
        .to_crs("EPSG:4326")
        .reset_index(drop=True)
    )
    if gdf.empty:
        raise ValueError(
            "No line geometry found. Make sure the file contains road/path line data."
        )

    G, node_array, coord_to_id, speed_stats = _build_graph(gdf, unit_system)
    return gdf, G, node_array, coord_to_id, speed_stats


# ── Speed-limit detection ──────────────────────────────────────────────────────
def _detect_speed_column(gdf: gpd.GeoDataFrame, unit_system: str) -> tuple[str | None, str | None]:
    """Return (column_name, source_unit) — source_unit is "kph" or "mph".

    Priority: an explicit `speed_limit_kph` / `speed_limit_mph` column wins over a bare
    `speed_limit` column, whose unit is ambiguous and is assumed to follow unit_system.
    """
    cols_lower = {c.lower(): c for c in gdf.columns}
    if "speed_limit_kph" in cols_lower:
        return cols_lower["speed_limit_kph"], "kph"
    if "speed_limit_mph" in cols_lower:
        return cols_lower["speed_limit_mph"], "mph"
    if "speed_limit" in cols_lower:
        return cols_lower["speed_limit"], ("mph" if unit_system == "Imperial" else "kph")
    return None, None


def _speed_kph_series(gdf: gpd.GeoDataFrame, speed_col: str | None, source_unit: str | None) -> pd.Series:
    """Per-row speed limit in km/h (NaN where absent), aligned with gdf's index."""
    if speed_col is None:
        return pd.Series(np.nan, index=gdf.index, dtype=float)
    factor = MPH_TO_KPH if source_unit == "mph" else 1.0
    return pd.to_numeric(gdf[speed_col], errors="coerce") * factor


# ── Graph construction ─────────────────────────────────────────────────────────
def _build_graph(gdf: gpd.GeoDataFrame, unit_system: str):
    geod = Geod(ellps="WGS84")
    G: nx.Graph = nx.Graph()
    coord_to_id: dict[tuple[float, float], int] = {}
    positions: list[tuple[float, float]] = []

    speed_col, source_unit = _detect_speed_column(gdf, unit_system)
    speeds_kph = _speed_kph_series(gdf, speed_col, source_unit)
    total_edges = 0
    missing_edges = 0

    def _node(lon: float, lat: float) -> int:
        key = (round(lon, 7), round(lat, 7))
        if key not in coord_to_id:
            nid = len(positions)
            coord_to_id[key] = nid
            positions.append(key)
            G.add_node(nid, x=key[0], y=key[1])
        return coord_to_id[key]

    for row_idx, geom in enumerate(gdf.geometry):
        row_speed = speeds_kph.iloc[row_idx]
        speed_kph = None if pd.isna(row_speed) else float(row_speed)

        segs = list(geom.geoms) if isinstance(geom, MultiLineString) else [geom]
        for seg in segs:
            if not isinstance(seg, LineString):
                continue
            coords = list(seg.coords)
            for i in range(len(coords) - 1):
                lon1, lat1 = coords[i][0], coords[i][1]
                lon2, lat2 = coords[i + 1][0], coords[i + 1][1]
                n1, n2 = _node(lon1, lat1), _node(lon2, lat2)
                if n1 != n2 and not G.has_edge(n1, n2):
                    _, _, dist = geod.inv(lon1, lat1, lon2, lat2)
                    G.add_edge(n1, n2, weight=abs(dist), speed_kph=speed_kph)
                    if speed_col is not None:
                        total_edges += 1
                        if speed_kph is None:
                            missing_edges += 1

    speed_stats = {
        "column": speed_col,
        "source_unit": source_unit,
        "total_edges": total_edges,
        "missing_edges": missing_edges,
        "min_kph": float(speeds_kph.min()) if speed_col is not None and speeds_kph.notna().any() else None,
        "max_kph": float(speeds_kph.max()) if speed_col is not None and speeds_kph.notna().any() else None,
    }
    return G, np.array(positions, dtype=np.float64), coord_to_id, speed_stats


# ── Snapping to nearest point ON the line ─────────────────────────────────────
def _find_enclosing_segment(
    line: LineString, d_along: float, coord_to_id: dict
) -> tuple[int | None, int | None]:
    """Return the two graph node IDs that flank the projection point on *line*."""
    coords = list(line.coords)
    d_cum = 0.0
    for i in range(len(coords) - 1):
        c1, c2 = coords[i][:2], coords[i + 1][:2]
        seg_len = Point(c1).distance(Point(c2))
        if d_cum + seg_len >= d_along - 1e-10:
            n1 = coord_to_id.get((round(c1[0], 7), round(c1[1], 7)))
            n2 = coord_to_id.get((round(c2[0], 7), round(c2[1], 7)))
            return n1, n2
        d_cum += seg_len
    # Fallback: last segment
    c1, c2 = coords[-2][:2], coords[-1][:2]
    return (
        coord_to_id.get((round(c1[0], 7), round(c1[1], 7))),
        coord_to_id.get((round(c2[0], 7), round(c2[1], 7))),
    )


def _snap_to_line(lon: float, lat: float) -> tuple[float, float, int | None, int | None]:
    """Return (snap_lon, snap_lat, n1, n2) — snap point and the two flanking graph nodes."""
    gdf: gpd.GeoDataFrame = st.session_state.gdf
    geom_tree: STRtree = st.session_state.geom_tree
    coord_to_id: dict = st.session_state.coord_to_id

    click_pt = Point(lon, lat)
    nearest_idx = int(geom_tree.nearest(click_pt))
    nearest_geom = gdf.geometry.iloc[nearest_idx]

    if isinstance(nearest_geom, MultiLineString):
        nearest_line = min(nearest_geom.geoms, key=lambda g: g.distance(click_pt))
    else:
        nearest_line = nearest_geom

    d_along = nearest_line.project(click_pt)
    snap_pt = nearest_line.interpolate(d_along)
    snap_lon, snap_lat = snap_pt.x, snap_pt.y

    n1, n2 = _find_enclosing_segment(nearest_line, d_along, coord_to_id)
    return snap_lon, snap_lat, n1, n2


def _default_speed_kph() -> float:
    """The user-configurable fallback speed (sidebar input), converted to km/h."""
    value = st.session_state.default_speed_input
    return value * MPH_TO_KPH if st.session_state.unit_system == "Imperial" else value


# ── Path finding ───────────────────────────────────────────────────────────────
def _find_path(
    start_snap: tuple[float, float, int | None, int | None],
    end_snap: tuple[float, float, int | None, int | None],
):
    """Path between two snapped points, via temporary virtual nodes.

    Each snap point is wired to *both* of its flanking graph nodes (not just
    the nearer one) so Dijkstra can pick whichever side is actually optimal.
    When both points snap onto the same edge, a direct virtual-to-virtual
    edge is added too, so the direct sub-segment is considered instead of a
    detour out to the edge's endpoints and back.

    When speed-limit data was found on the source file, routing optimizes for
    fastest estimated travel time (edges with no speed value use the sidebar
    fallback speed); otherwise it falls back to plain shortest-distance
    routing, unchanged from before speed limits were supported.
    """
    G: nx.Graph = st.session_state.graph
    arr: np.ndarray = st.session_state.node_array
    speed_stats: dict = st.session_state.speed_stats
    has_speed = bool(speed_stats and speed_stats["column"] is not None)
    fallback_kph = _default_speed_kph() if has_speed else None
    geod = Geod(ellps="WGS84")

    slon, slat, s_n1, s_n2 = start_snap
    elon, elat, e_n1, e_n2 = end_snap

    def _dist(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        _, _, d = geod.inv(lon1, lat1, lon2, lat2)
        return abs(d)

    def _edge_time(_u, _v, data: dict) -> float:
        speed_kph = data.get("speed_kph") or fallback_kph
        return data["weight"] / (speed_kph * 1000.0 / 3600.0)

    def _flank_speed(n1: int | None, n2: int | None) -> float | None:
        """Speed of the original segment a virtual point was snapped onto."""
        if n1 is not None and n2 is not None and G.has_edge(n1, n2):
            return G[n1][n2].get("speed_kph")
        return None

    START, END = "__start__", "__end__"
    added_nodes: list[str] = []
    try:
        for label, lon, lat, n1, n2 in (
            (START, slon, slat, s_n1, s_n2),
            (END, elon, elat, e_n1, e_n2),
        ):
            flanks = [n for n in (n1, n2) if n is not None and G.has_node(n)]
            if not flanks:
                diffs = arr - np.array([lon, lat])
                flanks = [int(np.argmin(np.hypot(diffs[:, 0], diffs[:, 1])))]
            G.add_node(label)
            added_nodes.append(label)
            edge_speed = _flank_speed(n1, n2) if has_speed else None
            for n in flanks:
                G.add_edge(
                    label,
                    n,
                    weight=_dist(lon, lat, float(arr[n, 0]), float(arr[n, 1])),
                    speed_kph=edge_speed,
                )

        if (
            s_n1 is not None and s_n2 is not None
            and e_n1 is not None and e_n2 is not None
            and {s_n1, s_n2} == {e_n1, e_n2}
        ):
            G.add_edge(
                START,
                END,
                weight=_dist(slon, slat, elon, elat),
                speed_kph=_flank_speed(s_n1, s_n2) if has_speed else None,
            )

        try:
            weight_fn = _edge_time if has_speed else "weight"
            path_nodes = nx.shortest_path(G, START, END, weight=weight_fn)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

        coords = []
        for n in path_nodes:
            if n == START:
                coords.append((slon, slat))
            elif n == END:
                coords.append((elon, elat))
            else:
                coords.append((float(arr[n, 0]), float(arr[n, 1])))

        total_dist = 0.0
        total_time = 0.0 if has_speed else None
        speed_dist_sum = 0.0  # Σ(segment distance × segment speed), for the distance-weighted average
        unknown_edges = 0
        total_route_edges = 0
        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
            data = G[u][v]
            total_dist += data["weight"]
            if has_speed:
                total_time += _edge_time(u, v, data)
                total_route_edges += 1
                edge_speed_kph = data.get("speed_kph")
                if edge_speed_kph is None:
                    unknown_edges += 1
                    edge_speed_kph = fallback_kph
                speed_dist_sum += data["weight"] * edge_speed_kph

        avg_speed_kph = speed_dist_sum / total_dist if has_speed and total_dist > 0 else None

        return {
            "coords": coords,
            "dist_m": total_dist,
            "time_s": total_time,
            "avg_speed_kph": avg_speed_kph,
            "unknown_edges": unknown_edges,
            "total_route_edges": total_route_edges,
        }
    finally:
        G.remove_nodes_from(added_nodes)


# ── Click handlers ─────────────────────────────────────────────────────────────
def _recompute_path() -> None:
    """Recompute path_result/path_warning from whatever is currently in snap_data."""
    snaps: list = st.session_state.snap_data
    if snaps[0] is not None and snaps[1] is not None:
        result = _find_path(snaps[0], snaps[1])
        if result:
            st.session_state.path_result = result
            st.session_state.path_warning = None
        else:
            st.session_state.path_result = None
            st.session_state.path_warning = (
                "No connected path found between the two selected points. "
                "They may be on disconnected parts of the network."
            )
    else:
        st.session_state.path_result = None
        st.session_state.path_warning = None


def _set_point(index: int, lat: float, lon: float) -> None:
    """Live mode: set/overwrite one fixed slot (0=Start, 1=End) directly — no cycling."""
    pts = list(st.session_state.points)
    snaps = list(st.session_state.snap_data)
    snap_lon, snap_lat, n1, n2 = _snap_to_line(lon, lat)
    pts[index] = (snap_lon, snap_lat)
    snaps[index] = (snap_lon, snap_lat, n1, n2)
    st.session_state.points = pts
    st.session_state.snap_data = snaps
    _recompute_path()


def _handle_click(lat: float, lon: float) -> None:
    """Plan mode: 1st tap sets Start, 2nd sets End, 3rd starts over."""
    pts = list(st.session_state.points)
    snaps = list(st.session_state.snap_data)

    if pts[0] is not None and pts[1] is not None:
        pts, snaps = [None, None], [None, None]

    index = 0 if pts[0] is None else 1
    snap_lon, snap_lat, n1, n2 = _snap_to_line(lon, lat)
    pts[index] = (snap_lon, snap_lat)
    snaps[index] = (snap_lon, snap_lat, n1, n2)
    st.session_state.points = pts
    st.session_state.snap_data = snaps
    _recompute_path()


def _on_mode_change() -> None:
    """Plan and Live assign Start/End differently, so a stale selection from the
    other mode would be misleading — clear it whenever the mode toggle flips."""
    st.session_state.points = [None, None]
    st.session_state.snap_data = [None, None]
    st.session_state.path_result = None
    st.session_state.path_warning = None


# ── Speed color ramp ───────────────────────────────────────────────────────────
# Anchors reuse the design system's status palette (critical → serious → warning →
# good), interpolated into a continuous red(slow) → amber → green(fast) ramp.
_SPEED_RAMP_STOPS: list[tuple[float, tuple[int, int, int]]] = [
    (0.0, (0xD0, 0x3B, 0x3B)),   # critical — red    — slowest
    (1 / 3, (0xEC, 0x83, 0x5A)),  # serious  — orange
    (2 / 3, (0xFA, 0xB2, 0x19)),  # warning  — amber
    (1.0, (0x0C, 0xA3, 0x0C)),   # good     — green   — fastest
]
_SPEED_UNKNOWN_COLOR = "#555555"


def _speed_ramp_color(t: float) -> str:
    """Interpolate the red→amber→green speed ramp at t ∈ [0, 1]."""
    t = max(0.0, min(1.0, t))
    for (t0, c0), (t1, c1) in zip(_SPEED_RAMP_STOPS, _SPEED_RAMP_STOPS[1:]):
        if t <= t1:
            frac = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            r = round(c0[0] + (c1[0] - c0[0]) * frac)
            g = round(c0[1] + (c1[1] - c0[1]) * frac)
            b = round(c0[2] + (c1[2] - c0[2]) * frac)
            return f"#{r:02x}{g:02x}{b:02x}"
    r, g, b = _SPEED_RAMP_STOPS[-1][1]
    return f"#{r:02x}{g:02x}{b:02x}"


def _build_speed_legend(
    min_kph: float, max_kph: float, unit_system: str, compact: bool = False
) -> str:
    """Small floating HTML legend for the red→green speed-limit color ramp.

    compact=True (mobile view) renders larger text and a thicker bar — legible
    outdoors at arm's length — rather than a smaller, desktop-style footprint.
    """
    if unit_system == "Imperial":
        lo, hi, unit = min_kph / MPH_TO_KPH, max_kph / MPH_TO_KPH, "mph"
    else:
        lo, hi, unit = min_kph, max_kph, "km/h"
    stops_css = ", ".join(f"{_speed_ramp_color(t)} {t * 100:.0f}%" for t, _ in _SPEED_RAMP_STOPS)
    font_size = "16px" if compact else "12px"
    bar_width = "220px" if compact else "160px"
    bar_height = "16px" if compact else "10px"
    swatch_size = "16px" if compact else "10px"
    return f"""
    <div style="
        position: fixed; bottom: 24px; left: 24px; z-index: 9999;
        background: white; padding: 8px 12px; border-radius: 6px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.3); font-size: {font_size}; color: #222;
        font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
    ">
        <div style="margin-bottom: 4px; font-weight: 600;">Speed limit</div>
        <div style="width: {bar_width}; height: {bar_height}; border-radius: 4px;
                    background: linear-gradient(to right, {stops_css});"></div>
        <div style="display:flex; justify-content:space-between; margin-top: 2px;">
            <span>{lo:.0f} {unit}</span><span>{hi:.0f} {unit}</span>
        </div>
        <div style="margin-top: 4px; color: #666;">
            <span style="display:inline-block;width:{swatch_size};height:{swatch_size};background:{_SPEED_UNKNOWN_COLOR};
                         border-radius:2px;vertical-align:middle;margin-right:4px;"></span>
            No data
        </div>
    </div>
    """


# ── Map rendering ──────────────────────────────────────────────────────────────
def _build_base_map(
    gdf: gpd.GeoDataFrame,
    center: list[float],
    zoom: int,
    speed_stats: dict | None,
    unit_system: str,
    compact: bool = False,
) -> folium.Map:
    """Build the base map (tiles + roads layer) once per file load.

    st_folium remounts the whole Leaflet map (black flash, view snapping back
    to the previous position) whenever the rendered map HTML changes — which
    happens on every rerun if we reconstruct folium.Map() from scratch, since
    Folium bakes fresh random element IDs into it every time. Building this
    once, then handing a copy.deepcopy() of it to st_folium on each rerun
    (see _map_section), keeps those IDs — and the generated HTML — identical
    across reruns, so the map is never remounted. Passing the *same* live
    object instead would look identical here, but streamlit-folium's
    feature_group_to_add mutates its map argument (permanently attaches the
    overlay as a real child), so every next render's HTML would silently pick
    up the previous render's markers/path baked into the "base" map too.

    Road segments are colored by speed limit (red = slowest, green = fastest)
    when the source file has speed-limit data; segments without a value, or
    the whole layer when no speed column was found at all, render in neutral
    gray.
    """
    # Esri World Imagery (satellite/aerial) — free, no API key, unlike CartoDB's
    # now-gated tiles. Built as an explicit TileLayer (rather than via folium.Map's
    # tiles=/attr= shortcut) because that shortcut only forwards max_zoom, not
    # max_native_zoom — and Esri's own tiles run out around zoom 19, so without
    # max_native_zoom, Leaflet caps out there instead of upscaling further.
    m = folium.Map(location=center, zoom_start=zoom, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics",
        max_zoom=21,
        max_native_zoom=19,
    ).add_to(m)

    if compact:
        # Leaflet's default attribution bar eats a disproportionate slice of a
        # short mobile map height — shrink it; desktop's taller map doesn't need this.
        m.get_root().html.add_child(folium.Element(
            """<style>
            .leaflet-control-attribution {
                font-size: 9px !important;
                padding: 0 4px !important;
                line-height: 1.3 !important;
                max-width: 65vw;
                white-space: normal !important;
            }
            </style>"""
        ))

    speed_col = speed_stats["column"] if speed_stats else None
    min_kph = speed_stats.get("min_kph") if speed_stats else None
    max_kph = speed_stats.get("max_kph") if speed_stats else None
    has_speed_range = speed_col is not None and min_kph is not None and max_kph is not None

    gray_weight = 3 if compact else 2
    colored_weight = 4 if compact else 3

    if has_speed_range:
        source_unit = speed_stats["source_unit"]
        factor = MPH_TO_KPH if source_unit == "mph" else 1.0
        span = max_kph - min_kph
        uniform = span < 1e-6

        def _style(feature: dict) -> dict:
            raw = feature["properties"].get(speed_col)
            if pd.isna(raw):
                return {"color": _SPEED_UNKNOWN_COLOR, "weight": gray_weight, "opacity": 0.65, "interactive": False}
            speed_kph = float(raw) * factor
            t = 0.5 if uniform else (speed_kph - min_kph) / span
            return {"color": _speed_ramp_color(t), "weight": colored_weight, "opacity": 0.85, "interactive": False}
    else:
        # No speed-limit column at all anywhere in the file (as opposed to a column
        # that exists but is missing on some rows — that case stays gray above) —
        # fall back to plain red at the original, pre-speed-ramp line thickness.
        no_data_color = _speed_ramp_color(0.0)

        def _style(_feature: dict) -> dict:
            return {"color": no_data_color, "weight": 3, "opacity": 0.85, "interactive": False}

    folium.GeoJson(gdf.__geo_interface__, name="Roads", style_function=_style).add_to(m)

    if has_speed_range:
        m.get_root().html.add_child(
            folium.Element(_build_speed_legend(min_kph, max_kph, unit_system, compact=compact))
        )

    return m


def _format_distance(dist_m: float, unit_system: str) -> tuple[str, str, str, str]:
    """Returns (primary_label, primary_value, secondary_label, secondary_value)."""
    if unit_system == "Imperial":
        miles = dist_m / 1609.344
        feet = dist_m * 3.28084
        return "Path length", f"{miles:.3f} mi", "In feet", f"{feet:,.0f} ft"
    return "Path length", f"{dist_m / 1000:.3f} km", "In metres", f"{dist_m:,.0f} m"


def _format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _walking_time_s(dist_m: float, walking_mph: float) -> float:
    """Walking time for a distance, at a flat walking speed — ignores speed limits entirely."""
    walking_mps = walking_mph * MPH_TO_KPH * 1000.0 / 3600.0
    return dist_m / walking_mps if walking_mps > 0 else float("inf")


def _build_overlay() -> folium.FeatureGroup:
    """Markers + path line — rebuilt on every interaction and applied
    via feature_group_to_add, which streamlit-folium updates in place without
    remounting the map."""
    points: list = st.session_state.points
    path_result = st.session_state.path_result

    fg = folium.FeatureGroup(name="selection")

    _marker_cfg = [("Start", "green"), ("End", "red")]
    for i, pt in enumerate(points):
        if pt is None:
            continue
        lon, lat = pt
        label, color = _marker_cfg[i]
        folium.Marker(
            location=[lat, lon],
            tooltip=f"<b>{label}</b><br>({lat:.5f}, {lon:.5f})",
            icon=folium.Icon(color=color, icon="map-marker", prefix="fa"),
        ).add_to(fg)

    if path_result:
        _, primary_val, _, secondary_val = _format_distance(
            path_result["dist_m"], st.session_state.unit_system
        )
        tooltip = f"Path — {primary_val} ({secondary_val})"
        if path_result["time_s"] is not None:
            tooltip += f" · ~{_format_duration(path_result['time_s'])}"
        folium.PolyLine(
            locations=[[lat, lon] for lon, lat in path_result["coords"]],
            color="#1565C0",
            weight=6,
            opacity=0.9,
            tooltip=tooltip,
        ).add_to(fg)

    return fg


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🗺️ Road Network Analyzer")
    st.divider()

    st.radio(
        "Mode",
        ["Plan", "Live"],
        horizontal=True,
        key="app_mode",
        on_change=_on_mode_change,
        help="Plan: tap two points to plan a route ahead of time. "
             "Live: your current location is the Start — tap the map to set the destination.",
    )
    if st.session_state.app_mode == "Live":
        st.checkbox(
            "🔄 Auto-update location",
            key="auto_update_location",
            help="On: refreshes your position on a timer as you move. "
                 "Off: fetches your location once and holds it until you toggle this again.",
        )
        if st.session_state.auto_update_location:
            st.number_input(
                "Update every (seconds)",
                min_value=1.0,
                max_value=60.0,
                step=1.0,
                value=st.session_state.location_update_interval_s,
                key="location_update_interval_s",
            )
    st.radio("Units", ["Metric", "Imperial"], horizontal=True, key="unit_system")
    st.checkbox(
        "📱 Compact / mobile view",
        key="mobile_view",
        help="Shorter map, stacked results, bigger tap-friendly legend — turn on when using this on a phone.",
    )
    st.divider()

    uploaded = st.file_uploader(
        "Upload road network file",
        type=["geojson", "json", "gpkg", "zip"],
        help="GeoJSON, GeoPackage (.gpkg), or a ZIP containing a Shapefile.",
    )

    if uploaded is not None:
        file_key = (uploaded.name, uploaded.size, st.session_state.unit_system)
        if file_key != st.session_state.get("loaded_file_key"):
            with st.spinner("Loading file and building network graph…"):
                try:
                    gdf, G, node_array, coord_to_id, speed_stats = load_and_build(
                        uploaded.getvalue(), uploaded.name, st.session_state.unit_system
                    )
                    st.session_state.gdf = gdf
                    st.session_state.graph = G
                    st.session_state.node_array = node_array
                    st.session_state.coord_to_id = coord_to_id
                    st.session_state.geom_tree = STRtree(list(gdf.geometry))
                    st.session_state.speed_stats = speed_stats
                    st.session_state.loaded_file_key = file_key
                    st.session_state.points = [None, None]
                    st.session_state.snap_data = [None, None]
                    st.session_state.path_result = None
                    st.session_state.path_warning = None
                    # Deliberately NOT resetting last_click (or last_geolocation) here:
                    # they only exist to de-duplicate against each component's persisted
                    # last value. Both components keep that value across this rerun (they
                    # aren't remounted), so clearing our tracker would make the old value
                    # look "new" again on the next render and re-fire _handle_click with
                    # stale coordinates the user didn't just (re-)submit — which, worse,
                    # can happen during a full rerun (not a fragment rerun), where the
                    # resulting st.rerun(scope="fragment") call raises.
                except Exception as exc:
                    st.error(f"Error loading file: {exc}")

    # Rebuild just the lightweight base map (tiles + styled roads + legend) whenever
    # the loaded file/units change (above) or the mobile-view toggle flips. This is
    # cheap — it reuses the already-built graph/gdf, no re-parsing or re-graphing —
    # so it's fine to check on every rerun rather than gating it behind file_key.
    if st.session_state.gdf is not None:
        base_map_key = (st.session_state.loaded_file_key, st.session_state.mobile_view)
        if base_map_key != st.session_state.get("base_map_key"):
            gdf = st.session_state.gdf
            bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
            center = [
                (bounds[1] + bounds[3]) / 2.0,
                (bounds[0] + bounds[2]) / 2.0,
            ]
            st.session_state.base_map = _build_base_map(
                gdf, center, zoom=13,
                speed_stats=st.session_state.speed_stats,
                unit_system=st.session_state.unit_system,
                compact=st.session_state.mobile_view,
            )
            st.session_state.base_map_key = base_map_key

    if st.session_state.gdf is not None:
        G: nx.Graph = st.session_state.graph
        gdf: gpd.GeoDataFrame = st.session_state.gdf
        st.success(
            f"**{len(gdf):,}** road segments  \n"
            f"**{G.number_of_nodes():,}** nodes · **{G.number_of_edges():,}** edges"
        )

        st.divider()
        st.number_input(
            "Walking speed (mph)",
            min_value=0.1,
            step=0.1,
            value=st.session_state.walking_speed_mph,
            key="walking_speed_mph",
            help="Used for the walking-time estimate — ignores road speed limits entirely.",
        )

        speed_stats: dict | None = st.session_state.speed_stats
        has_speed = bool(speed_stats and speed_stats["column"] is not None)
        if has_speed:
            st.divider()
            unit_label = "mph" if st.session_state.unit_system == "Imperial" else "km/h"
            st.caption(f"Speed limits: **{speed_stats['column']}** ({unit_label}) · routing by fastest time")
            if speed_stats["missing_edges"] > 0:
                st.warning(
                    f"{speed_stats['missing_edges']:,} of {speed_stats['total_edges']:,} "
                    "road segments have no speed limit assigned. A default speed is "
                    "assumed for those below."
                )
            fallback_unit = "mph" if st.session_state.unit_system == "Imperial" else "km/h"
            st.number_input(
                f"Default speed for segments without data ({fallback_unit})",
                min_value=1.0,
                step=1.0,
                value=st.session_state.default_speed_input,
                key="default_speed_input",
            )
            # Re-run routing so an edited fallback speed is reflected immediately,
            # rather than only on the next map click.
            _recompute_path()

        st.divider()
        tap_or_click = "Tap" if st.session_state.mobile_view else "Click"
        if st.session_state.app_mode == "Live":
            st.markdown(
                "**Instructions**\n"
                "1. Allow location access when prompted — **Start** (green marker) "
                "tracks your current position automatically\n"
                f"2. {tap_or_click} anywhere on the map → **End** (red marker)\n"
                + ("3. Fastest route by travel time is drawn automatically, updating as you move\n" if has_speed
                   else "3. Shortest path is drawn automatically, updating as you move\n")
            )
        else:
            st.markdown(
                "**Instructions**\n"
                f"1. {tap_or_click} anywhere on the map → **Start** (green marker)\n"
                f"2. {tap_or_click} again → **End** (red marker)\n"
                + ("3. Fastest route by travel time is drawn automatically\n" if has_speed
                   else "3. Shortest path is drawn automatically\n")
                + f"4. {tap_or_click} a third time to start over"
            )
        if st.button("↺  Reset selection", use_container_width=True):
            st.session_state.points = [None, None]
            st.session_state.snap_data = [None, None]
            st.session_state.path_result = None
            st.session_state.path_warning = None
            # last_click / last_geolocation intentionally left alone — see the loader above.
            st.rerun()
    else:
        st.info("Upload a road network file to get started.")
        st.markdown(
            "**Supported formats**\n"
            "- `.geojson` / `.json`\n"
            "- `.gpkg` (GeoPackage)\n"
            "- `.zip` containing a Shapefile\n\n"
            "The file must contain **line geometry** (roads, paths, etc.).\n\n"
            "**Free road data sources**\n"
            "- [Geofabrik OSM extracts](https://download.geofabrik.de/)\n"
            "- [BBBike extracts](https://extract.bbbike.org/)"
        )


# ── Main area ──────────────────────────────────────────────────────────────────
if st.session_state.gdf is None:
    st.markdown("## Road Network Analyzer")
    st.markdown(
        "Upload a vector file containing road line data using the **sidebar** to get started.  \n"
        "Once loaded, click two points on the map to compute the shortest path between them."
    )
else:
    st.markdown(
        """<style>
        .leaflet-container { cursor: crosshair !important; }
        .leaflet-dragging .leaflet-container { cursor: grabbing !important; }
        </style>""",
        unsafe_allow_html=True,
    )

    @st.fragment
    def _map_section() -> None:
        is_live = st.session_state.app_mode == "Live"

        if is_live:
            auto_update = st.session_state.auto_update_location
            interval_s = st.session_state.location_update_interval_s
            result = _live_location(
                # The key folds in both settings so toggling auto-update or changing the
                # interval forces a clean remount (old timer cleared, new one started)
                # instead of relying on unclear in-place "data changed" semantics.
                key=f"live_location_{auto_update}_{interval_s}",
                data={"enabled": auto_update, "intervalMs": int(interval_s * 1000)},
                default={"location": None},
                on_location_change=lambda: None,
            )
            loc = result.location

            if loc and loc.get("error"):
                st.caption(f"⚠️ Location unavailable — {loc['error']}")
            elif loc and loc.get("latitude") is not None:
                # Rounded to ~11m (4 decimal places) rather than the map-click precision
                # (6 decimals / ~11cm), since polling fires repeatedly and GPS noise
                # alone would otherwise re-trigger routing every update.
                geo_key = (round(loc["latitude"], 4), round(loc["longitude"], 4))
                if geo_key != st.session_state.last_geolocation:
                    st.session_state.last_geolocation = geo_key
                    _set_point(0, loc["latitude"], loc["longitude"])
                    # Same scope="fragment" caveat as the map-click handler below — fall
                    # back to a full rerun if this isn't actually a fragment rerun.
                    try:
                        st.rerun(scope="fragment")
                    except StreamlitAPIException:
                        st.rerun()

        fg = _build_overlay()
        map_data = st_folium(
            copy.deepcopy(st.session_state.base_map),
            feature_group_to_add=fg,
            key="road_map",
            returned_objects=["last_clicked"],
            use_container_width=True,
            height=540 if st.session_state.mobile_view else 680,
        )

        if st.session_state.path_warning:
            st.warning(st.session_state.path_warning)

        # Distance / time result — shown here so it updates on every fragment rerun
        if st.session_state.path_result:
            result = st.session_state.path_result
            label1, val1, label2, val2 = _format_distance(
                result["dist_m"], st.session_state.unit_system
            )
            metrics = [(label1, val1), (label2, val2)]
            if result["time_s"] is not None:
                metrics.append(("Est. travel time", _format_duration(result["time_s"])))
                avg_speed = result["avg_speed_kph"]
                if st.session_state.unit_system == "Imperial":
                    metrics.append(("Avg. speed limit", f"{avg_speed / MPH_TO_KPH:.0f} mph"))
                else:
                    metrics.append(("Avg. speed limit", f"{avg_speed:.0f} km/h"))
            walking_time_s = _walking_time_s(result["dist_m"], st.session_state.walking_speed_mph)
            metrics.append(("Walking time", _format_duration(walking_time_s)))

            if st.session_state.mobile_view:
                # st.metric's font size isn't configurable and up to 5 of them
                # wrapped onto multiple rows pushes everything below the fold on a
                # phone. Small HTML chips in one scroll-free horizontal row instead.
                chips = "".join(
                    '<div style="background:rgba(128,128,128,0.15); border-radius:6px; '
                    'padding:5px 12px; flex:0 0 auto;">'
                    f'<div style="font-size:16px; opacity:0.7; line-height:1.2; white-space:nowrap;">{label}</div>'
                    f'<div style="font-size:20px; font-weight:600; line-height:1.3; white-space:nowrap;">{val}</div>'
                    "</div>"
                    for label, val in metrics
                )
                st.markdown(
                    f'<div style="display:flex; gap:6px; overflow-x:auto; padding-bottom:2px;">{chips}</div>',
                    unsafe_allow_html=True,
                )
            else:
                for col, (label, val) in zip(st.columns(len(metrics)), metrics):
                    col.metric(label, val)

            if result["unknown_edges"] > 0:
                st.caption(
                    f"⚠️ {result['unknown_edges']} of {result['total_route_edges']} "
                    "segments on this route had no speed limit — the default speed "
                    "was assumed for those."
                )

        if map_data and map_data.get("last_clicked"):
            raw = map_data["last_clicked"]
            click_key = (round(raw["lat"], 6), round(raw["lng"], 6))
            if click_key != st.session_state.last_click:
                st.session_state.last_click = click_key
                if is_live:
                    _set_point(1, raw["lat"], raw["lng"])
                else:
                    _handle_click(raw["lat"], raw["lng"])
                # scope="fragment" is only valid during an actual fragment rerun; if this
                # code is executing as part of a full-script rerun (e.g. triggered by a
                # sidebar widget elsewhere), fall back to a full rerun instead of crashing.
                try:
                    st.rerun(scope="fragment")
                except StreamlitAPIException:
                    st.rerun()

    _map_section()
