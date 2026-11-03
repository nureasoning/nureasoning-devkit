"""
Map adapter for nuReasoning clips.

DiffusionDrive's TransfuserTargetBuilder (navsim/agents/diffusiondrive/
transfuser_features.py) needs a `map_api` object that supports:

    map_api.get_proximal_map_objects(point, radius, layers) -> Dict[SemanticMapLayer, List[MapObject]]

where each MapObject needs a `.polygon` (shapely Polygon) for polygon
layers, and a `.baseline_path.linestring` (shapely LineString) for
linestring layers (LANE, LANE_CONNECTOR).

Verified against an uploaded map.pkl:
  - Coordinates in `polygon` / `geometry` / `centerline` fields are in the
    SAME large-scale planar frame as `mission_goal.route_path` in
    metadata.json (e.g. easting/northing ~664000/3998000 for a Las Vegas
    clip) -- i.e. GLOBAL coordinates, not ego-relative. This matches how
    nuPlan's own AbstractMap stores geometry, so `point` passed in here
    should be the ego's GLOBAL (x, y), exactly as NAVSIM's target builder
    already does (`ego_pose.point` from `scene.frames[i].ego_status.ego_pose`,
    built with `in_global_frame=True`).
  - Real schema (9 categories): lanes, baseline_paths, boundaries,
    crosswalks, intersections, stop_polygons, road_blocks, traffic_lights,
    lane_connectors.
  - There is no WALKWAYS-equivalent category in nuReasoning's map schema.
    That BEV channel will come back empty until/unless a sidewalk source
    is added -- flagged below, don't silently assume it's covered.

This only covers the layers DiffusionDrive's bev_semantic_classes config
actually queries: LANE, INTERSECTION, LANE_CONNECTOR, WALKWAYS (empty).
Extending to CROSSWALK/STOP_LINE etc. is straightforward -- the raw
objects are already parsed below, just add them to `_layer_objects`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from shapely.geometry import LineString, Point, Polygon

from nureasoning.common.pickle_io import load_pickle

try:
    from nuplan.common.maps.abstract_map import SemanticMapLayer
except ImportError:
    # Fallback ONLY for environments without nuplan-devkit installed (e.g.
    # inspecting/testing this adapter standalone). At actual training time
    # the real navsim/nuplan SemanticMapLayer enum will be importable and
    # is used automatically instead of this stand-in.
    from enum import Enum

    class SemanticMapLayer(Enum):
        LANE = "LANE"
        INTERSECTION = "INTERSECTION"
        LANE_CONNECTOR = "LANE_CONNECTOR"
        WALKWAYS = "WALKWAYS"
        CROSSWALK = "CROSSWALK"


def _drop_z(points: Sequence[Sequence[float]]) -> List[tuple]:
    return [(p[0], p[1]) for p in points]


def _safe_polygon(points: Sequence[Sequence[float]]):
    coords = _drop_z(points)
    if len(coords) < 3:
        return None
    try:
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)  # common fix for self-intersecting rings
        return poly
    except Exception:
        return None


def _safe_linestring(points: Sequence[Sequence[float]]):
    coords = _drop_z(points)
    if len(coords) < 2:
        return None
    try:
        return LineString(coords)
    except Exception:
        return None


@dataclass
class _BaselinePathRef:
    linestring: Any  # shapely LineString


@dataclass
class MapObjectAdapter:
    """Minimal stand-in for nuPlan's MapObject, sized to what
    TransfuserTargetBuilder actually reads: `.polygon` and, for
    LANE / LANE_CONNECTOR layers, `.baseline_path.linestring`."""

    id: str
    polygon: Any = None  # shapely Polygon or None
    baseline_path: _BaselinePathRef = None  # only set for lane-like objects


class NuReasoningMapAdapter:
    """
    Wraps a nuReasoning `map.pkl` and exposes the subset of nuPlan's
    AbstractMap interface that DiffusionDrive's feature/target builders
    call, backed by the already-loaded pickle instead of a live GIS query.
    """

    def __init__(self, map_pkl_path: str):
        raw_map = load_pickle(map_pkl_path)

        self._lane_objects: List[MapObjectAdapter] = []
        for lane in getattr(raw_map, "lanes", []):
            polygon = _safe_polygon(lane.polygon)
            centerline = _safe_linestring(getattr(lane, "centerline", []) or [])
            baseline_path = _BaselinePathRef(linestring=centerline) if centerline is not None else None
            self._lane_objects.append(
                MapObjectAdapter(id=str(lane.id), polygon=polygon, baseline_path=baseline_path)
            )

        # Prefer explicit baseline_paths (matched by lane_id) over a lane's
        # own centerline where available -- baseline paths are usually the
        # actual drivable reference line, centerline is a geometric fallback.
        baseline_by_lane_id: Dict[Any, Any] = {}
        for bp in getattr(raw_map, "baseline_paths", []):
            ls = _safe_linestring(bp.geometry)
            if ls is not None and getattr(bp, "lane_id", None) is not None:
                baseline_by_lane_id[bp.lane_id] = ls
        for lane, obj in zip(getattr(raw_map, "lanes", []), self._lane_objects):
            if lane.id in baseline_by_lane_id:
                obj.baseline_path = _BaselinePathRef(linestring=baseline_by_lane_id[lane.id])

        self._intersection_objects: List[MapObjectAdapter] = []
        for inter in getattr(raw_map, "intersections", []):
            polygon = _safe_polygon(inter.geometry)
            self._intersection_objects.append(MapObjectAdapter(id=str(inter.id), polygon=polygon))

        self._lane_connector_objects: List[MapObjectAdapter] = []
        for lc in getattr(raw_map, "lane_connectors", []):
            # lane connectors here are linestrings (connector geometry), not
            # polygons -- only used for the "centerline" linestring layer.
            ls = _safe_linestring(lc.geometry)
            baseline_path = _BaselinePathRef(linestring=ls) if ls is not None else None
            self._lane_connector_objects.append(
                MapObjectAdapter(id=str(lc.id), polygon=None, baseline_path=baseline_path)
            )

        self._crosswalk_objects: List[MapObjectAdapter] = []
        for cw in getattr(raw_map, "crosswalks", []):
            polygon = _safe_polygon(cw.geometry)
            self._crosswalk_objects.append(MapObjectAdapter(id=str(cw.id), polygon=polygon))

        self._layer_objects: Dict[Any, List[MapObjectAdapter]] = {
            # baseline_path is required in addition to polygon -- LANE and
            # LANE_CONNECTOR are rendered via baseline_path.linestring, which
            # crashes on None if we don't filter it out here (some lanes have
            # a polygon but no matching centerline/baseline_paths entry).
            SemanticMapLayer.LANE: [o for o in self._lane_objects if o.polygon is not None and o.baseline_path is not None],
            SemanticMapLayer.INTERSECTION: [o for o in self._intersection_objects if o.polygon is not None],
            SemanticMapLayer.LANE_CONNECTOR: [o for o in self._lane_connector_objects if o.baseline_path is not None],
            SemanticMapLayer.CROSSWALK: [o for o in self._crosswalk_objects if o.polygon is not None],
            SemanticMapLayer.WALKWAYS: [],
        }

    def get_proximal_map_objects(
        self, point: Any, radius: float, layers: Sequence[Any]
    ) -> Dict[Any, List[MapObjectAdapter]]:
        """
        :param point: object with .x / .y (e.g. StateSE2.point), in the
            SAME global planar frame as the map.pkl geometries.
        :param radius: query radius in meters.
        :param layers: list of SemanticMapLayer values to query.
        :return: dict of layer -> list of MapObjectAdapter within radius.
        """
        center = Point(point.x, point.y)
        result: Dict[Any, List[MapObjectAdapter]] = {}
        for layer in layers:
            candidates = self._layer_objects.get(layer, [])
            nearby = []
            for obj in candidates:
                geom = obj.polygon if obj.polygon is not None else obj.baseline_path.linestring
                if geom is not None and geom.distance(center) <= radius:
                    nearby.append(obj)
            result[layer] = nearby
        return result

    def sanity_check_alignment(self, ego_global_xy: Sequence[float], expect_within_m: float = 100.0) -> bool:
        """
        Quick coordinate-frame sanity check: confirms the ego's global
        position actually falls near the map content (catches the case
        where ego_state.pkl turns out to use a different frame/units than
        map.pkl). Call this once when you first wire up real ego_state
        data, before trusting any cached BEV targets.
        """
        center = Point(ego_global_xy[0], ego_global_xy[1])
        all_polys = [o.polygon for o in self._lane_objects if o.polygon is not None]
        if not all_polys:
            return False
        min_dist = min(p.distance(center) for p in all_polys)
        return min_dist <= expect_within_m


if __name__ == "__main__":
    import sys

    adapter = NuReasoningMapAdapter(sys.argv[1])
    print(f"Parsed lanes: {len(adapter._lane_objects)}")
    print(f"Parsed intersections: {len(adapter._intersection_objects)}")
    print(f"Parsed lane_connectors: {len(adapter._lane_connector_objects)}")
    print(f"Parsed crosswalks: {len(adapter._crosswalk_objects)}")

    # Query around the centroid of the first lane polygon as a smoke test.
    first_lane_poly = next(o.polygon for o in adapter._lane_objects if o.polygon is not None)
    centroid = first_lane_poly.centroid
    result = adapter.get_proximal_map_objects(
        point=centroid,
        radius=50.0,
        layers=list(adapter._layer_objects.keys()),
    )
    for layer, objs in result.items():
        print(f"  {layer}: {len(objs)} objects within 50m of lane centroid")