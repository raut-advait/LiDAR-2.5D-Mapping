"""
Real Dynamic Object Tracker for Project Balerion.

Extracts real PointNet++ dynamic object predictions (labels == 2), performs
spatial clustering in BEV coordinates (+X forward, +Y left), and tracks detected
clusters across consecutive frames to estimate real velocity and TTC.
"""

import numpy as np
from sklearn.cluster import DBSCAN


class RealDynamicObjectTracker:
    """
    Real-time dynamic object clustering and tracking for PointNet++ predictions.
    Detects dynamic clusters where labels == 2 within BEV range.
    Tracks clusters across consecutive frames to estimate real velocity and TTC.
    """

    def __init__(
        self,
        eps: float = 1.2,
        min_samples: int = 8,
        max_cluster_dim: float = 12.0,
        max_dist_association: float = 3.5,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.max_cluster_dim = max_cluster_dim
        self.max_dist_association = max_dist_association
        self.prev_clusters = []

    def reset(self):
        """Resets tracking history."""
        self.prev_clusters = []

    def process_frame(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        dt: float = 0.5,
        bev_bounds: tuple = (-15.0, 65.0, -40.0, 40.0),
    ) -> list:
        """
        Extracts dynamic points (labels == 2), performs XY spatial clustering,
        and associates clusters with previous frame to estimate velocity.

        Returns list of dicts:
        [
            {
                "id": "DYN-01",
                "centroid": np.array([cx, cy]),
                "min_x": float, "max_x": float,
                "min_y": float, "max_y": float,
                "vx": float, "vy": float,
                "has_track": bool,
                "point_count": int,
                "distance": float,
            },
            ...
        ]
        """
        if points is None or labels is None or len(points) == 0:
            self.prev_clusters = []
            return []

        dynamic_mask = (labels == 2)
        x_min_b, x_max_b, y_min_b, y_max_b = bev_bounds
        bev_mask = (
            (points[:, 0] >= x_min_b) & (points[:, 0] <= x_max_b) &
            (points[:, 1] >= y_min_b) & (points[:, 1] <= y_max_b)
        )
        dyn_pts = points[dynamic_mask & bev_mask]

        if len(dyn_pts) < self.min_samples:
            self.prev_clusters = []
            return []

        xy = dyn_pts[:, :2]
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(xy)

        cluster_labels = db.labels_
        unique_cids = [c for c in set(cluster_labels) if c != -1]

        raw_clusters = []
        for cid in unique_cids:
            c_pts = xy[cluster_labels == cid]
            min_x, max_x = float(c_pts[:, 0].min()), float(c_pts[:, 0].max())
            min_y, max_y = float(c_pts[:, 1].min()), float(c_pts[:, 1].max())
            dx = max_x - min_x
            dy = max_y - min_y

            # Filter out giant background component noise or sub-cluster them
            if dx > self.max_cluster_dim or dy > self.max_cluster_dim:
                # Secondary tighter DBSCAN pass for dense cores
                sub_db = DBSCAN(eps=0.6, min_samples=6).fit(c_pts)
                sub_labels = sub_db.labels_
                for scid in set(sub_labels):
                    if scid == -1:
                        continue
                    sc_pts = c_pts[sub_labels == scid]
                    s_min_x, s_max_x = float(sc_pts[:, 0].min()), float(sc_pts[:, 0].max())
                    s_min_y, s_max_y = float(sc_pts[:, 1].min()), float(sc_pts[:, 1].max())
                    s_dx, s_dy = s_max_x - s_min_x, s_max_y - s_min_y
                    if s_dx <= self.max_cluster_dim and s_dy <= self.max_cluster_dim and len(sc_pts) >= self.min_samples:
                        raw_clusters.append({
                            "centroid": sc_pts.mean(axis=0),
                            "min_x": s_min_x, "max_x": s_max_x,
                            "min_y": s_min_y, "max_y": s_max_y,
                            "point_count": len(sc_pts),
                        })
            else:
                raw_clusters.append({
                    "centroid": c_pts.mean(axis=0),
                    "min_x": min_x, "max_x": max_x,
                    "min_y": min_y, "max_y": max_y,
                    "point_count": len(c_pts),
                })

        if not raw_clusters:
            self.prev_clusters = []
            return []

        # Sort clusters by point count descending
        raw_clusters.sort(key=lambda c: c["point_count"], reverse=True)

        tracked_clusters = []
        used_prev = set()

        for idx, cl in enumerate(raw_clusters):
            c_curr = cl["centroid"]
            best_prev_idx = None
            best_dist = self.max_dist_association

            for p_idx, pcl in enumerate(self.prev_clusters):
                if p_idx in used_prev:
                    continue
                d = float(np.linalg.norm(c_curr - pcl["centroid"]))
                if d < best_dist:
                    best_dist = d
                    best_prev_idx = p_idx

            if best_prev_idx is not None and dt > 0:
                used_prev.add(best_prev_idx)
                pcl = self.prev_clusters[best_prev_idx]
                vx = float((c_curr[0] - pcl["centroid"][0]) / dt)
                vy = float((c_curr[1] - pcl["centroid"][1]) / dt)
                has_track = True
                track_id = pcl.get("id", f"DYN-{idx+1:02d}")
            else:
                vx = 0.0
                vy = 0.0
                has_track = False
                track_id = f"DYN-{idx+1:02d}"

            dist_to_ego = float(np.hypot(c_curr[0], c_curr[1]))

            cl_info = {
                "id": track_id,
                "centroid": c_curr,
                "vx": vx,
                "vy": vy,
                "has_track": has_track,
                "min_x": cl["min_x"], "max_x": cl["max_x"],
                "min_y": cl["min_y"], "max_y": cl["max_y"],
                "point_count": cl["point_count"],
                "distance": dist_to_ego,
            }
            tracked_clusters.append(cl_info)

        self.prev_clusters = tracked_clusters
        return tracked_clusters
