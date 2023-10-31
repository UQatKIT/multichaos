"""
Utility functions.
"""
import alphashape
import numpy as np

from shapely.geometry import MultiPolygon


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la, order="C")

def mse(x, y):
    return ((x - y) ** 2).mean()

def lower_envelope(points: np.array, alpha: float=1.) -> np.array:
    hull = alphashape.alphashape(points, alpha=alpha)
    if isinstance(hull, MultiPolygon):
        pts = []
        for polygon in hull.geoms:
            pts.append(np.array(polygon.exterior.coords.xy).T)
        hull_pts = np.vstack(pts)
    else:
        hull_pts = np.array(hull.exterior.coords.xy).T
    s = np.argmin(hull_pts[:, 1])
    e = np.argmax(hull_pts[:, 1])
    if s > e:
        hull_pts = np.roll(hull_pts, hull_pts.shape[0] - s, axis=0)
        lower_envelope = hull_pts[:hull_pts.shape[0] - s + e + 1]
    else:
        lower_envelope = hull_pts[s:e + 1]
    return lower_envelope

def make_array_strictly_increasing(arr: np.array) -> np.array:
    for i in range(1, len(arr)):
        if arr[i-1] >= arr[i]:
            arr[i] = arr[i-1] + 1
    return arr

def unit_to_domain(x: np.array, domain: list[tuple]) -> np.array:
    """
    Maps `x` from unit hypercube to the product domain
    given by the intervals in `domain`.
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)

    a = np.array([a for a, _ in domain])
    b = np.array([b for _, b in domain])

    x = (b - a) * x + a
    return x.squeeze()


def domain_to_unit(x: np.array, domain: list[tuple]) -> np.array:
    """
    Maps `x` from the product domain given by the intervals
    in `domain` to unit hypercube.
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)

    a = np.array([a for a, _ in domain])
    b = np.array([b for _, b in domain])

    x_ = (x - a) / (b - a)
    return x_.squeeze()