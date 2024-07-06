import numpy as np
from scipy.spatial import Delaunay
from sklearn.neighbors import KDTree
from PIL import Image, ImageDraw
import math


def db_eval_iou(annotation, segmentation):
    annotation = annotation.astype(np.bool_)
    segmentation = segmentation.astype(np.bool_)

    if np.isclose(np.sum(annotation), 0) and np.isclose(np.sum(segmentation), 0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / \
            np.sum((annotation | segmentation), dtype=np.float32)


def distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def find_nearest_point(current_point, points, visited):
    min_distance = float('inf')
    nearest_point = None
    for point in points:
        if point not in visited:
            dist = distance(current_point, point)
            if dist < min_distance:
                min_distance = dist
                nearest_point = point
    return nearest_point


def order_points(poly_points):
    ordered_points = []
    visited = set()

    start_point = min(poly_points, key=lambda point: point[0])
    current_point = start_point
    ordered_points.append(current_point)
    visited.add(current_point)

    while len(ordered_points) < len(poly_points):
        nearest_point = find_nearest_point(current_point, poly_points, visited)
        ordered_points.append(nearest_point)
        visited.add(nearest_point)
        current_point = nearest_point

    return ordered_points


def plot_circle(centers, rs, ax):
    N = centers.shape[0]
    for i in range(N):
        theta = np.arange(0, 2 * np.pi, 0.01)
        x = centers[i, 0] + rs[i] * np.cos(theta)
        y = centers[i, 1] + rs[i] * np.sin(theta)
        ax.plot(x, y, 'b-', alpha=0.1)


def edge_check_valid(e, tree, r, err):
    xp = e[0]
    xq = e[1]
    L = np.sqrt(np.dot(xq - xp, xq - xp))
    if L > 2 * r:
        return False, -1
    vec = (xq - xp) / L  # the vector from p to q
    normal = np.array([vec[1], -vec[0]])
    c1 = (xp + xq) / 2 + normal * np.sqrt(r ** 2 - (L / 2) ** 2)
    c2 = (xp + xq) / 2 - normal * np.sqrt(r ** 2 - (L / 2) ** 2)
    c = np.array([c1, c2])
    count = tree.query_radius(c, r=r + err, return_distance=False, count_only=True, sort_results=False)
    if count[0] <= 2:
        return True, c[0]
    elif count[1] <= 2:
        return True, c[1]
    else:
        return False, -1


def boundary_extract(points, alpha, err=10e-3):
    R = 1 / alpha
    pts = np.copy(points)
    tree = KDTree(pts, leaf_size=2)
    tri = Delaunay(pts)
    s = tri.simplices
    N = s.shape[0]
    i = 0
    edges = []
    centers = []
    while i <= N - 1:
        if s[i, 0] == -1:
            i = i + 1
            continue
        p3 = s[i]
        e1 = np.array([points[p3[0], :], points[p3[1], :]])
        e2 = np.array([points[p3[1], :], points[p3[2], :]])
        e3 = np.array([points[p3[0], :], points[p3[2], :]])
        e = [e1, e2, e3]
        for j in range(3):
            flag, center = edge_check_valid(e[j], tree, R, err)
            if flag:
                edges.append(e[j])
                centers.append(center)
        nb = tri.neighbors[i]
        i = i + 1
    return edges, centers


def create_picture(points, resolution):
    height, width = resolution
    mask = Image.new('1', (width, height), 0)

    draw = ImageDraw.Draw(mask)

    draw.polygon(points, fill=1)

    return mask


def generate_mask_from_points(pts, resolution):
    alpha = 0.005
    edges, centers = boundary_extract(pts, alpha, err=10e-5)
    vertices = []
    for i in range(len(edges)):
        xy_point = edges[i]
        for j in range(2):
            point_tuple = (xy_point[j][0], xy_point[j][1])
            if point_tuple not in vertices:
                vertices.append(point_tuple)

    vertices = order_points(vertices)

    mask = create_picture(vertices, resolution)
    mask = np.asarray(mask) == 1
    return mask

