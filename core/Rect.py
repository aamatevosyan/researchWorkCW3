from typing import List


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Rect(object):
    def __init__(self, p1, p2):
        """Store the top, bottom, left and right values for points
               p1 and p2 are the (corners) in either order
        """
        self.left = min(p1.x, p2.x)
        self.right = max(p1.x, p2.x)
        self.bottom = min(p1.y, p2.y)
        self.top = max(p1.y, p2.y)

    def __str__(self):
        return "Rect[%d, %d, %d, %d]" % (self.left, self.top, self.right, self.bottom)


def range_overlap(a_min, a_max, b_min, b_max):
    """Neither range is completely greater than the other
    """
    return (a_min <= b_max) and (b_min <= a_max)


def rect_overlaps(r1, r2):
    return range_overlap(r1.left, r1.right, r2.left, r2.right) and range_overlap(r1.bottom, r1.top, r2.bottom, r2.top)


def get_overlap_clusters(rectangles: List[Rect]):
    clusters = []

    for rect in rectangles:
        matched = 0
        for cluster in clusters:
            if rect_overlaps(rect, cluster):
                matched = 1
                cluster.left = min(cluster.left, rect.left, cluster.right, rect.right)
                cluster.right = max(cluster.left, rect.left, cluster.right, rect.right)
                cluster.top = max(cluster.top, rect.top, cluster.bottom, rect.bottom)
                cluster.bottom = min(cluster.top, rect.top, cluster.bottom, rect.bottom)

        if not matched:
            clusters.append(rect)

    return clusters
