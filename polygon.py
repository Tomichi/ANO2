class Polygon:
    def __init__(self, points):
        self._points = [] + points
        self._occupy = False

    def add_point(self, point):
        self._points.append(point)

    @property
    def points(self):
        return self._points

    @property
    def occupy(self):
        return self._occupy

