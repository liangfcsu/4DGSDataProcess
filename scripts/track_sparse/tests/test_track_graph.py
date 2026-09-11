from __future__ import annotations

import unittest

from track_sparse.track_graph import ConstrainedComponents


class TrackGraphTests(unittest.TestCase):
    def test_same_camera_conflict_is_rejected(self):
        graph = ConstrainedComponents()
        self.assertTrue(graph.merge((0, 1), (1, 2)))
        self.assertTrue(graph.merge((1, 2), (2, 3)))
        self.assertFalse(graph.merge((2, 3), (0, 9)))
        groups = graph.groups()
        self.assertTrue(any({(0, 1), (1, 2), (2, 3)} == group for group in groups))
        self.assertTrue(any({(0, 9)} == group for group in groups))


if __name__ == "__main__":
    unittest.main()

