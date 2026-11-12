import unittest

import analyze


class PacingCurveTests(unittest.TestCase):
    def test_counts_cuts_per_segment(self):
        # 10 s video: scene starts at 0, 0.5 (opening), 3 (plateau), 6.5 (re-hook), 8 (closing)
        scenes = [(0, 0.5), (0.5, 3), (3, 6.5), (6.5, 8), (8, 10)]
        curve = {c["segment"]: c["cuts"] for c in analyze.pacing_curve(scenes, 10)}
        self.assertEqual(curve["opening 0-8%"], 2)
        self.assertEqual(curve["plateau 8-60%"], 1)
        self.assertEqual(curve["re-hook 60-70%"], 1)
        self.assertEqual(curve["closing 70-100%"], 1)

    def test_rate_is_per_second_of_segment(self):
        curve = analyze.pacing_curve([(0, 1)], 10)
        opening = curve[0]
        self.assertEqual(opening["cuts"], 1)
        self.assertAlmostEqual(opening["cuts_per_s"], round(1 / 0.8, 2))


class SlugTests(unittest.TestCase):
    def test_slug_is_filesystem_safe(self):
        self.assertEqual(analyze.slug("My Video #1 (final)!"), "my-video-1-final")

    def test_slug_falls_back_when_empty(self):
        self.assertEqual(analyze.slug("!!!"), "ref")

    def test_slug_is_capped(self):
        self.assertLessEqual(len(analyze.slug("a" * 200)), 60)


if __name__ == "__main__":
    unittest.main()
