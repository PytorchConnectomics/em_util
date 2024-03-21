""" Test functions for html visualization generation """

import sys
from em_util.video import *
from em_util.io import writeTxt


def test_shot_detection():
    fn = "../biomed/umb/"
    out = html_shot(
        frame_name="im_every1/%04d.png",
        file_result="./shot.js",
        frame_num=2432,
        frame_fps=1,
    )
    writeTxt(f"{fn}shot_detection.html", out.getHtml())


if __name__ == "__main__":
    # python tests/test_html.py 0
    opt = sys.argv[1]
    if opt == "0":
        test_shot_detection()
