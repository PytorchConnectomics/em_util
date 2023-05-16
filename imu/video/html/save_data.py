from imu.io import write_txt, arr_to_str


def saveShotDetection(filename, shot_start=[0, 10], shot_label=[1, 0]):
    assert len(shot_start) == len(shot_label)
    out = 'var shot_start_str="%s";' % arr_to_str(shot_start)
    out += 'var shot_selection_str="%s";' % arr_to_str(shot_label)
    write_txt(filename, out)
