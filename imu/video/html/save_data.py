from imu.io import writeTxt, arrToStr

def saveShotDetection(filename, shot_start=[0,10], shot_label=[1,0]):
    assert len(shot_start) == len(shot_label)
    out = 'var shot_start_str="%s";' % arrToStr(shot_start)
    out += 'var shot_selection_str="%s";' % arrToStr(shot_label)
    writeTxt(filename, out)

