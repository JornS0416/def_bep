def normalize_translation(trans, axis=2, offset=-2.5):
    trans = trans.copy()
    trans[:, axis] -= trans[0, axis]
    trans[:, axis] += offset
    return trans