import numpy as np

def get_data(pred_type):
    au_data = np.load('data/aus.npy').transpose(0, 2, 1)
    em_data = np.load('data/ems.npy').transpose(0, 2, 1)
    hp_data = np.load('data/hps.npy').transpose(0, 2, 1)

    bp_data = np.load('data/bps.npy')
    bp_data = bp_data.reshape(bp_data.shape[0], bp_data.shape[1], -1).transpose(0, 2, 1)

    combine_data =np.column_stack((au_data, em_data, hp_data, bp_data))    # person_num * seq_num * seq_len(300)

    quality_label = np.load('data/quality.npy', allow_pickle=True)
    quality_label = np.array(quality_label, dtype=int)
    ra_label = np.load('data/ra.npy', allow_pickle=True)
    ra_label = np.array(ra_label, dtype=int)
    readiness_label = np.load('data/readiness.npy', allow_pickle=True)
    readiness_label = np.array(readiness_label, dtype=int)

    if pred_type == 'quality':
        combine_label = quality_label
    elif pred_type == 'ra':
        combine_label = ra_label
    else:
        combine_label = readiness_label

    return combine_data, combine_label


if __name__ == '__main__':
    label_type = 'quality'
    get_data(label_type)