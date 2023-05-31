import scipy.io as scio
import os
import nibabel as nb
import numpy as np

THRESHOLD = 0.4

def get_range(data_path):
    img = nb.load(data_path)
    img_data = img.get_fdata()
    img_data[np.isnan(img_data)] = 0
    img_non_zero = np.nonzero(img_data)
    print(len(img_non_zero[0]))
    min_x = min(img_non_zero[0])
    max_x = max(img_non_zero[0])
    min_y = min(img_non_zero[1])
    max_y = max(img_non_zero[1])
    min_z = min(img_non_zero[2])
    max_z = max(img_non_zero[2])
    return min_x, max_x, min_y, max_y, min_z, max_z

def sparse_generate(data_path):
    img = nb.load(data_path)
    img_data = img.get_fdata()
    data_exist = np.zeros(img_data.shape)
    data_exist[~np.isnan(img_data)] = 1
    dimx_sp = np.sum(np.sum(data_exist,axis=0),axis=0)
    dimy_sp = np.sum(np.sum(data_exist,axis=1),axis=1)
    dimz_sp = np.sum(np.sum(data_exist,axis=2),axis=0)
    dimx_thre = int(max(dimx_sp)*THRESHOLD)
    dimy_thre = int(max(dimy_sp)*THRESHOLD)
    dimz_thre = int(max(dimz_sp)*THRESHOLD)
    return dimy_sp>dimy_thre, dimz_sp>dimz_thre, dimx_sp>dimx_thre

def get_sub_data(data_path, y_sp_index, z_sp_index, x_sp_index):
    data = []
    for filename in os.listdir(data_path):
        img = nb.load(data_path + '/' + filename)
        img_affine = img.affine
        img_data = img.get_fdata()
        sp_y = img_data[y_sp_index,:,:]
        sp_yz = sp_y[:,z_sp_index,:]
        spy_yzx = sp_yz[:,:,x_sp_index]
        img_data = spy_yzx
        # img_data = img_data.transpose(2,0,1)
        img_data_vec = img_data[~np.isnan(img_data)]
        print(img_data_vec.shape)
        data.append(img_data_vec)
        nb.Nifti1Image(img_data, img_affine).to_filename('fMRI_data/sub01_processed/'+filename)
        # break
    return data

def get_sub_meta(path, y_sp_index, z_sp_index, x_sp_index):
    img = nb.load(path)
    img_data = img.get_fdata()
    sp_y = img_data[y_sp_index,:,:]
    sp_yz = sp_y[:,z_sp_index,:]
    spy_yzx = sp_yz[:,:,x_sp_index]
    img_data = spy_yzx
    # img_data = img_data.transpose(2,0,1)
    shape = img_data.shape
    nvoxels = np.sum(~np.isnan(img_data))
    indices = np.argwhere(~np.isnan(img_data))
    coord2Col = np.zeros(img_data.shape)
    count = 1
    for induce in indices:
        coord2Col[induce[0]][induce[1]][induce[2]] = count
        count += 1
    meta = {
        'coordToCol': coord2Col,
        'colToCoord': indices+1,
        'nvoxels': nvoxels,
        'dimx': shape[0],
        'dimy': shape[1],
        'dimz': shape[2],
        'ntrials': len(sub_data),
    }
    return meta


categoryIndicesPath = "stimulus/category_indices.mat"
categoryIndices = scio.loadmat(categoryIndicesPath)

for i in range(15):
    i = i+1
    index = i
    if i < 10:
        index = f"0{i}"
    min_x, max_x, min_y, max_y, min_z, max_z = get_range(f"fMRI_data/subj{index}/beta_0002.nii")

    data_path = f"fMRI_data/subj{index}/beta_0001.nii"

    y_sp_index, z_sp_index, x_sp_index = sparse_generate(data_path)
    print(data_path)

    sub_data = get_sub_data(f'fMRI_data/subj{index}', y_sp_index, z_sp_index, x_sp_index)

    sub_meta = get_sub_meta(f"fMRI_data/subj{index}/beta_0002.nii", y_sp_index, z_sp_index, x_sp_index)

    subj01_data = {
        'data': sub_data,
        'meta': sub_meta,
        'info': categoryIndices,
    }

    scio.savemat(f'data_processed/subj{index}-data_threshold0.4.mat', mdict=subj01_data)


