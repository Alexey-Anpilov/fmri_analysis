
sub_info_hc = {
'sub-01' : {'slice_num' : 546, 'tr' : 1.0},
'sub-02' : {'slice_num' : 553, 'tr' : 1.0},
'sub-03' : {'slice_num' : 457, 'tr' : 1.0},
'sub-04' : {'slice_num' : 492, 'tr' : 1.0},
'sub-05' : {'slice_num' : 929, 'tr' : 1.0},
'sub-06' : {'slice_num' : 478, 'tr' : 1.0},
'sub-07' : {'slice_num' : 546, 'tr' : 1.0},
'sub-08' : {'slice_num' : 468, 'tr' : 1.11},
'sub-09' : {'slice_num' : 433, 'tr' : 1.11},
'sub-10' : {'slice_num' : 462, 'tr' : 1.11},
'sub-11' : {'slice_num' : 441, 'tr' : 1.11},
'sub-12' : {'slice_num' : 451, 'tr' : 1.11},
'sub-13' : {'slice_num' : 416, 'tr' : 1.11},
'sub-14' : {'slice_num' : 419, 'tr' : 1.11},
'sub-15' : {'slice_num' : 420, 'tr' : 1.11},
'sub-16' : {'slice_num' : 453, 'tr' : 1.11},
'sub-17' : {'slice_num' : 404, 'tr' : 1.11},
'sub-18' : {'slice_num' : 471, 'tr' : 1.11}
}


sub_info_schz = {
'sub-01' : { 'slice_num' : 423, 'tr' : 1.11},
'sub-02' : { 'slice_num' : 524, 'tr' : 1.0},
'sub-03' : { 'slice_num' : 618, 'tr' : 1.0},
'sub-04' : { 'slice_num' : 502, 'tr' : 1.0},
'sub-05' : { 'slice_num' : 485, 'tr' : 1.0},
'sub-06' : { 'slice_num' : 531, 'tr' : 1.0},
'sub-07' : { 'slice_num' : 551, 'tr' : 1.0},
'sub-08' : { 'slice_num' : 491, 'tr' : 1.0},
'sub-09' : { 'slice_num' : 744, 'tr' : 1.1},
'sub-10' : { 'slice_num' : 399, 'tr' : 1.1},
'sub-11' : { 'slice_num' : 513, 'tr' : 1.1},
'sub-12' : { 'slice_num' : 472, 'tr' : 1.1},
'sub-13' : { 'slice_num' : 409, 'tr' : 1.1},
'sub-14' : { 'slice_num' : 426, 'tr' : 1.1},
'sub-15' : { 'slice_num' : 472, 'tr' : 1.0},
'sub-16' : { 'slice_num' : 431, 'tr' : 1.11},
'sub-17' : { 'slice_num' : 441, 'tr' : 1.11},
'sub-18' : { 'slice_num' : 495, 'tr' : 1.11},
'sub-19' : { 'slice_num' : 449, 'tr' : 1.11},
}


black_list_hc = ['sub-01', 'sub-07']
black_list_schz = ['sub-01', 'sub-17', 'sub-16', 'sub-15']

atlas_path = './atlas/atlas_resample.nii'
data_file = 'denoised_data.nii.gz'
events_file = "time_file.csv"

data_dir_hc = './data/HC'
out_dir_hc = './numpy_data/HC/'

data_dir_schz = './data/SCHZ/'
out_dir_schz = './numpy_data/SCHZ/'

is_hc = True

if is_hc:
    sub_info = sub_info_hc
    black_list = black_list_hc
    data_dir = data_dir_hc
    out_dir = out_dir_hc
else:
    sub_info = sub_info_schz
    black_list = black_list_schz
    data_dir = data_dir_schz
    out_dir = out_dir_schz
