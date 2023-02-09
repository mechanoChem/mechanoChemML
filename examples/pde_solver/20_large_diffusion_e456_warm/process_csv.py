import sys
import glob, os

def exe_cmd(cmd, output=False):
    import subprocess, os
    if output:
        output_info = os.popen(cmd).read()
        return output_info
    else:
        os.system(cmd)

# pre_fix = 'e5-stats-r'

file_dict = {}
for file in glob.glob("*ckpt*.csv"):
    # print(file)
    s_ind = file.find('2021')
    file_key = file[s_ind:]
    # print(file_key)
    try:
        a = file_dict[file_key]
        # print('a1:',file_dict[file_key])
        file_dict[file_key].append(file)
        # print('a:',file_key)
        # print('a:',file)
        # print('a:',file_dict[file_key])
    except:
        file_dict[file_key] = []
        file_dict[file_key].append(file)
        # print('b:',file_key)
        # print('b:',file_dict[file_key])

# print(file_dict)
for key, val in file_dict.items():
    # print('key:', key, len(val))
    new_val = []
    for v0 in val:
        # print('--val:', v0)
        if v0.find('part') > 0:
            # print('--val:', v0)
            p_ind = v0.find('part')
            # print(v0[p_ind:])
            tmp_v0 = v0[0:p_ind]+'*'+v0[p_ind+5:]
            # print(tmp_v0)
            new_val.append(tmp_v0)
        else:
            new_val.append(v0)
    new_val = list(set(new_val))
    # print(new_val)
    file_dict[key] = new_val

for key, val in file_dict.items():
    print('key:', key, len(val))
    sim_log = 'sim_log_' + key[:-3] + 'log'
    print(sim_log)
    for v0 in val:
        print('--val:', v0)
        if v0.find('*') >=0:
            file_id = v0.replace('-*-', '-')
        else:
            file_id = v0
        cmd = "echo '" + file_id + "' >> " + sim_log
        print(cmd)
        exe_cmd(cmd)
        cmd = "python ../report_l2_error.py " + v0 + " >> "  + sim_log
        print(cmd)
        exe_cmd(cmd)
