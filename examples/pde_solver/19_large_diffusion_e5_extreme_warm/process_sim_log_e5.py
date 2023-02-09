import sys

def read_file(path):
  try:
    f = open(path, 'r')
    content = f.readlines()
    f.close()
  except OSError as err:
      print("OS error: {0}".format(err))
  except:
      print("Unexpected error:", sys.exc_info()[0])
      raise
  return content

if len(sys.argv) <= 1:
    print("usage: python ", sys.argv[0], " sim*.log")

one_dict = {
        'file_id':'-1',
        'pickle_file':'',
        'e5-all-mean':'',
        'e5-all-std':'',
        'e5-w-Neu-mean':'',
        'e5-w-Neu-std':'',
        'e5-wo-Neu-mean':'',
        'e5-wo-Neu-std':'',
        'e5-regular-all-mean':'',
        'e5-regular-all-std':'',
        'e5-regular-w-Neu-mean':'',
        'e5-regular-w-Neu-std':'',
        'e5-regular-wo-Neu-mean':'',
        'e5-regular-wo-Neu-std':'',
        'e5-extreme-all-mean':'',
        'e5-extreme-all-std':'',
        'e5-extreme-w-Neu-mean':'',
        'e5-extreme-w-Neu-std':'',
        'e5-extreme-wo-Neu-mean':'',
        'e5-extreme-wo-Neu-std':'',
        'e4-32k-all-mean':'',
        'e4-32k-all-std':'',
        'e4-32k-w-Neu-mean':'',
        'e4-32k-w-Neu-std':'',
        'e4-32k-wo-Neu-mean':'',
        'e4-32k-wo-Neu-std':'',
        'e5-64k-all-mean':'',
        'e5-64k-all-std':'',
        'e5-64k-w-Neu-mean':'',
        'e5-64k-w-Neu-std':'',
        'e5-64k-wo-Neu-mean':'',
        'e5-64k-wo-Neu-std':'',
        'e6-96k-all-mean':'',
        'e6-96k-all-std':'',
        'e6-96k-w-Neu-mean':'',
        'e6-96k-w-Neu-std':'',
        'e6-96k-wo-Neu-mean':'',
        'e6-96k-wo-Neu-std':'',
        'e9-96k-all-mean':'',
        'e9-96k-all-std':'',
        'e9-96k-w-Neu-mean':'',
        'e9-96k-w-Neu-std':'',
        'e9-96k-wo-Neu-mean':'',
        'e9-96k-wo-Neu-std':'',
        's4-all-mean':'',
        's4-all-std':'',
        's4-w-Neu-mean':'',
        's4-w-Neu-std':'',
        's4-wo-Neu-mean':'',
        's4-wo-Neu-std':'',
        's5-all-mean':'',
        's5-all-std':'',
        's5-w-Neu-mean':'',
        's5-w-Neu-std':'',
        's5-wo-Neu-mean':'',
        's5-wo-Neu-std':'',
        's6-all-mean':'',
        's6-all-std':'',
        's6-w-Neu-mean':'',
        's6-w-Neu-std':'',
        's6-wo-Neu-mean':'',
        's6-wo-Neu-std':'',
        's7-all-mean':'',
        's7-all-std':'',
        's7-w-Neu-mean':'',
        's7-w-Neu-std':'',
        's7-wo-Neu-mean':'',
        's7-wo-Neu-std':'',
        's8-all-mean':'',
        's8-all-std':'',
        's8-w-Neu-mean':'',
        's8-w-Neu-std':'',
        's8-wo-Neu-mean':'',
        's8-wo-Neu-std':'',
        's9-all-mean':'',
        's9-all-std':'',
        's9-w-Neu-mean':'',
        's9-w-Neu-std':'',
        's9-wo-Neu-mean':'',
        's9-wo-Neu-std':'',
        's10-all-mean':'',
        's10-all-std':'',
        's10-w-Neu-mean':'',
        's10-w-Neu-std':'',
        's10-wo-Neu-mean':'',
        's10-wo-Neu-std':'',
        's11-all-mean':'',
        's11-all-std':'',
        's11-w-Neu-mean':'',
        's11-w-Neu-std':'',
        's11-wo-Neu-mean':'',
        's11-wo-Neu-std':'',
        's12-all-mean':'',
        's12-all-std':'',
        's12-w-Neu-mean':'',
        's12-w-Neu-std':'',
        's12-wo-Neu-mean':'',
        's12-wo-Neu-std':'',
        }
dict_data = []
file_id = 0
for f0 in sys.argv[1:]:
    _dict = one_dict.copy()
    _dict['file_id'] = str(file_id)
    # if f0.find('-e456-') > 0:
    # if f0.find('-e4569-') > 0:
    # if f0.find('-pentagon-') > 0:
    file_id += 1
    if True :
        # print('f0:', f0)
        file_prefix = f0.replace('sim_log_','')
        file_prefix = file_prefix.replace('.pickle.log','.pickle')
        print('process file: ', file_prefix)
        _dict['pickle_file'] = file_prefix

        file_content = read_file(f0)
        csv_filename = ''
        for i0 in range(len(file_content)):
            l0 = file_content[i0].strip()
            if (l0.find('.csv') >= 0):
                csv_filename = l0.replace('-'+file_prefix+'.csv','')
                print('cvs filename: ', csv_filename)
            if (l0[0:4] == 'mean'):
                l_old = file_content[i0-1].strip()
                l_new = file_content[i0+1].strip()
                # print(l_old.split(":")[0], l0, l_new)
                def assign_value(prefix):
                    # print(l_old.split(), l0.split(), l_new.split())
                    if l_old.split()[0] == 'all:':
                        _dict[prefix+'-all-mean'] = l0.split()[1]
                        _dict[prefix+'-all-std']  = l_new.split()[1]
                    if l_old.split()[0] == 'with':
                        _dict[prefix+'-w-Neu-mean'] = l0.split()[1]
                        _dict[prefix+'-w-Neu-std']  = l_new.split()[1]
                    if l_old.split()[0] == 'without':
                        _dict[prefix+'-wo-Neu-mean'] = l0.split()[1]
                        _dict[prefix+'-wo-Neu-std']  = l_new.split()[1]

                if csv_filename == 'e5-stats': assign_value(prefix='e5')
                if csv_filename == 'e5-stats-regular': assign_value(prefix='e5-regular')
                if csv_filename == 'e5-stats-extreme': assign_value(prefix='e5-extreme')

                if csv_filename == 'e456-stats-large-64x64-e4-32k': assign_value(prefix='e4-32k')
                if csv_filename == 'e456-stats-large-64x64-e5-64k': assign_value(prefix='e5-64k')
                if csv_filename == 'e456-stats-large-64x64-e6-96k': assign_value(prefix='e6-96k')
                if csv_filename == 'e456-stats-large-64x64-e9-96k': assign_value(prefix='e9-96k')
                if csv_filename == 'e456-stats-s4':  assign_value(prefix='s4')
                if csv_filename == 'e456-stats-s5':  assign_value(prefix='s5')
                if csv_filename == 'e456-stats-s6':  assign_value(prefix='s6')
                if csv_filename == 'e456-stats-s7':  assign_value(prefix='s7')
                if csv_filename == 'e456-stats-s8':  assign_value(prefix='s8')
                if csv_filename == 'e456-stats-s9':  assign_value(prefix='s9')
                if csv_filename == 'e456-stats-s10': assign_value(prefix='s10')
                if csv_filename == 'e456-stats-s11': assign_value(prefix='s11')
                if csv_filename == 'e456-stats-s12': assign_value(prefix='s12')

                if csv_filename == 'e4569-stats-large-64x64-e4-32k': assign_value(prefix='e4-32k')
                if csv_filename == 'e4569-stats-large-64x64-e5-64k': assign_value(prefix='e5-64k')
                if csv_filename == 'e4569-stats-large-64x64-e6-96k': assign_value(prefix='e6-96k')
                if csv_filename == 'e4569-stats-large-64x64-e9-96k': assign_value(prefix='e9-96k')
                if csv_filename == 'e4569-stats-s4':  assign_value(prefix='s4')
                if csv_filename == 'e4569-stats-s5':  assign_value(prefix='s5')
                if csv_filename == 'e4569-stats-s6':  assign_value(prefix='s6')
                if csv_filename == 'e4569-stats-s7':  assign_value(prefix='s7')
                if csv_filename == 'e4569-stats-s8':  assign_value(prefix='s8')
                if csv_filename == 'e4569-stats-s9':  assign_value(prefix='s9')
                if csv_filename == 'e4569-stats-s10': assign_value(prefix='s10')
                if csv_filename == 'e4569-stats-s11': assign_value(prefix='s11')
                if csv_filename == 'e4569-stats-s12': assign_value(prefix='s12')

        dict_data.append(_dict)
        del _dict

import csv
csv_columns = dict_data[0].keys()
csv_file = "all_summary.csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in dict_data:
            writer.writerow(data)
except IOError:
    print("I/O error")
