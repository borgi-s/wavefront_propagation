import configparser
import numpy as np

def init_parameter_dictionary(filename = None):

    params = {}
    params['I/O'] = {}
    params['Beam'] = {}
    params['Material'] = {}
    params['Optics'] = {}
    params['Geometry'] = {}
    params['Status'] = {}

    if filename is not None:
        params['I/O']['filename'] = filename
        par_write(params)
        
    return params

def par_write(params_raw):

    params_printable = configparser.ConfigParser()
    params_printable.optionxform = str
    for field in params_raw.keys():
        params_printable[field] = {}
        for key in params_raw[field].keys():
            value = params_raw[field][key]
            if type(value) == str:
                # write over unmodifield
                params_printable[field][key] = 'string:' + value 
            elif not hasattr(value, "__len__"):
                # Cast to string and pray to the gods
                if type(value) == int:
                    params_printable[field][key] = 'int:' + str(value)
                if type(value) == float or type(value) == np.float64:
                    params_printable[field][key] = 'float:' + str(value)
                if type(value) == complex:
                    params_printable[field][key] = 'complex:' + str(value)
            else:
                # Assume we are dealing with a vector and go from there
                lst = [str(element) for element in value]
                string = ', '.join(lst)
                if type(value[0]) == int or type(value[0]) == np.int64:
                    params_printable[field][key] = 'int vector:'+string
                if type(value[0]) == np.float64 or type(value[0]) == float:
                    params_printable[field][key] = 'float vector:'+string
                    

    if 'filename' in params_raw['I/O']:
        filename = params_raw['I/O']['filename']
        with open(filename, 'w+') as fn:
            params_printable.write(fn)
        
        return 1
    else:
        print("No filename given, cannot save")
        return 0

def par_read(filename):

    params_raw = configparser.ConfigParser()
    params_raw.optionxform = str
    params_raw.read(filename)

    params_parsed = {}

    for field in params_raw.keys():
        params_parsed[field] = {}
        for key in params_raw[field].keys():
            string = params_raw[field][key]
            specifier, value = string.split(':')

            if specifier == 'string':
                params_parsed[field][key] = value
            elif specifier == 'float':
                params_parsed[field][key] = float(value)
            elif specifier == 'int':
                params_parsed[field][key] = int(value)
            elif specifier == 'complex':
                params_parsed[field][key] = complex(value)
            elif specifier.endswith('vector'):
                specifier = specifier.split(' ')[0]
                params_parsed[field][key] = np.fromstring(value, sep = ', ', dtype = specifier)
            else:

                print('Unreadable value in parameter file:')
                # print(specifier)
                # print(value)
    
    return params_parsed




#################### OLD STUFF #######################
# import json

# def init_parameter_dictionary_json(filename = None):

#     params = {'I/O':{}, 'BEAM':{}, 'SAMPLE':{}, 'GEOMETRY':{}, 'OPTICS':{}, 'STATUS':{}}

#     if filename is not None:
#         params['I/O']['filename'] = filename
#         with open(filename, 'w+') as fn:
#             json.dump(params, fn)
        
#     return params
