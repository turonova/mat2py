def string_path_complete(input_path):
    if input_path == '':
        return './'
    elif input_path[-1] == '/':
        return input_path
    else:
        return input_path + '/'
