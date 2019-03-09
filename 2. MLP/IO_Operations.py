class Importer:
    def __init__(self):
        pass
    
    @staticmethod
    def import_output(path):
        lines = []

        with open(path, 'r') as file_obj:
            for i, line in enumerate(file_obj):
                line = line.replace("\n", "")
                lines.append(float(line))

        return lines
    
    @staticmethod
    def import_input(path):
        lines = []

        with open(path, 'r') as file_obj:
            for i, line in enumerate(file_obj):
                line = line.replace("\n", "")
                x1,x2 = line.split('\t')
                lines.append([float(x1),float(x2)])

        return lines


class Printer:
    def __init__(self):
        pass

    @staticmethod
    def print_msg(msg):
        print(msg)
        

class PrinterFile:
    path = "output/log"
    def __init__(self):
        pass

    @staticmethod
    def print_msg(msg):
        with open(PrinterFile.path, 'a') as file_obj:
            file_obj.write(msg)
            file_obj.write('\n')
            