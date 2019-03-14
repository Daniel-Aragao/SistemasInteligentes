from os import path, makedirs
import codecs

if not path.isdir("output/"):
        makedirs("output/")

if not path.isdir("output/log"):
        makedirs("output/log")

if not path.isdir("output/img"):
        makedirs("output/img")

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

class Exporter:
    result_file_path = "output/log/epocha_results.csv"
    result_file_header= "ID;1;2;3;4_1;4_2;4_3;6_1;6_2;6_3;6_4_1;6_4_2;6_4_3"
    #1_1;1_2;1_3;1_4;1_5;2_1;2_2;2_3;2_4;2_5;3_1;3_2;3_3;3_4;3_5;4_1_1;4_1_2;4_1_3;4_1_4;4_1_5;4_2_1;4_2_2;4_2_3;4_2_4;4_2_5;4_3_1;4_3_2;4_3_3;4_3_4;4_3_5
    @staticmethod
    def create_results_file():
        if not path.isfile(Exporter.result_file_path):
                with open(Exporter.result_file_path, 'a') as file_obj:
                    file_obj.write(Exporter.result_file_header)
                    file_obj.write('\n')
    
    @staticmethod
    def get_next_result_index():
        Exporter.create_results_file()

        with open(Exporter.result_file_path, 'r') as file_obj:
            lines = file_obj.readlines()
            line = lines[len(lines) - 1]
            columns = line.split(";")

            if columns[0] == "ID":
                return 1
            else:
                return int(columns[0]) + 1

    @staticmethod
    def end_results_line():
        Exporter.create_results_file()
        with open(Exporter.result_file_path, 'a') as file_obj:
                    file_obj.write('\n')

    @staticmethod
    def add_result_entry(exec, value):
        Exporter.create_results_file()

        with open(Exporter.result_file_path, 'a') as file_obj:
            if exec == "1":
                line_number = Exporter.get_next_result_index()
                file_obj.write(str(line_number))

            file_obj.write(";" + str(value))
        


class Printer:
    def __init__(self):
        pass

    @staticmethod
    def print_msg(msg):
        print(msg)
        

class PrinterFile:
    path = "output/log/Logs de dados pro relatorio.txt"
    def __init__(self):
        pass

    @staticmethod
    def print_msg(msg):
        Printer.print_msg(msg)
        
        with codecs.open(PrinterFile.path, 'a', encoding='utf8') as file_obj:
            file_obj.write(msg)
            file_obj.write('\n')
            