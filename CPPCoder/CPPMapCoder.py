import os
import csv

class CPPMapCoder:
    model_name = None
    map = []

    def __init__(self, model_name):
        self.model_name = model_name
        self.map = []

    def parse_from_csv(self, csv_file):
        self.map = []
        with open(csv_file) as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                self.map.append(row[1])

    def dump_to_file(self, filename):
        print("start dump map length: "+str(len(self.map)))

        file = open(filename, 'w')

        macro_name = "__"
        macro_name = macro_name + os.path.basename(filename).replace('.', '_').upper()
        macro_name = macro_name + "__"
        file.write("#ifndef "+macro_name+"\n#define "+macro_name+"\n\n")

        # write namespace
        if len(self.model_name) > 0:
            file.write("namespace "+self.model_name+"{\n")

        # write c++ define code
        data_define_string = ""
        data_define_string = "\n#pragma - mark value_map\n\n"
        data_define_string = data_define_string+"const char* value_map[] = {\n"
        file.write(data_define_string)

        # write data
        data_define_string = "    "
        data_count = 0
        for element in self.map:
            data_define_string = data_define_string + "\""+element+"\", "
            data_count = data_count + 1
            
            if data_count % 16 == 0:
                file.write(data_define_string[:-2])
                data_define_string = ", \n    "

        # remove last data's ','
        trim_index = data_define_string.rfind(',')
        if trim_index >= 0:
            data_define_string = data_define_string[:trim_index]

        # close c++ define code
        data_define_string = data_define_string+"\n}; // value_map\n"
        file.write(data_define_string)

        # close namespace
        if len(self.model_name) > 0:
            file.write("\n} // namespace "+self.model_name+"\n\n")

        file.write("#endif // "+macro_name+"\n\n")

        # write finished
        file.close()
