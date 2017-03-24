from . import CPPDataType
import struct
import os

# c++ data file generator
class CPPDataCoder:
    model_name = ""
    datas = []

    # initialize with c++ model name
    def __init__(self, model_name):
        self.model_name = model_name
        self.datas = []

    # add data to file
    def append(self, name, data, cpp_type):
        assert len(data) > 0, name+" length must > 0"

        single_data = {}
        single_data["name"] = name
        single_data["data"] = data
        single_data["bits"] = CPPDataType.bits(cpp_type)
        single_data["pack_type"] = CPPDataType.pack_type(cpp_type)
        self.datas.append(single_data)

    # check data exists
    def has(self, name):
        for single_data in self.datas:
            if single_data["name"] == name: return True
        return False

    # remove all data
    def clear(self):
        self.datas = []

    # dump all datas to c++ file
    def dump_to_file(self, filename):
        file = open(filename, 'w')

        macro_name = "__"
        macro_name = macro_name + os.path.basename(filename).replace('.', '_').upper()
        macro_name = macro_name + "__"
        file.write("#ifndef "+macro_name+"\n#define "+macro_name+"\n\n")
        # write namespace
        if len(self.model_name) > 0:
            file.write("namespace "+self.model_name+"{\n")

        # write datas
        for single_data in self.datas:
            print("start dump data: "+single_data["name"]+" data length: "+str(len(single_data["data"])))

            # begin c++ define code
            data_define_string = ""
            data_define_string = "\n#pragma - mark " + single_data["name"]+"\n\n"
            data_define_string = data_define_string+"const uint8_t "+single_data["name"]+"[] = {\n"

            # save data as single byte, use uint8_t in c++
            file.write(data_define_string)
            data_define_string = "    "
            data_count = 0
            for data in single_data["data"]:
                bytes = struct.pack(single_data["pack_type"], data)

                for b in bytes:
                    data_define_string = data_define_string+"0x"+hex(b)[2:].zfill(2).upper()+", "
                data_count = data_count + 1

                if data_count % 16 == 0:
                    file.write(data_define_string[:-2])
                    data_define_string = ", \n    "
            
            # remove last data's ','
            trim_index = data_define_string.rindex(',')
            if trim_index >= 0:
                data_define_string = data_define_string[:trim_index]

            # close c++ define code
            data_define_string = data_define_string + "\n}; // " + single_data["name"]+"\n"
            file.write(data_define_string)

        # close namespace
        if len(self.model_name) > 0:
            file.write("\n} // namespace "+self.model_name+"\n\n")

        file.write("#endif // "+macro_name+"\n\n")

        # write finished
        file.close()

