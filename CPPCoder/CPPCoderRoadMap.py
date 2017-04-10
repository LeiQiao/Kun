from . import CPPDataCoder
from . import CPPDataType
from . import CPPMapCoder

class CPPCoderRoadMap:
    model_name = ""
    ops = []
    data_holder = None
    map_coder = None

    def __init__(self, model_name):
        self.model_name = model_name
        self.ops = []
        self.data_holder = CPPDataCoder.CPPDataCoder(model_name)
        self.map_coder = None

    def set_map(self, mapfile):
        self.map_coder = CPPMapCoder.CPPMapCoder(self.model_name)
        if mapfile is not None: self.map_coder.parse_from_csv(mapfile)

    def append_op(self, new_op):
        if len(self.ops) > 0:
            last_op = self.ops[len(self.ops)-1]
            match = (last_op.output_shape == new_op.input_shape)
            assert match, "last op's output dimensions != new op's input dimensions. ("+str(last_op.output_shape)+" != "+str(new_op.input_shape)+")"
        self.ops.append(new_op)

    def dump_codes(self, path, cpp_type):
        filename = path + "/" + self.model_name
        header_filename = filename + ".hpp"
        cpp_filename = filename + ".cpp"
        data_filename = filename + "_data.hpp"
        map_filename = filename + "_map.hpp"

        input_dimensions = str(len(self.ops[0].input_shape))
        output_dimensions = str(len(self.ops[len(self.ops)-1].output_shape))

        # write header file
        file = open(header_filename, 'w')
        template_code = open('templates/predict_header.tmp').read()
        template_code = template_code.replace("%FILE_NAME%", self.model_name+".hpp")
        template_code = template_code.replace("%CAPTAL_MODEL_NAME%", self.model_name.upper())
        template_code = template_code.replace("%MODEL_NAME%", self.model_name)
        template_code = template_code.replace("%CPP_TYPE%", CPPDataType.type_string(cpp_type))
        template_code = template_code.replace("%INPUT_DISMENSIONS%", input_dimensions)
        template_code = template_code.replace("%OUTPUT_DISMENSIONS%", output_dimensions)
        file.write(template_code)
        file.close()
        
        # write cpp file
        file = open(cpp_filename, 'w')
        file.write("//\n// "+self.model_name+"cpp"+"\n")
        file.write("//\n// generated time: "+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+"\n//\n\n")
        file.write("#include \""+self.model_name+".hpp\"\n\n")
        file.write("#include \""+self.model_name+"_data.hpp\"\n\n")
        file.write("#include \""+self.model_name+"_map.hpp\"\n\n")
        file.write("namespace "+self.model_name+" {\n\n")

        # write unit test function
        template_code = ""
        if self.ops[0].dim_ordering == CPPDataType.ORDER_NHWC:
            template_code = open('templates/unit_test_NHWC.tmp').read()
        elif self.ops[0].dim_ordering == CPPDataType.ORDER_NCHW:
            template_code = open('templates/unit_test_NCHW.tmp').read()
        template_code = template_code.replace("%CPP_TYPE%", CPPDataType.type_string(cpp_type))
        file.write(template_code)

        # write predict function
        self.data_holder.clear()
        for op in self.ops:
            op.dump_to_file(file, self.data_holder, cpp_type)

        # function declare
        predict_function_declare = "void predict(const Eigen::Tensor<"+CPPDataType.type_string(cpp_type)+", "+input_dimensions+", Eigen::RowMajor>& input,\n"
        predict_function_declare = predict_function_declare + "             Eigen::Tensor<"+CPPDataType.type_string(cpp_type)+", "+output_dimensions+", Eigen::RowMajor>& output"

        file.write(predict_function_declare+")\n")
        file.write("")
        file.write("{\n")
        file.write("#ifdef DEBUG\n")
        file.write("    predict(input, output, false);\n")
        file.write("}\n\n")
        file.write(predict_function_declare+",\n")
        file.write("             bool unit_test)\n")
        file.write("{\n")
        file.write("    if( unit_test )\n")
        file.write("    {\n")
        file.write("        start_unit_test_function(\"input\", input.data(), "+str(len(self.ops[0].input_shape)))
        for index in range(len(self.ops[0].input_shape)):
            file.write(", "+str(self.ops[0].input_shape[index]))
        file.write(");\n")
        file.write("    }\n")
        file.write("#endif // DEBUG\n")

        # function body
        last_op_output_name = "input"
        last_op_output_shape = None
        last_op_output_shape_len = 0
        for op in self.ops:
            op_output_dimensions = str(len(op.output_shape))
            op_output_name = "output_"+op.name
            op_output_shape = ""
            for shape in op.output_shape:
                op_output_shape = op_output_shape + str(shape) + ", "
            op_output_shape = op_output_shape[:-2] + ""
            file.write("    Eigen::Tensor<"+CPPDataType.type_string(cpp_type)+", "+op_output_dimensions+", Eigen::RowMajor> "+op_output_name+"("+op_output_shape+");\n")
            file.write("    "+op.name+"("+last_op_output_name+", "+op_output_name+");\n\n")

            file.write("#ifdef DEBUG\n")
            file.write("    if( unit_test )\n")
            file.write("    {\n")
            file.write("        unit_test_function(\""+op.name+"\", "+op_output_name+".data(), "+str(len(op.output_shape))+", "+op_output_shape+");\n")
            file.write("    }\n")
            file.write("#endif // DEBUG\n\n")

            last_op_output_name = op_output_name
            last_op_output_shape = op_output_shape
            last_op_output_shape_len = len(op.output_shape)
        file.write("    output = "+last_op_output_name+";\n")

        file.write("\n#ifdef DEBUG\n")
        file.write("    if( unit_test )\n")
        file.write("    {\n")
        file.write("        end_unit_test_function(\"output\", output.data(), "+str(last_op_output_shape_len)+", "+last_op_output_shape+");\n")
        file.write("    }\n")
        file.write("#endif // DEBUG\n\n")

        file.write("}\n")

        # write mapping table
        file.write("std::string getMappingTableValue(Eigen::Index index)\n")
        file.write("{\n")
        file.write("    if( index >= sizeof(value_map)/sizeof(const char*) ) return \"<null>\";\n")
        file.write("    else return value_map[index];\n")
        file.write("}\n")
        
        file.write("\n\n};\n\n")
        file.close()

        # write data file
        self.data_holder.dump_to_file(data_filename)

        # write map file
        self.map_coder.dump_to_file(map_filename)





