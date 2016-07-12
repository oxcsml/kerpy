class GenericTests():
    @staticmethod
    def check_type(varvalue, varname, vartype, required_shapelen=None):
        if not type(varvalue) is vartype:
            raise TypeError("Variable " + varname + " must be of type " + vartype.__name__ + \
                            ". Given is " + str(type(varvalue)))
        if not required_shapelen is None:
            if not len(varvalue.shape) is required_shapelen:
                raise ValueError("Variable " + varname + " must be " + str(required_shapelen) + "-dimensional")
        return 0
