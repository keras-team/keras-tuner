try:
    import namex
except ImportError:
    namex = None

if namex:

    class keras_tuner_export(namex.export):
        def __init__(self, path):
            super().__init__(package="keras_tuner", path=path)

else:

    class keras_tuner_export:
        def __init__(self, _):
            pass

        def __call__(self, symbol):
            return symbol
