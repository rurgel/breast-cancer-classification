def call_prediction():
    import sys
    from detect_breast_cancer.predict import make_prediction
    output = make_prediction(sys.argv[1:])
    
    def colorize_output(item):
        if item=='Malignant':
            return '\033[1;91mMalignant\033[0m'
        elif item=='Benign':
            return '\033[1;96mBenign\033[0m'
        return item
    
    
    print(*[colorize_output(item) for item in output], sep='\n')