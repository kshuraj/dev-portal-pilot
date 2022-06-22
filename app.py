from transformers import pipeline

def run_model(text):
    classifier = pipeline('text-classification',model = 'distilbert-base-uncased-finetuned-sst-2-english')
    return classifier(text)[0]

def main(input_text):
    """
    Argument(s) name in the function signature are matching with the id in the
    app.config
    """
    output = run_model(input_text)
    # print(output)
    return output

# if __name__=="__main__":
    # main('Today is a beautiful day!')
