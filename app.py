from transformers import pipeline

def run_model(text):

    unmasker = pipeline('fill-mask', model='bert-base-uncased')
    output = unmasker(text)
    return output[0]['token_str']


def main(input_text):
    """
    Argument(s) name in the function signature are matching with the id in the
    app.config
    """
    output = run_model(input_text)
    # print(output)
    return output

# if __name__=="__main__":
#     main('The goal of life is to [MASK].')
