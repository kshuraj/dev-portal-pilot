from transformers import pipeline

def run_model(text):

    model = pipeline('text-generation', model='gpt2')
    output = model(text)
    return output[0].get('generated_text')

def main(input_text):
    """
    Argument(s) name in the function signature are matching with the id in the
    app.config
    """
    output = run_model(input_text)
    print(output)
    return output

# if __name__=="__main__":
#     main('My name is Mariama, my favorite')
