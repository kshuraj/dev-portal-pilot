from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-base-squad2"

def run_model(text):
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    res = nlp(text)
    return res['answer']

def main(input_text):
    """
    Argument(s) name in the function signature are matching with the id in the
    app.config
    """
    output = run_model(input_text)
    # print(output)
    return output

if __name__=="__main__":
    # QA_input = {
    #     'question': 'Why is model conversion important?',
    #     'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
    # }
    main(QA_input)
