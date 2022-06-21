from transformers import M2M100ForConditionalGeneration, AutoTokenizer
from model_config import language_code

def run_model(text,src,target):
    source_code = None
    target_code = None
    for lan, code in language_code.items():
        if lan == src:
            source_code = code
        if lan == target:
            target_code = code

    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")
    tokenizer.src_lang = source_code
    user_input = text
    encoded_hi = tokenizer(user_input, return_tensors="pt")
    generate = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id(target_code))
    result = tokenizer.batch_decode(generate, skip_special_tokens=True)
    return result[0]
    
def main(input_text,src_selection,target_selection):
    """
    Argument(s) name in the function signature are matching with the id in the
    app.config
    """
    output = run_model(input_text,src_selection,target_selection)
    print(output)
    return output

# if __name__=="__main__":
#     main("this is for you","English","Hindi")
