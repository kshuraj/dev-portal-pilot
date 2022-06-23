from transformers import pipeline

def run_model(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    output = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return output[0]['summary_text']

def main(input_text):
    """
    Argument(s) name in the function signature are matching with the id in the
    app.config
    """
    output = run_model(input_text)
    # print(output)
    return output

# if __name__=="__main__":
#     main('The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.')
