from transformers import AutoModel, AutoTokenizer, BertModel

def get_text_reader(reader_name):
    # Messenger Corpus is korean dataset.
    # So, model is fixed to Korean Model such as multilingual-BERT, kobert, koelectra, etc.
    if reader_name == "bert":
        model_name = "bert-base-multilingual-cased"
        text_reader = AutoModel.from_pretrained(model_name)

    elif reader_name == "kobert":
        # model_name = "monologg/kobert"
        # text_reader = BertModel.from_pretrained(model_name)
        raise NotImplementedError("Kobert is not supported in this version.")

    elif reader_name == "koelectra":
        raise NotImplementedError("Koelectra is not supported in this version.")

    else:
        raise KeyError(reader_name)

    return text_reader

def get_tokenizer(reader_name):
    # Messenger Corpus is korean dataset.
    # So, tokenized is fixed to Korean Tokenizer such as multilingual-BERT tokenizer, kobert tokenizer, etc.

    if reader_name == "bert":
        model_name = "bert-base-multilingual-cased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif reader_name == "kobert":
        # from utils.tokenization_kobert import KoBertTokenizer
        # tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")
        raise NotImplementedError("Kobert is not supported in this version.")

    elif reader_name == "koelectra":
        # from utils.~~~ import ~~~~
        # tokenizer = ~~~
        raise NotImplementedError("Koelectra is not supported in this version.")

    else:
        raise KeyError(reader_name)

    return tokenizer
