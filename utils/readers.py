from transformers import AutoModel, AutoTokenizer, BertForSequenceClassification

def get_text_reader(reader_name, num_labels):
    # Messenger Corpus is korean dataset.
    # So, model is fixed to Korean Model such as multilingual-BERT, kobert, koelectra, etc.

    if reader_name == "bert":
        model_name = "bert-base-multilingual-cased"
        # text_reader = AutoModel.from_pretrained(model_name)
        text_reader = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    elif reader_name == "kobert":
        from transformers import BertModel
        model_name = "monologg/kobert"
        # text_reader = BertModel.from_pretrained(model_name)
        text_reader = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    elif reader_name == "koelectra":
        from transformers import ElectraModel, ElectraForSequenceClassification
        model_name = "monologg/koelectra-base-discriminator"
        # text_reader = ElectraModel.from_pretrained(model_name)
        text_reader = ElectraForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

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
        from utils.tokenization_kobert import KoBertTokenizer
        model_name = "monologg/kobert"
        tokenizer = KoBertTokenizer.from_pretrained(model_name)

    elif reader_name == "koelectra":
        from transformers import ElectraTokenizer
        model_name = "monologg/koelectra-base-discriminator"
        tokenizer = ElectraTokenizer.from_pretrained(model_name)

    else:
        raise KeyError(reader_name)

    return tokenizer
