from transformers import BertTokenizer, BertModel, BertConfig
import torch
import pandas as pd


# region: Constants
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_CLASSES = 674
# endregion


class USPatentClassification(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = BertConfig.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = torch.nn.Dropout(self.config.hidden_dropout_prob)
        self.linear = torch.nn.Linear(self.config.hidden_size, NUM_CLASSES)
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, abstract_input):
        abstract_output = self.model(**abstract_input)

        last_hidden_state = abstract_output.last_hidden_state
        outputs = self.dropout(last_hidden_state)
        outputs = self.linear(outputs)
        outputs = self.softmax(outputs)

        return outputs


def inference(abstract):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    config = BertConfig.from_pretrained("bert-base-uncased")
    abstract_input = tokenizer(abstract, padding=True, max_length=config.max_position_embeddings,
                               truncation=True, return_tensors="pt")

    abstract_input = {k: v.to(DEVICE) for k, v in abstract_input.items()}
    model = USPatentClassification()
    model.to(DEVICE)

    outputs = model(abstract_input)

    cpc_code_ind = torch.argmax(outputs)
    cpc_code_ind = cpc_code_ind.cpu().detach().numpy()

    cpc_codes_upto_subclass_df = pd.read_csv('CPC_codes_upto_subclass.csv')

    cpc_code = cpc_codes_upto_subclass_df.iloc[cpc_code_ind].code

    return cpc_code


if __name__ == '__main__':
    Abstract = "A multi-beam frequency-modulated continuous wave (FMCW) radar system designed for short range (<20 " \
               "km) operation in a high-density threat environment against highly maneuverable threats. The " \
               "multi-beam FMCW system is capable of providing continuous updates, both search and track, " \
               "for an entire hemisphere against short-range targets. The multi-beam aspect is used to cover the " \
               "entire field of regard, whereas the FMCW aspect is used to achieve resolution at a significantly " \
               "reduced computational effort. "

    CPC_Code = inference(Abstract)
    print(CPC_Code)


