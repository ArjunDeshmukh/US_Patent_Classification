from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
import pandas as pd

# region: Constants
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class CFG:
    CPC_CODES_PATH = "CPC_codes_upto_subclass.csv"
    PATENTS_DATA_PATH = "../input/us-patents-abstracts-cpc/US_Patents_Titles_CPC_Section.csv"
    BERT_FOR_PATENTS_PATH = "anferico/bert-for-patents"
    DEBERTA_V3_LARGE_PATH = "../input/deberta-v3-large/deberta-v3-large"
    CPC_CODES_NUM_PATENTS_PATH = "../input/cpc-code-num-patents/cpc_code_num_patents.csv"
    TRAINED_MODEL_PATH = "Trained_Models/BERT_For_Patents_FineTuned_3.pth"
    OUTPUT_DIR = "./"
    MAX_TOKEN_LEN = 512
    DROPOUT_PROB = 0.2
    ATTENTION_INTERMEDIATE_SIZE = 512
    GOOGLE_CLOUD_CRED_JSON = '../input/googlecloudcred/us-patent-classification-d344a6ddc702.json'
    GOOGLE_CLOUD_PROJ_ID = 'us-patent-classification'
    BATCH_SIZE = 128
    NUM_EPOCHS = 5
    TRAIN_PATENT_START_YEAR = 2001
    TRAIN_PATENT_END_YEAR = 2018
    TEST_PATENT_START_YEAR = 2019
    TEST_PATENT_END_YEAR = 2022
    NUM_FOLDS = 6  # Total number of years used for training should be divisible by this number
    ENCODER_LR = 2e-5
    DECODER_LR = 2e-5
    MIN_LR = 1e-6
    EPS = 1e-8
    WEIGHT_DECAY = 0.01
    SCHEDULER = 'linear'
    NUM_WARMUP_STEPS = 0
    NUM_CYCLES = 0.5
    BETAS = (0.9, 0.999)
    MAX_GRAD_NORM = 1000
    PRINT_FREQ = 1000
    INFINITY = 1e6
    F_TRAIN = 0
    MID_EPOCH_SAVE_PERCENT = 0.5
    OUTPUT_TRUE_THRESH = 0.5


# Parameters which will be added in subsequent code:
# NUM_CLASSES, TRAIN_NUM_PATENTS

# endregion


class PatentClassificationModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_config = AutoConfig.from_pretrained(self.cfg.BERT_FOR_PATENTS_PATH, output_hidden_states=False)
        self.model_config.max_position_embeddings = self.cfg.MAX_TOKEN_LEN
        self.model = AutoModel.from_pretrained(self.cfg.BERT_FOR_PATENTS_PATH, config=self.model_config,
                                               ignore_mismatched_sizes=True)
        '''
        self.attention = torch.nn.Sequential(torch.nn.Linear(self.model_config.hidden_size, self.cfg.ATTENTION_INTERMEDIATE_SIZE),
                                             torch.nn.Tanh(),
                                             torch.nn.Linear(self.cfg.ATTENTION_INTERMEDIATE_SIZE, 1),
                                            torch.nn.Softmax(dim=1))
        self.fc_dropout = torch.nn.Dropout(self.cfg.DROPOUT_PROB)
        '''
        self.fc = torch.nn.Linear(self.model_config.hidden_size, self.cfg.NUM_CLASSES)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs):
        # one patent can have multiple cpc codes, hence pass a list of cpc codes for each abstract
        model_outputs = self.model(**inputs)
        outputs = model_outputs.pooler_output
        outputs = self.fc(outputs)
        return outputs


def inference(title):

    cpc_codes_upto_subclass_df = pd.read_csv(CFG.CPC_CODES_PATH)
    sections = pd.unique(cpc_codes_upto_subclass_df['section'])
    cpc_code_section_dict = dict(zip(range(len(sections)), sections))

    CFG.NUM_CLASSES = len(sections)

    tokenizer = AutoTokenizer.from_pretrained(CFG.BERT_FOR_PATENTS_PATH)

    title_input = tokenizer(title, truncation=True, add_special_tokens=True, max_length=CFG.MAX_TOKEN_LEN,
                            return_tensors="pt")

    for k, v in title_input.items():
        title_input[k] = v.type(torch.long)

    for k, v in title_input.items():
        title_input[k] = v.to(DEVICE)

    model = PatentClassificationModel(CFG)
    model.to(DEVICE)
    state = torch.load(CFG.TRAINED_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state['model'])

    outputs = model(title_input)

    sigmoid_outputs = torch.sigmoid(outputs)
    thresh_outputs = sigmoid_outputs >= CFG.OUTPUT_TRUE_THRESH

    try:
        indices = thresh_outputs[0].nonzero().numpy()
        cpc_sections = [cpc_code_section_dict[key] for key in indices[0]]
    except:
        cpc_sections = []

    return cpc_sections


if __name__ == '__main__':

    Title = "Lasers with InGaAs quantum wells with InGaP barrier layers with reduced decomposition"

    cpc_sections = inference(Title)
    print(cpc_sections)
