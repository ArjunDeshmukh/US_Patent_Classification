from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
import pandas as pd

# region: Constants
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Device: ", DEVICE)


class CFG:
    CPC_CODES_PATH = "CPC_codes_upto_subclass.csv"
    PATENTS_DATA_PATH = "../input/us-patents-abstracts-cpc/US_Patents_Abstracts_CPC_Section.csv"
    BERT_FOR_PATENTS_PATH = "anferico/bert-for-patents"
    TRAINED_MODEL_PATH = "Trained_Models/BERT_For_Patents_FineTuned.bin"  # use None if trained model doesn't exist
    DEBERTA_V3_LARGE_PATH = "../input/deberta-v3-large/deberta-v3-large"
    CPC_CODES_NUM_PATENTS_PATH = "../input/cpc-code-num-patents/cpc_code_num_patents.csv"
    MAX_TOKEN_LEN = 200
    DROPOUT_PROB = 0.1
    ATTENTION_INTERMEDIATE_SIZE = 512
    GOOGLE_CLOUD_CRED_JSON = '../input/googlecloudcred/us-patent-classification-d344a6ddc702.json'
    GOOGLE_CLOUD_PROJ_ID = 'us-patent-classification'
    BATCH_SIZE = 1024
    BATCH_SIZE_INFER = 4096
    NUM_EPOCHS = 3
    TRAIN_PATENT_START_YEAR = 2001
    TRAIN_PATENT_END_YEAR = 2018
    TEST_PATENT_START_YEAR = 2019
    TEST_PATENT_END_YEAR = 2022
    NUM_FOLDS = 6  # Total number of years used for training should be divisible by this number
    ENCODER_LR = 1e-4
    DECODER_LR = 2e-5
    MIN_LR = 1e-6
    EPS = 1e-6
    WEIGHT_DECAY = 0.01
    SCHEDULER = 'linear'
    NUM_WARMUP_STEPS = 0
    NUM_CYCLES = 0.5
    BETAS = (0.9, 0.999)
    MAX_GRAD_NORM = 1000
    PRINT_FREQ = 50
    INFINITY = 1e6
    F_TRAIN = 0
    N_PROCS = 1
    START_EPOCH = 2  # 0 indexed, starts from 0
    BEST_LOSS = 0.223360  # using 1e6 if no training has been done
    OUTPUT_TRUE_THRESH = 0.3

# Parameters which will be added in subsequent code:
# NUM_CLASSES, TRAIN_NUM_PATENTS

class PatentClassificationModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_config = AutoConfig.from_pretrained(self.cfg.BERT_FOR_PATENTS_PATH, output_hidden_states=False)
        # self.model_config.max_position_embeddings = self.cfg.MAX_TOKEN_LEN
        self.model = AutoModel.from_pretrained(self.cfg.BERT_FOR_PATENTS_PATH, config=self.model_config)
        '''
        self.attention = torch.nn.Sequential(torch.nn.Linear(self.model_config.hidden_size, self.cfg.ATTENTION_INTERMEDIATE_SIZE),
                                             torch.nn.Tanh(),
                                             torch.nn.Linear(self.cfg.ATTENTION_INTERMEDIATE_SIZE, 1),
                                            torch.nn.Softmax(dim=1))
        '''
        self.fc_dropout = torch.nn.Dropout(self.cfg.DROPOUT_PROB)
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

    def forward(self, input_ids, token_type_ids, attention_mask):
        # one abstract can have multiple cpc codes, hence pass a list of cpc codes for each abstract
        model_outputs = self.model(input_ids, token_type_ids, attention_mask)

        outputs = model_outputs.pooler_output

        outputs = self.fc_dropout(outputs)
        outputs = self.fc(outputs)

        return outputs


def inference(abstract):
    cpc_codes_upto_subclass_df = pd.read_csv(CFG.CPC_CODES_PATH)
    sections = pd.unique(cpc_codes_upto_subclass_df['section'])
    cpc_code_section_dict = dict(zip(range(len(sections)), sections))

    CFG.NUM_CLASSES = len(sections)

    tokenizer = AutoTokenizer.from_pretrained(CFG.BERT_FOR_PATENTS_PATH)
    CFG.tokenizer = tokenizer

    abstract_input = CFG.tokenizer(abstract, truncation=True, add_special_tokens=True, max_length=CFG.MAX_TOKEN_LEN, padding='max_length',
                               return_tensors='pt')

    for k, v in abstract_input.items():
        abstract_input[k] = v.type(torch.long)

    MX = PatentClassificationModel(CFG)
    model = MX.to(DEVICE)
    state = torch.load(CFG.TRAINED_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state['model'])
    model.to(DEVICE)
    model.eval()

    input_ids = abstract_input['input_ids']
    token_type_ids = abstract_input['token_type_ids']
    attention_mask = abstract_input['attention_mask']

    input_ids = input_ids.to(DEVICE)
    token_type_ids = token_type_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, token_type_ids, attention_mask)

    sigmoid_outputs = torch.sigmoid(outputs)
    thresh_outputs = sigmoid_outputs >= CFG.OUTPUT_TRUE_THRESH

    try:
        indices = thresh_outputs[0].nonzero().numpy()
        if indices.size == 0:
            cpc_sections = [cpc_code_section_dict[torch.argmax(sigmoid_outputs).item()]]
        else:
            cpc_sections = [cpc_code_section_dict[key] for key in indices[0]]
    except:
        cpc_sections = []

    return cpc_sections


if __name__ == '__main__':
    Abstract = "A building control system determines the uncertainty in a break even temperature parameter of an energy use model. The energy use model is used to predict energy consumption of a building site as a function of the break even temperature parameter and one or more predictor variables. The uncertainty in the break even temperature parameter is used to analyze energy performance of the building site."

    print("Abstract: ", Abstract)
    cpc_sections = inference(Abstract)
    print("CPC Section: ", cpc_sections)
