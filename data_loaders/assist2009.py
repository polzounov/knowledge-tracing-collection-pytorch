import os
import pickle
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from models.utils import match_seq_len


DATASET_DIR = "datasets/ASSIST2009/"


class ASSIST2009(Dataset):
    def __init__(self, seq_len, dataset_dir=DATASET_DIR, **kwargs) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir
        self.dataset_path = os.path.join(self.dataset_dir, "skill_builder_data.csv")

        if os.path.exists(os.path.join(self.dataset_dir, "q_seqs.pkl")):
            with open(os.path.join(self.dataset_dir, "q_seqs.pkl"), "rb") as f:
                self.q_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "r_seqs.pkl"), "rb") as f:
                self.r_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "q_list.pkl"), "rb") as f:
                self.q_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "u_list.pkl"), "rb") as f:
                self.u_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "q2idx.pkl"), "rb") as f:
                self.q2idx = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "u2idx.pkl"), "rb") as f:
                self.u2idx = pickle.load(f)
        else:
            (
                self.q_seqs,  # question sequences
                self.r_seqs,  # response sequences (correct/incorrect)
                self.q_list,  # question list
                self.u_list,  # user list
                self.q2idx,  # question to index
                self.u2idx,  # user to index
            ) = self.preprocess()

        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]

        if seq_len:
            self.q_seqs, self.r_seqs = match_seq_len(self.q_seqs, self.r_seqs, seq_len)

        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        return self.q_seqs[index], self.r_seqs[index]

    def __len__(self):
        return self.len

    def preprocess(self):
        df = (
            pd.read_csv(self.dataset_path, encoding="unicode_escape")
            .dropna(subset=["skill_name"])
            .drop_duplicates(subset=["order_id", "skill_name"])
            .drop_duplicates(subset=["order_id", "problem_id"])
            .sort_values(by=["order_id"])
        )

        u_list = np.unique(df["user_id"].values)
        # q_list = np.unique(df["skill_name"].values)
        q_list = np.unique(df["problem_id"].values)

        u2idx = {u: idx for idx, u in enumerate(u_list)}
        q2idx = {q: idx for idx, q in enumerate(q_list)}

        q_seqs = []
        r_seqs = []

        for u in u_list:
            df_u = df[df["user_id"] == u]

            q_seq = np.array([q2idx[q] for q in df_u["problem_id"]])
            r_seq = df_u["correct"].values

            q_seqs.append(q_seq)
            r_seqs.append(r_seq)

        with open(os.path.join(self.dataset_dir, "q_seqs.pkl"), "wb") as f:
            pickle.dump(q_seqs, f)
        with open(os.path.join(self.dataset_dir, "r_seqs.pkl"), "wb") as f:
            pickle.dump(r_seqs, f)
        with open(os.path.join(self.dataset_dir, "q_list.pkl"), "wb") as f:
            pickle.dump(q_list, f)
        with open(os.path.join(self.dataset_dir, "u_list.pkl"), "wb") as f:
            pickle.dump(u_list, f)
        with open(os.path.join(self.dataset_dir, "q2idx.pkl"), "wb") as f:
            pickle.dump(q2idx, f)
        with open(os.path.join(self.dataset_dir, "u2idx.pkl"), "wb") as f:
            pickle.dump(u2idx, f)

        return q_seqs, r_seqs, q_list, u_list, q2idx, u2idx


"""WRONG VERISON:
(pt) ➜  knowledge-tracing-collection-pytorch git:(main) ✗ p train.py
Epoch: 1,   AUC: 0.7426833085653219,   Loss Mean: 0.651930034160614
Epoch: 2,   AUC: 0.7788455849454976,   Loss Mean: 0.6188383102416992
Epoch: 3,   AUC: 0.7961521558778824,   Loss Mean: 0.6070793271064758
Epoch: 4,   AUC: 0.8040046535753171,   Loss Mean: 0.6008087396621704
Epoch: 5,   AUC: 0.8080574437582007,   Loss Mean: 0.5959572196006775
Epoch: 6,   AUC: 0.8118292030264298,   Loss Mean: 0.5943689346313477
Epoch: 7,   AUC: 0.8146803047538316,   Loss Mean: 0.5900989174842834
Epoch: 8,   AUC: 0.8164375447318173,   Loss Mean: 0.5892611145973206
Epoch: 9,   AUC: 0.8174869100365982,   Loss Mean: 0.5864916443824768
Epoch: 10,   AUC: 0.8175284558098723,   Loss Mean: 0.5849495530128479
Epoch: 11,   AUC: 0.8188460237201447,   Loss Mean: 0.5848451852798462
Epoch: 12,   AUC: 0.8205017129278251,   Loss Mean: 0.5831145644187927
Epoch: 13,   AUC: 0.8207450461613204,   Loss Mean: 0.5813812613487244
Epoch: 14,   AUC: 0.8207844765856755,   Loss Mean: 0.5790265202522278
Epoch: 15,   AUC: 0.8212282912116884,   Loss Mean: 0.5781517624855042
Epoch: 16,   AUC: 0.8204424390883276,   Loss Mean: 0.5786222815513611
Epoch: 17,   AUC: 0.8216242861318381,   Loss Mean: 0.5770255923271179
Epoch: 18,   AUC: 0.8216713566516257,   Loss Mean: 0.5760564208030701
Epoch: 19,   AUC: 0.8226370895536009,   Loss Mean: 0.5758625864982605
Epoch: 20,   AUC: 0.8233291632113964,   Loss Mean: 0.5750952959060669
Epoch: 21,   AUC: 0.8230894597243414,   Loss Mean: 0.5735712647438049
Epoch: 22,   AUC: 0.8237822084508735,   Loss Mean: 0.5726742148399353
Epoch: 23,   AUC: 0.8228128377897725,   Loss Mean: 0.5717707276344299
Epoch: 24,   AUC: 0.8227552406046738,   Loss Mean: 0.5709912776947021
Epoch: 25,   AUC: 0.8229203720328245,   Loss Mean: 0.569848358631134
Epoch: 26,   AUC: 0.8224093029327566,   Loss Mean: 0.5693582892417908
Epoch: 27,   AUC: 0.8225787712268253,   Loss Mean: 0.5685738921165466
Epoch: 28,   AUC: 0.8231681202559193,   Loss Mean: 0.5686150193214417
Epoch: 29,   AUC: 0.8224538214122765,   Loss Mean: 0.5674295425415039
Epoch: 30,   AUC: 0.8223708260179521,   Loss Mean: 0.5670996308326721
Epoch: 31,   AUC: 0.8214760374316119,   Loss Mean: 0.566155195236206
Epoch: 32,   AUC: 0.8215874518175198,   Loss Mean: 0.5657122731208801
Epoch: 33,   AUC: 0.8217577554340308,   Loss Mean: 0.5647773146629333
Epoch: 34,   AUC: 0.8220638319997581,   Loss Mean: 0.5632044672966003
Epoch: 35,   AUC: 0.8217055287512536,   Loss Mean: 0.5633799433708191
Epoch: 36,   AUC: 0.8200978315221701,   Loss Mean: 0.5630050301551819
Epoch: 37,   AUC: 0.8204099937223817,   Loss Mean: 0.5616648197174072
Epoch: 38,   AUC: 0.8200657827841462,   Loss Mean: 0.5620717406272888
Epoch: 39,   AUC: 0.8201301727232071,   Loss Mean: 0.5604421496391296
Epoch: 40,   AUC: 0.8204494602038206,   Loss Mean: 0.5597252249717712
Epoch: 41,   AUC: 0.8199081532326066,   Loss Mean: 0.5585888028144836
Epoch: 42,   AUC: 0.8182278170049471,   Loss Mean: 0.5588469505310059
Epoch: 43,   AUC: 0.8174860326475581,   Loss Mean: 0.557762622833252
Epoch: 44,   AUC: 0.8187693603503774,   Loss Mean: 0.5575568079948425
Epoch: 45,   AUC: 0.8171389411489579,   Loss Mean: 0.5572953820228577
Epoch: 46,   AUC: 0.8177314872394861,   Loss Mean: 0.5564112067222595
Epoch: 47,   AUC: 0.8171726425033243,   Loss Mean: 0.5566766262054443
Epoch: 48,   AUC: 0.81499442805877,   Loss Mean: 0.5546102523803711
Epoch: 49,   AUC: 0.8162293811772695,   Loss Mean: 0.5548411011695862
Epoch: 50,   AUC: 0.8157198084461812,   Loss Mean: 0.5542720556259155
Epoch: 51,   AUC: 0.81505715336246,   Loss Mean: 0.5529970526695251
Epoch: 52,   AUC: 0.8136949247475038,   Loss Mean: 0.5532617568969727
Epoch: 53,   AUC: 0.8141261955146815,   Loss Mean: 0.5516216158866882
Epoch: 54,   AUC: 0.8123311597001416,   Loss Mean: 0.5514847040176392
Epoch: 55,   AUC: 0.8130560813613193,   Loss Mean: 0.5497470498085022
Epoch: 56,   AUC: 0.8125316611243724,   Loss Mean: 0.5502811074256897
Epoch: 57,   AUC: 0.8113464447466936,   Loss Mean: 0.5494081974029541
^C[E thread_pool.cpp:109] Exception in thread pool task: mutex lock failed: Invalid argument
[E thread_pool.cpp:109] Exception in thread pool task: mutex lock failed: Invalid argument
[E thread_pool.cpp:109] Exception in thread pool task: mutex lock failed: Invalid argument
Traceback (most recent call last):
  File "/Users/kirill/code/THESIS/knowledge-tracing-collection-pytorch/train.py", line 155, in <module>
  File "/Users/kirill/code/THESIS/knowledge-tracing-collection-pytorch/train.py", line 124, in main
    parser.add_argument(
  File "/Users/kirill/code/THESIS/knowledge-tracing-collection-pytorch/models/dkt.py", line 100, in train_model
    y = self(q.long(), r.long())
  File "/Users/kirill/Applications/miniforge3/envs/pt/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/kirill/code/THESIS/knowledge-tracing-collection-pytorch/models/dkt.py", line 56, in forward
    h, _ = self.lstm_layer(self.interaction_emb(x))
  File "/Users/kirill/Applications/miniforge3/envs/pt/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/kirill/Applications/miniforge3/envs/pt/lib/python3.9/site-packages/torch/nn/modules/rnn.py", line 812, in forward
    result = _VF.lstm(input, hx, self._flat_weights, self.bias, self.num_layers,
KeyboardInterrupt

(pt) ➜  knowledge-tracing-collection-pytorch git:(main) ✗ p get_data.py
"""


"""CORRECT VERSION:
Epoch: 1,   AUC: 0.543242485547906,   Loss Mean: 0.6933905482292175
Epoch: 2,   AUC: 0.5801690779605806,   Loss Mean: 0.6820884346961975
Epoch: 3,   AUC: 0.6373882055078559,   Loss Mean: 0.6613982915878296
Epoch: 4,   AUC: 0.6620846588741769,   Loss Mean: 0.6274881958961487
Epoch: 5,   AUC: 0.6706315355045844,   Loss Mean: 0.604880690574646
Epoch: 6,   AUC: 0.676452575681123,   Loss Mean: 0.5889681577682495
Epoch: 7,   AUC: 0.6793252807941053,   Loss Mean: 0.5743709206581116
Epoch: 8,   AUC: 0.6822333456805469,   Loss Mean: 0.5571814179420471
Epoch: 9,   AUC: 0.6837335848599276,   Loss Mean: 0.5457220077514648
Epoch: 10,   AUC: 0.6856064928515365,   Loss Mean: 0.5317967534065247
Epoch: 11,   AUC: 0.6858195519132282,   Loss Mean: 0.5218749046325684
Epoch: 12,   AUC: 0.6869454404130494,   Loss Mean: 0.5112181305885315
Epoch: 13,   AUC: 0.6884747688669484,   Loss Mean: 0.5009915232658386
Epoch: 14,   AUC: 0.6893240631978685,   Loss Mean: 0.49191346764564514
Epoch: 15,   AUC: 0.6899247331921629,   Loss Mean: 0.4833987057209015
Epoch: 16,   AUC: 0.6909000134468267,   Loss Mean: 0.4763519763946533
Epoch: 17,   AUC: 0.691539104104965,   Loss Mean: 0.4686163067817688
Epoch: 18,   AUC: 0.6923850374051892,   Loss Mean: 0.46050217747688293
Epoch: 19,   AUC: 0.6926450751855536,   Loss Mean: 0.45483022928237915
Epoch: 20,   AUC: 0.6934165682805824,   Loss Mean: 0.44855189323425293
Epoch: 21,   AUC: 0.6929769005797115,   Loss Mean: 0.4440975785255432
Epoch: 22,   AUC: 0.6933873356958251,   Loss Mean: 0.4395618438720703
Epoch: 23,   AUC: 0.6935447308975766,   Loss Mean: 0.4347134530544281
Epoch: 24,   AUC: 0.6937717532436718,   Loss Mean: 0.42936626076698303
Epoch: 25,   AUC: 0.6943650414086824,   Loss Mean: 0.4243057668209076
Epoch: 26,   AUC: 0.6947776126585448,   Loss Mean: 0.42120710015296936
Epoch: 27,   AUC: 0.6944195290430772,   Loss Mean: 0.4174083471298218
Epoch: 28,   AUC: 0.6946640433502116,   Loss Mean: 0.41385504603385925
Epoch: 29,   AUC: 0.695070849742919,   Loss Mean: 0.4109474718570709
Epoch: 30,   AUC: 0.6951518335476993,   Loss Mean: 0.4075356125831604
Epoch: 31,   AUC: 0.695663272638563,   Loss Mean: 0.4048226773738861
Epoch: 32,   AUC: 0.695876350628022,   Loss Mean: 0.40077871084213257
Epoch: 33,   AUC: 0.6957811899254633,   Loss Mean: 0.3977942168712616
Epoch: 34,   AUC: 0.6958257215542087,   Loss Mean: 0.3966958224773407
Epoch: 35,   AUC: 0.6959268201677957,   Loss Mean: 0.3935547471046448
Epoch: 36,   AUC: 0.6961487347206612,   Loss Mean: 0.39156803488731384
Epoch: 37,   AUC: 0.6955239020793045,   Loss Mean: 0.3897130489349365
Epoch: 38,   AUC: 0.6958326842686431,   Loss Mean: 0.3876948654651642
Epoch: 39,   AUC: 0.6959052235851988,   Loss Mean: 0.38599976897239685
Epoch: 40,   AUC: 0.6956949550172145,   Loss Mean: 0.3846176266670227
Epoch: 41,   AUC: 0.6956898634477853,   Loss Mean: 0.3835219442844391
Epoch: 42,   AUC: 0.6961531151468295,   Loss Mean: 0.38183653354644775
Epoch: 43,   AUC: 0.6961516523008067,   Loss Mean: 0.3791954517364502
Epoch: 44,   AUC: 0.6961919062541211,   Loss Mean: 0.37661290168762207
Epoch: 45,   AUC: 0.696562030633618,   Loss Mean: 0.3768516182899475
Epoch: 46,   AUC: 0.6962980964371371,   Loss Mean: 0.37505412101745605
Epoch: 47,   AUC: 0.6965723408589145,   Loss Mean: 0.3738075792789459
Epoch: 48,   AUC: 0.6960892447477406,   Loss Mean: 0.37209975719451904
Epoch: 49,   AUC: 0.6963362196646363,   Loss Mean: 0.37286803126335144
Epoch: 50,   AUC: 0.6960792833341576,   Loss Mean: 0.37165316939353943
Epoch: 51,   AUC: 0.6960870626465567,   Loss Mean: 0.37017256021499634
Epoch: 52,   AUC: 0.6959935919232296,   Loss Mean: 0.3695909082889557
Epoch: 53,   AUC: 0.6959207551703293,   Loss Mean: 0.36816418170928955
Epoch: 54,   AUC: 0.6962435790591077,   Loss Mean: 0.3671002686023712
Epoch: 55,   AUC: 0.6963051754221425,   Loss Mean: 0.3661898672580719
Epoch: 56,   AUC: 0.6960595984560674,   Loss Mean: 0.3650393486022949
Epoch: 57,   AUC: 0.6959679069428755,   Loss Mean: 0.36482101678848267
Epoch: 58,   AUC: 0.6958162793022458,   Loss Mean: 0.3647066354751587
Epoch: 59,   AUC: 0.695561146517603,   Loss Mean: 0.36482182145118713
Epoch: 60,   AUC: 0.6959591947619406,   Loss Mean: 0.36457589268684387
Epoch: 61,   AUC: 0.6958493028484153,   Loss Mean: 0.36279264092445374
Epoch: 62,   AUC: 0.6958511523616864,   Loss Mean: 0.36169520020484924
Epoch: 63,   AUC: 0.6961419748037346,   Loss Mean: 0.3611203134059906
Epoch: 64,   AUC: 0.6959419704936116,   Loss Mean: 0.3618166446685791
Epoch: 65,   AUC: 0.6957922518535219,   Loss Mean: 0.3605306148529053
Epoch: 66,   AUC: 0.6958968926635785,   Loss Mean: 0.36036714911460876
Epoch: 67,   AUC: 0.6956749159194773,   Loss Mean: 0.3590988516807556
Epoch: 68,   AUC: 0.6956753485541606,   Loss Mean: 0.36001718044281006
Epoch: 69,   AUC: 0.6954434779956458,   Loss Mean: 0.35933560132980347
Epoch: 70,   AUC: 0.6953656740557876,   Loss Mean: 0.3589036464691162
Epoch: 71,   AUC: 0.6955995266219455,   Loss Mean: 0.3576129674911499
Epoch: 72,   AUC: 0.6955887621302315,   Loss Mean: 0.3586527407169342
Epoch: 73,   AUC: 0.695490994803706,   Loss Mean: 0.35775357484817505
Epoch: 74,   AUC: 0.695565721629379,   Loss Mean: 0.3566918671131134
Epoch: 75,   AUC: 0.6955539783016942,   Loss Mean: 0.3561679720878601
Epoch: 76,   AUC: 0.695404914021563,   Loss Mean: 0.3575153350830078
Epoch: 77,   AUC: 0.695548205332639,   Loss Mean: 0.35665279626846313
Epoch: 78,   AUC: 0.6956168617529119,   Loss Mean: 0.3564140796661377
Epoch: 79,   AUC: 0.6956226833933691,   Loss Mean: 0.3561438024044037
Epoch: 80,   AUC: 0.6954004443644912,   Loss Mean: 0.3546167314052582
Epoch: 81,   AUC: 0.6952384388993957,   Loss Mean: 0.35474449396133423
Epoch: 82,   AUC: 0.6953748756547082,   Loss Mean: 0.3545520305633545
Epoch: 83,   AUC: 0.6954794515695621,   Loss Mean: 0.35380131006240845
Epoch: 84,   AUC: 0.6948214439598965,   Loss Mean: 0.35403895378112793
Epoch: 85,   AUC: 0.6951240854406991,   Loss Mean: 0.35288724303245544
Epoch: 86,   AUC: 0.6951112091509375,   Loss Mean: 0.354536235332489
Epoch: 87,   AUC: 0.6950106972981396,   Loss Mean: 0.3535253703594208
Epoch: 88,   AUC: 0.6948703668306775,   Loss Mean: 0.3531471788883209
Epoch: 89,   AUC: 0.6947758631920441,   Loss Mean: 0.35417208075523376
Epoch: 90,   AUC: 0.6947117602518128,   Loss Mean: 0.3531905710697174
Epoch: 91,   AUC: 0.6945971985876749,   Loss Mean: 0.35145100951194763
Epoch: 92,   AUC: 0.6947259885249599,   Loss Mean: 0.35267186164855957
Epoch: 93,   AUC: 0.6947474688369858,   Loss Mean: 0.35250964760780334
Epoch: 94,   AUC: 0.6944867631768289,   Loss Mean: 0.35261744260787964
Epoch: 95,   AUC: 0.694599907962379,   Loss Mean: 0.3528944253921509
Epoch: 96,   AUC: 0.6947686949761351,   Loss Mean: 0.35202035307884216
Epoch: 97,   AUC: 0.6943440424027416,   Loss Mean: 0.35329699516296387
Epoch: 98,   AUC: 0.6944678976006703,   Loss Mean: 0.3530251085758209
Epoch: 99,   AUC: 0.6944717886088532,   Loss Mean: 0.352465957403183
Epoch: 100,   AUC: 0.6945430597649934,   Loss Mean: 0.35258522629737854
(pt) ➜  knowledge-tracing-collection-pytorch git:(main) ✗

"""
