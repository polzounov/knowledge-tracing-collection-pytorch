import os
import pickle
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from models.utils import match_seq_len


DATASET_DIR = "datasets/simulated/"


class Simulated(Dataset):
    def __init__(self, seq_len, dataset_dir=DATASET_DIR, dataset_path=None) -> None:
        super().__init__()

        if dataset_path is None:
            self.dataset_dir = dataset_dir
            self.dataset_path = os.path.join(self.dataset_dir, "data.csv")
        else:
            self.dataset_path = dataset_path

        if False:  # os.path.exists(os.path.join(self.dataset_dir, "q_seqs.pkl")):
            # with open(os.path.join(self.dataset_dir, "q_seqs.pkl"), "rb") as f:
            #     self.q_seqs = pickle.load(f)
            # with open(os.path.join(self.dataset_dir, "r_seqs.pkl"), "rb") as f:
            #     self.r_seqs = pickle.load(f)
            # with open(os.path.join(self.dataset_dir, "q_list.pkl"), "rb") as f:
            #     self.q_list = pickle.load(f)
            # with open(os.path.join(self.dataset_dir, "u_list.pkl"), "rb") as f:
            #     self.u_list = pickle.load(f)
            # with open(os.path.join(self.dataset_dir, "q2idx.pkl"), "rb") as f:
            #     self.q2idx = pickle.load(f)
            # with open(os.path.join(self.dataset_dir, "u2idx.pkl"), "rb") as f:
            #     self.u2idx = pickle.load(f)
            pass
        else:
            (
                self.q_seqs,
                self.r_seqs,
                self.q_list,
                self.u_list,
                self.q2idx,
                self.u2idx,
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
        df = pd.read_csv(self.dataset_path, encoding="unicode_escape")

        u_list = np.unique(df["user_id"].values)
        # q_list = np.unique(df["skill_name"].values)
        q_list = np.unique(df["problem_id"].values)

        u2idx = {u: idx for idx, u in enumerate(u_list)}
        q2idx = {q: idx for idx, q in enumerate(q_list)}

        q_seqs = []
        r_seqs = []

        for u in u_list:
            df_u = df[df["user_id"] == u]

            # q_seq = np.array([q2idx[q] for q in df_u["skill_name"]])
            q_seq = np.array([q2idx[q] for q in df_u["problem_id"]])
            r_seq = df_u["correct"].values

            q_seqs.append(q_seq)
            r_seqs.append(r_seq)

        # with open(os.path.join(self.dataset_dir, "q_seqs.pkl"), "wb") as f:
        #     pickle.dump(q_seqs, f)
        # with open(os.path.join(self.dataset_dir, "r_seqs.pkl"), "wb") as f:
        #     pickle.dump(r_seqs, f)
        # with open(os.path.join(self.dataset_dir, "q_list.pkl"), "wb") as f:
        #     pickle.dump(q_list, f)
        # with open(os.path.join(self.dataset_dir, "u_list.pkl"), "wb") as f:
        #     pickle.dump(u_list, f)
        # with open(os.path.join(self.dataset_dir, "q2idx.pkl"), "wb") as f:
        #     pickle.dump(q2idx, f)
        # with open(os.path.join(self.dataset_dir, "u2idx.pkl"), "wb") as f:
        #     pickle.dump(u2idx, f)

        return q_seqs, r_seqs, q_list, u_list, q2idx, u2idx


"""Doing it wrong?? Using skill_name instead of problem_id

(pt) ➜  knowledge-tracing-collection-pytorch git:(main) ✗ p train.py --dataset-name simulated
Epoch: 1,   AUC: 0.6986889885038089,   Loss Mean: 0.5806016325950623
Epoch: 2,   AUC: 0.7178548409700103,   Loss Mean: 0.5378718972206116
Epoch: 3,   AUC: 0.73821584292732,   Loss Mean: 0.5312967896461487
Epoch: 4,   AUC: 0.7700102582821725,   Loss Mean: 0.5259580016136169
Epoch: 5,   AUC: 0.7888397019602278,   Loss Mean: 0.518903374671936
Epoch: 6,   AUC: 0.7926835360052719,   Loss Mean: 0.5144501328468323
Epoch: 7,   AUC: 0.7977755448299222,   Loss Mean: 0.5122776627540588
Epoch: 8,   AUC: 0.7976239352642635,   Loss Mean: 0.5111778378486633
Epoch: 9,   AUC: 0.7981836203900413,   Loss Mean: 0.5110293030738831
Epoch: 10,   AUC: 0.8012756876985785,   Loss Mean: 0.5111129283905029
Epoch: 11,   AUC: 0.7985903153360445,   Loss Mean: 0.5096561312675476
Epoch: 12,   AUC: 0.8004243724262083,   Loss Mean: 0.5098045468330383
Epoch: 13,   AUC: 0.8025572283836239,   Loss Mean: 0.5093474984169006
Epoch: 14,   AUC: 0.8042021758233252,   Loss Mean: 0.5087551474571228
Epoch: 15,   AUC: 0.8054425241428922,   Loss Mean: 0.5091610550880432
Epoch: 16,   AUC: 0.8036670543897695,   Loss Mean: 0.5081729292869568
Epoch: 17,   AUC: 0.805344271928331,   Loss Mean: 0.5075156688690186
Epoch: 18,   AUC: 0.8053305475256576,   Loss Mean: 0.5095053315162659
Epoch: 19,   AUC: 0.8061959601325439,   Loss Mean: 0.5080311894416809
Epoch: 20,   AUC: 0.8063648561108387,   Loss Mean: 0.5068845152854919
Epoch: 21,   AUC: 0.8052847995167457,   Loss Mean: 0.507168173789978
Epoch: 22,   AUC: 0.8056396096815641,   Loss Mean: 0.5063942074775696
Epoch: 23,   AUC: 0.806358945396566,   Loss Mean: 0.5070432424545288
Epoch: 24,   AUC: 0.8071933740085837,   Loss Mean: 0.5056472420692444
Epoch: 25,   AUC: 0.8074930490102328,   Loss Mean: 0.5077143311500549
Epoch: 26,   AUC: 0.8075643747794594,   Loss Mean: 0.5052270293235779
Epoch: 27,   AUC: 0.8063789585540102,   Loss Mean: 0.5035502910614014
Epoch: 28,   AUC: 0.8049880990296906,   Loss Mean: 0.5067989826202393
Epoch: 29,   AUC: 0.8060385905579461,   Loss Mean: 0.5056517720222473
Epoch: 30,   AUC: 0.8053491136564658,   Loss Mean: 0.5053260326385498
Epoch: 31,   AUC: 0.8041964822269063,   Loss Mean: 0.505673885345459
Epoch: 32,   AUC: 0.804693046083995,   Loss Mean: 0.5063468813896179
Epoch: 33,   AUC: 0.8059027092650328,   Loss Mean: 0.5030176043510437
Epoch: 34,   AUC: 0.8036636098788807,   Loss Mean: 0.505357027053833
Epoch: 35,   AUC: 0.8033638071608472,   Loss Mean: 0.5042595267295837
Epoch: 36,   AUC: 0.8035573013147881,   Loss Mean: 0.5042157769203186
Epoch: 37,   AUC: 0.802688492729239,   Loss Mean: 0.5029603838920593
Epoch: 38,   AUC: 0.8036275657608598,   Loss Mean: 0.5019087195396423
Epoch: 39,   AUC: 0.8021025018597548,   Loss Mean: 0.5031313896179199
Epoch: 40,   AUC: 0.8018392860543762,   Loss Mean: 0.5019769668579102
Epoch: 41,   AUC: 0.7999714607521624,   Loss Mean: 0.5005068182945251
Epoch: 42,   AUC: 0.7988873747061399,   Loss Mean: 0.5000132918357849
Epoch: 43,   AUC: 0.798689854931761,   Loss Mean: 0.49978819489479065
Epoch: 44,   AUC: 0.7985645230122038,   Loss Mean: 0.49986258149147034
Epoch: 45,   AUC: 0.7952284956975721,   Loss Mean: 0.498693585395813
Epoch: 46,   AUC: 0.7959966880382836,   Loss Mean: 0.4984089732170105
Epoch: 47,   AUC: 0.7957191335144222,   Loss Mean: 0.49882832169532776

"""


"""Actually use problem_id

(pt) ➜  knowledge-tracing-collection-pytorch git:(main) ✗ p train.py --dataset-name simulated
Epoch: 1,   AUC: 0.5858891090736806,   Loss Mean: 0.6283228397369385
Epoch: 2,   AUC: 0.57206541868042,   Loss Mean: 0.5487890243530273
Epoch: 3,   AUC: 0.5749115655186771,   Loss Mean: 0.5440208315849304
Epoch: 4,   AUC: 0.5850805825858829,   Loss Mean: 0.5423324704170227
Epoch: 5,   AUC: 0.6002613901160105,   Loss Mean: 0.5409175753593445
Epoch: 6,   AUC: 0.6264903541754103,   Loss Mean: 0.5383475422859192
Epoch: 7,   AUC: 0.6789596448871216,   Loss Mean: 0.5342023968696594
Epoch: 8,   AUC: 0.6907562434166911,   Loss Mean: 0.5325697064399719
Epoch: 9,   AUC: 0.6983111550038437,   Loss Mean: 0.5282415151596069
Epoch: 10,   AUC: 0.7051306445789591,   Loss Mean: 0.5256538987159729
Epoch: 11,   AUC: 0.7090991954120797,   Loss Mean: 0.5233500599861145
Epoch: 12,   AUC: 0.7117588438086969,   Loss Mean: 0.5220539569854736
Epoch: 13,   AUC: 0.711994868204018,   Loss Mean: 0.521001935005188
Epoch: 14,   AUC: 0.7127826141511515,   Loss Mean: 0.5203530192375183
Epoch: 15,   AUC: 0.7167460473704984,   Loss Mean: 0.5191914439201355
Epoch: 16,   AUC: 0.7159636570097848,   Loss Mean: 0.5185989737510681
Epoch: 17,   AUC: 0.71108109771253,   Loss Mean: 0.5162122249603271
Epoch: 18,   AUC: 0.7158869637142193,   Loss Mean: 0.5161945819854736
Epoch: 19,   AUC: 0.7202912100115814,   Loss Mean: 0.5162692666053772
Epoch: 20,   AUC: 0.717934254407662,   Loss Mean: 0.5158739686012268
Epoch: 21,   AUC: 0.7234989947458526,   Loss Mean: 0.5127949714660645
Epoch: 22,   AUC: 0.7176967396785975,   Loss Mean: 0.5110776424407959
Epoch: 23,   AUC: 0.7252526067619807,   Loss Mean: 0.5091267228126526
Epoch: 24,   AUC: 0.7295631988042717,   Loss Mean: 0.5060155987739563
Epoch: 25,   AUC: 0.7299023663134575,   Loss Mean: 0.5043004751205444
Epoch: 26,   AUC: 0.7344250316455544,   Loss Mean: 0.5032972693443298
Epoch: 27,   AUC: 0.7305132973826374,   Loss Mean: 0.5024955868721008
Epoch: 28,   AUC: 0.7321727563626435,   Loss Mean: 0.500077486038208
Epoch: 29,   AUC: 0.733172133306988,   Loss Mean: 0.49734675884246826
Epoch: 30,   AUC: 0.7286730609195551,   Loss Mean: 0.49534711241722107
Epoch: 31,   AUC: 0.7314550512939222,   Loss Mean: 0.4944469630718231
Epoch: 32,   AUC: 0.7378005522407649,   Loss Mean: 0.49009764194488525
Epoch: 33,   AUC: 0.7310583448834198,   Loss Mean: 0.4889298379421234
Epoch: 34,   AUC: 0.7324958991784415,   Loss Mean: 0.48649635910987854
Epoch: 35,   AUC: 0.7362657067955853,   Loss Mean: 0.4842962324619293
Epoch: 36,   AUC: 0.7274116604673591,   Loss Mean: 0.4800063371658325
Epoch: 37,   AUC: 0.7362550136509765,   Loss Mean: 0.4770199954509735
Epoch: 38,   AUC: 0.7325769120380987,   Loss Mean: 0.47453543543815613
Epoch: 39,   AUC: 0.7295595234489591,   Loss Mean: 0.4728780686855316
Epoch: 40,   AUC: 0.7276028823054597,   Loss Mean: 0.4681585431098938
Epoch: 41,   AUC: 0.7245116777263854,   Loss Mean: 0.4669555723667145
Epoch: 42,   AUC: 0.7213208852005533,   Loss Mean: 0.46299150586128235
Epoch: 43,   AUC: 0.7181833925380432,   Loss Mean: 0.4581659436225891
Epoch: 44,   AUC: 0.7241504424608584,   Loss Mean: 0.4556945562362671
Epoch: 45,   AUC: 0.7192890266726472,   Loss Mean: 0.4517582654953003
Epoch: 46,   AUC: 0.7195352814879876,   Loss Mean: 0.44799545407295227
Epoch: 47,   AUC: 0.7144189501223757,   Loss Mean: 0.44328972697257996
Epoch: 48,   AUC: 0.7163034062298834,   Loss Mean: 0.44078609347343445
Epoch: 49,   AUC: 0.7164366612966577,   Loss Mean: 0.4375838339328766
Epoch: 50,   AUC: 0.7036799308773595,   Loss Mean: 0.43451640009880066
Epoch: 51,   AUC: 0.7098532057319197,   Loss Mean: 0.4308849275112152
Epoch: 52,   AUC: 0.7091959373021401,   Loss Mean: 0.42904290556907654
Epoch: 53,   AUC: 0.7045195776971074,   Loss Mean: 0.4245370924472809
Epoch: 54,   AUC: 0.6994229812347752,   Loss Mean: 0.4206097424030304
Epoch: 55,   AUC: 0.6993444791067229,   Loss Mean: 0.4171261489391327
Epoch: 56,   AUC: 0.7010925155920158,   Loss Mean: 0.4149855673313141
Epoch: 57,   AUC: 0.7029079187681575,   Loss Mean: 0.414443701505661
Epoch: 58,   AUC: 0.6971171284537883,   Loss Mean: 0.4090209901332855
Epoch: 59,   AUC: 0.6990411156638403,   Loss Mean: 0.40691089630126953
Epoch: 60,   AUC: 0.6937191050209344,   Loss Mean: 0.40369370579719543

"""


""" More full version of the dataset (git: [main bc3de69] New data gen)
(pt) ➜  knowledge-tracing-collection-pytorch git:(main) ✗ p train.py --dataset-name simulated --dataset-path datasets/simulated/data-1.csv

Epoch: 1,   AUC: 0.6304304354425291,   Loss Mean: 0.6227496862411499
Epoch: 2,   AUC: 0.7032137181121447,   Loss Mean: 0.5958343744277954
Epoch: 3,   AUC: 0.7088925628206427,   Loss Mean: 0.5882705450057983
Epoch: 4,   AUC: 0.7185412359277646,   Loss Mean: 0.5853584408760071
Epoch: 5,   AUC: 0.7205392442532631,   Loss Mean: 0.5846168398857117
Epoch: 6,   AUC: 0.724049410053834,   Loss Mean: 0.5829036831855774
Epoch: 7,   AUC: 0.724854141009439,   Loss Mean: 0.5829893350601196
Epoch: 8,   AUC: 0.7277936180394863,   Loss Mean: 0.5820233225822449
Epoch: 9,   AUC: 0.7295680563085873,   Loss Mean: 0.5814183950424194
Epoch: 10,   AUC: 0.719937162458458,   Loss Mean: 0.5857495069503784
Epoch: 11,   AUC: 0.7247067165980274,   Loss Mean: 0.582547664642334
Epoch: 12,   AUC: 0.7317202145241682,   Loss Mean: 0.580822765827179
Epoch: 13,   AUC: 0.7368632303228217,   Loss Mean: 0.5790073275566101
Epoch: 14,   AUC: 0.7369838127987438,   Loss Mean: 0.577865481376648
Epoch: 15,   AUC: 0.7466270482062295,   Loss Mean: 0.5764923095703125
Epoch: 16,   AUC: 0.7545389957845844,   Loss Mean: 0.5733591318130493
Epoch: 17,   AUC: 0.7616519977086835,   Loss Mean: 0.5703660845756531
Epoch: 18,   AUC: 0.7718887980845448,   Loss Mean: 0.5666971802711487
Epoch: 19,   AUC: 0.778814484911615,   Loss Mean: 0.5637346506118774
Epoch: 20,   AUC: 0.7824880090929546,   Loss Mean: 0.5619558691978455
Epoch: 21,   AUC: 0.7897775098010951,   Loss Mean: 0.5581728219985962
Epoch: 22,   AUC: 0.7934517930451551,   Loss Mean: 0.5565412640571594
Epoch: 23,   AUC: 0.7956021688501941,   Loss Mean: 0.5549899935722351
Epoch: 24,   AUC: 0.797603131838961,   Loss Mean: 0.553114116191864
Epoch: 25,   AUC: 0.7989228065820314,   Loss Mean: 0.5514734983444214
Epoch: 26,   AUC: 0.7975290337744108,   Loss Mean: 0.5519169569015503
Epoch: 27,   AUC: 0.8001484556622973,   Loss Mean: 0.5502080321311951
Epoch: 28,   AUC: 0.801419627303255,   Loss Mean: 0.5495272874832153
Epoch: 29,   AUC: 0.8006395658251932,   Loss Mean: 0.5488694310188293
Epoch: 30,   AUC: 0.80119401127366,   Loss Mean: 0.5480186939239502
Epoch: 31,   AUC: 0.8019477535662844,   Loss Mean: 0.5478915572166443
Epoch: 32,   AUC: 0.8017075783076755,   Loss Mean: 0.5472243428230286
Epoch: 33,   AUC: 0.8000976988068493,   Loss Mean: 0.5468068718910217
Epoch: 34,   AUC: 0.8013664679519138,   Loss Mean: 0.5461136698722839
Epoch: 35,   AUC: 0.800538666639254,   Loss Mean: 0.5464708209037781
Epoch: 36,   AUC: 0.7998391802806397,   Loss Mean: 0.5455702543258667
Epoch: 37,   AUC: 0.800327832554758,   Loss Mean: 0.5457301139831543
Epoch: 38,   AUC: 0.7990484569143016,   Loss Mean: 0.5442065596580505
Epoch: 39,   AUC: 0.7987045599733874,   Loss Mean: 0.5437549948692322
Epoch: 40,   AUC: 0.7990072237710927,   Loss Mean: 0.5430088043212891
Epoch: 41,   AUC: 0.7997465436528551,   Loss Mean: 0.5429271459579468
Epoch: 42,   AUC: 0.7996760683917125,   Loss Mean: 0.5426223278045654
Epoch: 43,   AUC: 0.7965651221119018,   Loss Mean: 0.5417689681053162
^CTraceback (most recent call last):
  File "/Users/kirill/code/THESIS/knowledge-tracing-collection-pytorch/train.py", line 187, in <module>
    main(args.model_name, args.dataset_name, args.skill_id, args.dataset_path)
  File "/Users/kirill/code/THESIS/knowledge-tracing-collection-pytorch/train.py", line 142, in main
    aucs, loss_means = model.train_model(train_loader, test_loader, num_epochs, opt, ckpt_path)
  File "/Users/kirill/code/THESIS/knowledge-tracing-collection-pytorch/models/dkt.py", line 116, in train_model
    loss.backward()
  File "/Users/kirill/Applications/miniforge3/envs/pt/lib/python3.9/site-packages/torch/_tensor.py", line 487, in backward
    torch.autograd.backward(
  File "/Users/kirill/Applications/miniforge3/envs/pt/lib/python3.9/site-packages/torch/autograd/__init__.py", line 200, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
KeyboardInterrupt

(pt) ➜  knowledge-tracing-collection-pytorch git:(main) ✗
"""
