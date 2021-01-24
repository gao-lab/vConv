'''
final version for vConv explain
by AuAs April 24th, 2018
'''
import numpy as np
import pickle
import random
import math
import random
import os
import glob
import matplotlib
from pprint import pprint
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# root for code and result saving, using relative address instead
explain_root = "../OutputAnalyse/"
# save the raw data of scoring under this dir, in .pkl file
explain_save_root = explain_root + "TheoreticalAnalysis/raw_result/"
# save the visualized result under this dir
explain_img_dir = explain_root + "TheoreticalAnalysis/explain_img/"
# the dir to save motif pwm
# according to the result in NIPS manuscript, there are 2 motifs to analysis:
# "simuMtf_Len-8_totIC-10.txt", "simuMtf_Len-23_totIC-12.txt"
# and there are 3 data set generated from these two motifs, called [simu_01,simu_02,simu_03]
# details about these data set refer to the manuscript
explain_mtf_dir = explain_root + "mtf_pwm/"
# the dir saves the CNN AUC result in practice
explain_model_root = explain_root + "result/ICSimulation/"
# save the average AUC bar plot for CNN, with respect to different kernel length
CNN_average_AUC_dir = explain_root + "ModelAUC/ICSimulation/"


##################  help functions  #######################

# make sure the kernel end up with '/'
def check_dir_last(str):
    def mkdir(path):
        isExists=os.path.exists(path)
        if not isExists:
            os.makedirs(path)
            return(True)
        else :
            return(False)
    if str[-1] == "/":
        return str
    else:
        return str+"/"

# make a dir
def mkdir(path):
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return(True)
    else :
        return(False)

######################### theoretical scoring kernel length  #############################

class explain_core(object):
    '''
    core code for theoretical scoring
    motif is save in file (check_dir_last.mtf_dir)+mtf_name +".txt")
    generate the p_real for every kernel length in kerLen_lst, according
    to different p_ideal in p_ideal_lst
    '''
    def __init__(self,mtf_name,mtf_dir,p_ideal_lst,kerLen_lst,save_dir,seed=233):
        self.mtf_name = mtf_name
        self.mtf_dir = mtf_dir
        # load motif:
        self.mtf = np.array(np.loadtxt(check_dir_last(self.mtf_dir)+self.mtf_name +".txt"))
        self.p_ideal_lst = p_ideal_lst
        self.kerLen_lst = kerLen_lst
        self.save_dir = save_dir
        self.seed = seed
        self.base_sample_num = 50000
        self.XsXn_sample_num = 5000
        self.seq_len = 1000
        self.p_real_num = 50
        np.random.seed(seed)
        random.seed(seed)
        random.seed(seed)
        # background is defaulted be [0.25,0.25,0.25,0.25]
        self.background_dist = np.ones(4)*0.25
        self.non_Entropy = self.cal_Entropy(self.background_dist)
        self.IC_arr = np.array([self.non_Entropy - self.cal_Entropy(col) for col in self.mtf])
        return
    # given a normalized column b, generate a sample (an int) according to the distribution of b
    def generate_a_sampe(self,b):
        # make b into accumulated form
        acc_b = []
        tmp_b = 0
        for idx in range(4):
            tmp_b = tmp_b+b[idx]
            acc_b.append(tmp_b)
        r = random.uniform(0,1) # generate one between 0,1 in uniform
        j = [idx for idx,item in enumerate(acc_b) if item >r][0]
        return j
    # calculate the entropy of a normalized column
    def cal_Entropy(self,col):
        return np.array([-p*math.log(p,2.) for p in col if not p==0]).sum()
    # generate the convolution result of "a" and "b"
    # "a" is a column in kernel, and b is the distribution,
    # where a column of sequence is generated from
    def gen_conv_sample(self,a,b):
        b_sample_lst = [self.generate_a_sampe(b) for idx in range(self.base_sample_num)]
        ret = [a[b_sample] for b_sample in b_sample_lst]
        return ret
    # to be implemented in child classes
    # support using different distribution to generate random column
    def gen_rand_conv_sample(self):
        raise NotImplemented
    # simplify the computational stress using the linear property of conv
    # short_cut_dic is a dict
    # key is a mtf column
    # value is a dict {"s","r"} # each is a list of sample for conv(mtf[i],samp_mtf[i]) and conv(mtf[i],samp_r)
    def init_short_cut(self):
        self.short_cut_dic = {}
        for mtf_idx in range(self.mtf.shape[0]):
            self.short_cut_dic[tuple(self.mtf[mtf_idx])] = {}
            self.short_cut_dic[tuple(self.mtf[mtf_idx])]["s"] = \
                self.gen_conv_sample(np.array(self.mtf[mtf_idx]),np.array(self.mtf[mtf_idx]))
            self.short_cut_dic[tuple(self.mtf[mtf_idx])]["r"] = \
                self.gen_conv_sample(np.array(self.mtf[mtf_idx]),np.array(self.background_dist))
        self.short_cut_dic[tuple(self.background_dist)] = {}
        self.short_cut_dic[tuple(self.background_dist)]["r"] = \
            self.gen_rand_conv_sample()
    # sort for the save tmplate for save
    def sort_tmplate_data(self):
        ret = {}
        for p_ideal in self.p_ideal_lst:
            for kerLen in self.kerLen_lst:
                ret[(p_ideal,kerLen)] = {}
                ret[(p_ideal, kerLen)]["p_real_lst"] = []
        self.result = ret
        return
    # save the result in save_dir + save_name
    # save name is organized in format: "mtf-{0}_seed-{1}.pkl"
    # which record the motif name and random seed
    def save_result(self):
        self.save_name =  "mtf-{0}_seed-{1}.pkl".format(self.mtf_name,self.seed)
        tmp_save_dir = check_dir_last(self.save_dir)
        self.save_path = tmp_save_dir + self.save_name
        self.sf = open(self.save_path,"w")
        pickle.dump(self.result,self.sf)
        self.sf.close()
        return
    def run_simu(self):
        self.sort_tmplate_data()
        self.init_short_cut()
        return

# using uni distribution to generate the random signal
# (this is consist with the generation of simulation data)
# using normal distribution to approx a single convolution result
# the using extreme distribution to approx the max pooling of noise
class explain_vCNN_uni(explain_core):
    def __init__(self,mtf_name,mtf_dir,p_ideal_lst,kerLen_lst,save_dir,seed):
        super(explain_vCNN_uni,self).__init__(mtf_name,mtf_dir,p_ideal_lst,kerLen_lst,save_dir,seed)
        return
    # using Normal distribution to approximate
    def get_XsXnc_paras(self,p_ideal,kerLen):
        mtfLen = self.mtf.shape[0]
        # if kernel length is shorter that motif's, search for the highest IC with in the kerLen window in mtf
        if kerLen<mtfLen:
            rand_num = 0
            best_idx = 0
            best_IC = -1
            for idx in range(mtfLen-kerLen):
                tmp_IC = np.array(self.IC_arr[idx:idx+kerLen]).sum()
                if tmp_IC>best_IC:
                    best_IC = tmp_IC
                    best_idx = idx
            sig_col_lst = self.mtf[best_idx:best_idx+kerLen]
        else:
            sig_col_lst = self.mtf
            rand_num = kerLen-mtfLen
        # sig-sig
        Xs_col_lst = []
        #sig-rand
        Xn_c_col_lst = []
        # rand-rand
        rand_col_lst = []
        for sig_col in sig_col_lst:
            sig_samp = self.short_cut_dic[tuple(sig_col)]
            # get the index
            idx_lst = [random.randint(0,self.base_sample_num-1) for idx in range(self.XsXn_sample_num)]
            Xs_col_lst.append(p_ideal*np.array(sig_samp["s"])[idx_lst]+
                              (1-p_ideal)*np.array(sig_samp["r"])[idx_lst])
            idx_lst = [random.randint(0,self.base_sample_num-1) for idx in range(self.XsXn_sample_num)]
            Xn_c_col_lst.append(np.array(sig_samp["r"])[idx_lst])
        if not rand_num == 0:
            rand_samp = np.array(self.short_cut_dic[tuple(self.background_dist)]["r"])
            for idx in range(rand_num):
                idx_lst = [random.randint(0, self.base_sample_num - 1) for idx in range(self.XsXn_sample_num)]
                rand_col_lst.append(rand_samp[idx_lst])

            rand_col_lst = np.array(rand_col_lst)
            Xs_col_lst = np.concatenate([rand_col_lst,Xs_col_lst],axis=0)
            Xn_c_col_lst = np.concatenate([rand_col_lst,Xn_c_col_lst],axis=0)
        Xs_col_lst = np.array(Xs_col_lst).sum(axis = 0)
        Xn_c_col_lst = np.array(Xn_c_col_lst).sum(axis=0)
        # using Normal distribution to approximate
        return [[Xs_col_lst.mean(),Xs_col_lst.std()],[Xn_c_col_lst.mean(),Xn_c_col_lst.std()]]
    def gen_rand_conv_sample(self):
        def gen_a_sample():
            r = np.random.uniform(0,1,4)
            r = np.array([r[idx]*self.background_dist[idx] for idx in range(4)])
            tmp_idx = np.random.randint(0,4)
            return r[tmp_idx]/r.sum()
        return [gen_a_sample() for idx in range(self.base_sample_num)]

    # using extreme dist to approxmate max pooling result
    # paras = [mean,std] for normal distribution
    def extreme_approx(self,paras,kerLen):
        def get_m_from_n(n, prec=0.001):
            save_n = n
            const = 2.50663  # sqirt(2pi)
            n = math.log(n / const, math.e)
            if n < 0:
                print("error: too small the n! n = {0}".format(save_n))
            ceiling = math.sqrt(2 * n)
            flour = 1.00001

            def cal_result(m, n):
                return n - math.log(m, math.e) - 0.5 * m * m

            # r = solve(exp(0.5*m*m)*m*const-n)
            m = -1
            while abs(ceiling - flour) > prec:
                m = 0.5 * (ceiling + flour)
                res = cal_result(m, n)
                if res > 0:
                    flour = m
                else:
                    ceiling = m
            return m
        m = get_m_from_n(n=self.seq_len-kerLen+1)
        c = m / (1 + m * m)
        mean,std = paras
        pre_ret = np.random.gumbel(loc=m, scale=c, size=self.XsXn_sample_num)
        ret = pre_ret * std + mean
        return ret


    def call_a_preal(self,Xs_paras,Xnc_paras,kerLen):
        def larger_hp(x1, x2):
            '''
            given two sample sets: x1, x2,
            which generate from two distribution ,return the probability of x1>x2
            :param x1:
            :param x2:
            :return: probability of x1>x2
            '''
            ret = 0
            x1 = np.array(x1)
            x1.sort()
            x2 = np.array(x2)
            x2.sort()

            def cal_p2(tmp_x1):
                '''
                given a tmp_x1, return the probability of x2<tmp_x1
                :param tmp_x1:
                :return:
                '''
                lst = [idx for idx, item in enumerate(x2) if item < tmp_x1]
                return (len(lst)*1.0 / len(x2))

            interval = 1.0 / (1.0 * len(x1))
            for idx, tmp_x1 in enumerate(x1):
                p2 = cal_p2(tmp_x1)
                ret = ret + interval * p2
            return ret
        Xs_mean,Xs_std = Xs_paras
        Xs_lst = np.random.normal(Xs_mean,Xs_std,self.XsXn_sample_num)
        Xn_lst = np.array(self.extreme_approx(Xnc_paras,kerLen))
        return larger_hp(Xs_lst,Xn_lst)
    def run_simu(self):
        super(explain_vCNN_uni,self).run_simu()
        for tmp_key in self.result.keys():
            print(tmp_key)
            for p_real_epoch in range(self.p_real_num):
                p_ideal, kerLen = tmp_key
                Xs_paras, Xnc_paras = self.get_XsXnc_paras(p_ideal, kerLen)
                tmp_p_real = self.call_a_preal(Xs_paras, Xnc_paras, kerLen)
                self.result[tmp_key]["p_real_lst"].append(tmp_p_real)


###### codes for visualization ###########

# for the sake of demonstration, generate a motif from give IC and length
def gen_demo_mtf(mtf_dir,mtfLen,mtf_IC=10,background_dist = np.ones(4)*0.25):
    # unit for ic is bit, therefore, we choose base of log to be 2
    def cal_Entropy(col):
        return np.array([-p*math.log(p,2.) for p in col if not p==0]).sum()
    non_IC = cal_Entropy(background_dist)
    # calculate frq from IC
    def frq2IC(frq):
        minor_frq = (1.-frq)/3.
        col = np.ones(4)*minor_frq
        col[0] = frq
        return non_IC - cal_Entropy(col)
    def IC2frq(IC,prec = 0.00001):
        high = 1.
        low = 0.25
        tmp_frq = 0.5*(high+low)
        while high-low>prec:
            tmp_IC = frq2IC(tmp_frq)
            if tmp_IC>IC:
                high = tmp_frq
                tmp_frq = 0.5 * (high + low)
            else:
                low = tmp_frq
                tmp_frq = 0.5*(high+low)
        return tmp_frq
    def gen_IC_dist(tot_IC,tot_len):
        ret = []
        rest_IC = tot_IC*1.
        for idx in range(tot_len):
            min_IC = rest_IC/(tot_len-idx)
            max_IC = min(min_IC*1.5,2.)
            tmp_IC = random.uniform(0,max_IC)
            print("tmp_IC",tmp_IC)
            if rest_IC>(tot_len-idx-1)*2.:
                tmp_IC = random.uniform(min_IC,max_IC)
            pre_rest_IC = rest_IC
            rest_IC = max(0.,rest_IC - tmp_IC)
            ret.append(pre_rest_IC-rest_IC)
        idx_lst = np.arange(tot_len)
        random.shuffle(idx_lst)
        ret = np.array(ret)
        # return ret
        return ret[idx_lst]



    tot_IC = min(mtfLen*2.,mtf_IC)
    save_path = mtf_dir + "simuMtf_Len-{0}_totIC-{1}.txt".format(mtfLen,tot_IC)
    IC_lst = gen_IC_dist(tot_IC,mtfLen)
    mtf = []
    print(save_path)
    print(np.array(IC_lst).sum())
    for mtf_idx in range(mtfLen):
        major_frq = IC2frq(IC_lst[mtf_idx])
        minor_frq = (1-major_frq)/3.
        tmp = np.ones(4)*minor_frq
        major_idx = random.randint(0,3)
        tmp[major_idx] = major_frq
        mtf.append(tmp)
    np.savetxt(save_path,np.array(mtf))


class visu_core(object):
    '''
    all result are saved in .pkl file under raw_data dir
    each .pkl file records the data for a motif and a certain random seed
    finally, the result are sorted in a dict, in format: "sorted_data_dict[mtf_name][kerLen][p_ideal]"
    when calling method "load_sorted_data", it will search for file "sorted_data.pkl" under the file_dir
    if not have the file under the dir, it will calculate from the raw data and save it as "sorted_data.pkl"
    '''
    def __init__(self,raw_data_dir):
        self.raw_data_dir = raw_data_dir
        self.fp_lst = glob.glob(check_dir_last(self.raw_data_dir) + "*.pkl")
        self.raw_data_dict = {}
        self.sorted_data_dict = {}
    def sort_raw_data(self):
        # sort all the data in a big lst:
        for fp in self.fp_lst:
            f = open(fp,"r")
            tmp_data = pickle.load(f)
            tmp_key = check_dir_last(fp).split("/")[-2]
            self.raw_data_dict[tmp_key] = tmp_data
        for pre_name in self.raw_data_dict.keys():
            mtf_name = pre_name.split("simu")[-1].split("_seed")[0]
            mtf_name = mtf_name.replace("tot","")
            tmp_data = self.raw_data_dict[pre_name]
            if mtf_name not in self.sorted_data_dict:
                self.sorted_data_dict[mtf_name] = {}
            for tmp_key in tmp_data.keys():
                p_ideal, kerLen = tmp_key
                if kerLen not in self.sorted_data_dict[mtf_name]:
                    self.sorted_data_dict[mtf_name][kerLen] = {}
                self.sorted_data_dict[mtf_name][kerLen][p_ideal] = np.array(tmp_data[tmp_key])

    def load_sorted_data(self,file_dir):
        file_path = check_dir_last(file_dir) + "sorted_data.pkl"
        if os.path.exists(file_path):
            print("loaded sorted data from file :  ",file_path)
            self.sorted_data_dict = pickle.load(open(file_path,"r"))
        else:
            print("calculated the sorted data and save to file :  ",file_path)
            self.sort_raw_data()
            f = open(file_path,"w")
            pickle.dump(self.sorted_data_dict,f)
            f.close()


# calculate the average p_real and draw the plot
class visu_detail_AvePreal(visu_core):
    def __init__(self,raw_data_dir,save_dir):
        self.save_pkl_dir = save_dir
        self.save_dir = save_dir + "detail_Ave/"
        mkdir(self.save_dir)
        super(visu_detail_AvePreal,self).__init__(raw_data_dir)
        self.load_sorted_data(self.save_pkl_dir)
    def cal_AvePreal(self,p_real):
        p_real = np.array(p_real)
        return p_real.mean()
    def draw_plot(self):
        save_dir = check_dir_last(self.save_dir)
        for mtf_name in self.sorted_data_dict:
            save_path = save_dir + mtf_name+".png"
            tmp_data = self.sorted_data_dict[mtf_name]
            plt.clf()
            plt.title(mtf_name)
            plt.xlabel("p_ideal")
            plt.ylabel("Ave_p_real")
            plt.xlim(-0.01,1.1)
            ker_len_lst = np.array(tmp_data.keys())
            ker_len_lst.sort()
            for ker_len in ker_len_lst:
                Ave_lst = []
                p_ideal_lst = []
                for p_ideal in tmp_data[ker_len]:
                    p_ideal_lst.append(p_ideal)
                    p_real = tmp_data[ker_len][p_ideal].tolist()["p_real_lst"]
                    Ave_lst.append(self.cal_AvePreal(p_real))
                p_ideal_lst = np.array(p_ideal_lst)
                Ave_lst = np.array(Ave_lst)
                idx_lst = p_ideal_lst.argsort()
                plt.plot(p_ideal_lst[idx_lst],Ave_lst[idx_lst],label = "kerLen: {0}".format(ker_len))
            plt.legend(loc = "best")
            plt.savefig(save_path)

# calculate the theoretical score
def run_preal_plot():
    demo_mtf_name_lst = ["simuMtf_Len-8_totIC-10", "simuMtf_Len-23_totIC-12"]
    p_ideal_lst = (np.arange(19) + 1) * 0.05
    kerLen_lst = [4,6,8,16,24,32] # combine extra explain together

    for demo_mtf_name in demo_mtf_name_lst:
        print("demo_mtf_name: ",demo_mtf_name)
        auas_explain = explain_vCNN_uni(mtf_name=demo_mtf_name,
                                    mtf_dir=explain_mtf_dir,
                                    p_ideal_lst=p_ideal_lst,
                                    kerLen_lst=kerLen_lst,
                                    save_dir=explain_save_root,
                                    seed=233)
        auas_explain.run_simu()
        auas_explain.save_result()
'''
further discover the auc of CNN with different kernel length
draw the average AUC of different kernel lengths
each have many random seeds

Requirement:
CNN training result are saved in .pkl files
file name is in format like: "Report_KernelNum-100_KernelLen-8_seed-123_batch_size-100.pkl"
this is critical to decode model's hyper-parameter from file name.
'''
def draw_detail_CNN_kerLen(data_info,model_root,output_dir):
    print("dealing with data_info: ",data_info)
    model_dir = check_dir_last(check_dir_last(model_root)+data_info) + "CNN/"
    save_path  = check_dir_last(output_dir) + data_info + "_kerLen_auc.png"
    plt.clf()
    plt.title(data_info + " kerLen auc")
    plt.xlabel("kerLen")
    plt.ylabel("average_auc")
    # plt.ylim([0.3,1])
    def decode_CNN_fileName(fp):
        # Report_KernelNum-100_KernelLen-8_seed-123_batch_size-100.pkl
        KernelNum = 0
        KernelLen = 0
        seed = 0
        batch_size = 100 # preset
        r =  check_dir_last(fp).split("/")[-2]
        r = r.split("_")
        for it in r:
            tmp_it = it.split("-")
            if tmp_it[0] == "KernelNum":
                KernelNum = int(tmp_it[1])
            elif tmp_it[0] == "KernelLen":
                KernelLen = int(tmp_it[1])
            elif tmp_it[0] == "seed":
                seed = int(tmp_it[1])
        return [KernelNum,KernelLen,seed,batch_size]
    def get_auc(fp):
        f = open(fp,"r")
        d = pickle.load(f).tolist()
        f.close()
        return d["test_auc"]
    def get_simu_data(model_dir):
        ret = {}
        lst = glob.glob(check_dir_last(model_dir)+"*.pkl")
        for fp in lst:
            KernelNum, KernelLen, seed, batch_size = decode_CNN_fileName(fp)
            if KernelLen not in ret:
                ret[KernelLen] = []
            ret[KernelLen].append(get_auc(fp))
        return ret

    auc_data = get_simu_data(model_dir)
    kerLen_lst = np.array(auc_data.keys())
    kerLen_lst.sort()
    y = []
    for kerLen in kerLen_lst:
        y.append(np.array(auc_data[kerLen]).mean())
    plt.bar(np.arange(len(kerLen_lst)),y,width=0.2)
    plt.ylim(0.3,1,1)
    plt.xticks(np.arange(len(kerLen_lst)),kerLen_lst)
    plt.savefig(save_path)
    return

# analysis the CNN on simulation dataset with different kernel length
def run_draw_detail_CNN_kerLen(output_dir = CNN_average_AUC_dir,model_dir = explain_model_root):
    mkdir(output_dir)
    data_info_lst = ["simu_01", "simu_02", "simu_03"]
    for data_info in data_info_lst:
        draw_detail_CNN_kerLen(data_info, model_dir, output_dir)

# run_preal_plot()


if __name__ == "__main__":
    import sys
    if len(sys.argv)<2:
        print("please pass the parameter to specify the operation")
        print(" Using \"ipython sorted_explain.py [parameter]\" to regenerate the result.")
        print("Set [parameter] as \"run_preal_plot\"  to generate the theoretical score.")
        print("Set [parameter] as \"draw_detail_CNN_kerLen\" to draw average AUCs for different kernel lengths and data sets.")
        print("Set [parameter] as  \"visu_detail_AvePreal\" to further visualize the p_real.")
        exit(1)
    if sys.argv[1]=="run_preal_plot":
        run_preal_plot()
    elif sys.argv[1]=="draw_detail_CNN_kerLen":
        run_draw_detail_CNN_kerLen()
    elif sys.argv[1]== "visu_detail_AvePreal":
        mkdir(explain_img_dir)
        ave_visu = visu_detail_AvePreal(explain_save_root,explain_img_dir)
        ave_visu.draw_plot()
    else:
        print("not supported parameters")
        print("please pass the parameter to specify the operation")
        print(" Using \"ipython sorted_explain.py [parameter]\" to regenerate the result.")
        print("Set [parameter] as \"run_preal_plot\"  to generate the theoretical score.")
        print("Set [parameter] as \"draw_detail_CNN_kerLen\" to draw average AUCs for different kernel lengths and data sets.")
        print("Set [parameter] as  \"visu_detail_AvePreal\" to further visualize the p_real.")
        exit(1)
