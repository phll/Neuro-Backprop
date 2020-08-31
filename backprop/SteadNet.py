import numpy as np
import time
import PyraLNet as pyral
import matplotlib.pyplot as plt
import Dataset
import fcntl
from pathlib import Path
import mnist


dtype = pyral.dtype

class Layer(pyral.Layer):

    def update(self, r_in, u_next, dir_up):
        #### input rates
        r_pyr = np.zeros(self.u_pyr["soma"].shape[0] + self.bias, dtype=dtype)
        r_in_buf = np.zeros(len(r_in) + self.bias, dtype=dtype)
        r_pyr[: len(self.u_pyr["soma"])] = self.act(self.u_pyr["soma"])
        r_in_buf[: len(r_in)] = r_in
        if self.bias:
            r_in_buf[-1] = self.bias_val
            r_pyr[-1] = self.bias_val

        #### compute potentials
        # if upwards pass compute pyramidials first, if downwards pass compute laterals first
        if dir_up:
            self.update_pyr(r_in_buf, u_next)
            r_pyr[: len(self.u_pyr["soma"])] = self.act(self.u_pyr["soma"])
            if self.bias:
                r_pyr[-1] = self.bias_val
            self.update_inn(r_pyr, u_next)
        else:
            self.update_inn(r_pyr, u_next)
            self.update_pyr(r_in_buf, u_next)

    def apply(self):
        pass

    def update_weights(self, r_in, n_exposures):
        # input rates
        r_pyr = np.zeros(self.u_pyr["soma"].shape[0] + self.bias, dtype=dtype)
        r_in_buf = np.zeros(len(r_in) + self.bias, dtype=dtype)
        r_pyr[: len(self.u_pyr["soma"])] = self.act(self.u_pyr["soma"])
        r_in_buf[: len(r_in)] = r_in
        if self.bias:
            r_in_buf[-1] = self.bias_val
            r_pyr[-1] = self.bias_val

        #discretised lowpass
        dT = (self.params["t_pattern"] - self.params["learning_lag"])/n_exposures
        a = dT/(self.params["tau_w"]+dT)
        self.Delta_up = self.Delta_up * (1 - a) + a * np.outer(
            self.act(self.u_pyr["soma"]) - self.act(self.gb / (self.gl + self.gb + self.ga) * self.u_pyr["basal"]), r_in_buf)
        self.Delta_ip = self.Delta_ip * (1 - a) + a * np.outer(
            self.act(self.u_inn["soma"]) - self.act(self.gd / (self.gl + self.gd) * self.u_inn["dendrite"]), r_pyr)
        self.Delta_pi = self.Delta_pi * (1 - a) + a * np.outer(-self.u_pyr["apical"], self.act(self.u_inn["soma"]))

        self.W_up += dT * self.eta["up"] * self.Delta_up
        self.W_ip += dT * self.eta["ip"] * self.Delta_ip
        self.W_pi += dT * self.eta["pi"] * self.Delta_pi

    def update_pyr(self, r_in, u_next):
        ####compute pyramidial potentials
        self.u_pyr["basal"][:] = np.matmul(self.W_up, r_in)  # [:] to enforce rhs to be copied to lhs
        self.u_pyr["soma"] = self.gb / (self.gl + self.gb + self.ga) * self.u_pyr["basal"]
        if u_next is not None:
            r_inn = self.act(self.u_inn["soma"])
            r_next = self.act(u_next)
            self.u_pyr["apical"][:] = np.matmul(self.W_down, r_next) + np.matmul(self.W_pi, r_inn)
            self.u_pyr["soma"] += self.ga / (self.gl + self.gb + self.ga) * self.u_pyr["apical"]

    def update_inn(self, r_pyr, u_next):
        ####compute interneuron potentials
        self.u_inn["dendrite"][:] = np.matmul(self.W_ip, r_pyr)
        self.u_inn["soma"] = self.gd / (self.gl + self.gd) * self.u_inn["dendrite"]
        if u_next is not None:
            l = self.gsom / (self.gl + self.gd + self.gsom)
            self.u_inn["soma"] = (1 - l) * self.gd / (self.gl + self.gd) * self.u_inn["dendrite"] + l * u_next


class OutputLayer(pyral.OutputLayer):

    def update(self, r_in, u_target):
        ### input rates
        r_in_buf = np.zeros(len(r_in) + self.bias, dtype=dtype)
        if self.bias:
            r_in_buf[:-1] = r_in
            r_in_buf[-1] = self.bias_val
        else:
            r_in_buf = r_in

        ### compute potentials
        self.u_pyr["basal"][:] = np.matmul(self.W_up, r_in_buf)  # [:] to enforce rhs to be copied to lhs
        self.u_pyr["soma"] = self.gb / (self.gl + self.gb) * self.u_pyr["basal"]
        if u_target is not None:
            l = self.gsom / (self.gl + self.gb + self.gsom)
            self.u_pyr["soma"] = (1 - l) * self.gb / (self.gl + self.gb) * self.u_pyr["basal"] + l * u_target

    def update_weights(self, r_in, n_exposures):
        # input rates
        r_in_buf = np.zeros(len(r_in) + self.bias, dtype=dtype)
        if self.bias:
            r_in_buf[:-1] = r_in
            r_in_buf[-1] = self.bias_val
        else:
            r_in_buf = r_in

        # discretised lowpass
        dT = (self.params["t_pattern"] - self.params["learning_lag"]) / n_exposures
        a = dT / (self.params["tau_w"] + dT)
        self.Delta_up = self.Delta_up * (1 - a) + a * np.outer(
            self.act(self.u_pyr["soma"]) - self.act(self.gb / (self.gl + self.gb) * self.u_pyr["basal"]),
            r_in_buf)

        self.W_up += dT * self.eta["up"] * self.Delta_up

    def apply(self):
        pass


class Net(pyral.Net):

    def __init__(self, params, act=None, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.params = params
        self.layer = []
        if act is None:
            self.act = pyral.soft_relu
        else:
            self.act = act
        dims = params["dims"]
        self.dims = dims
        bias = params["bias"]["on"]
        bias_val = params["bias"]["val"]
        eta = {}
        for n in range(1, len(dims) - 1):
            eta["up"] = params["eta"]["up"][n - 1]
            eta["pi"] = params["eta"]["pi"][n - 1]
            eta["ip"] = params["eta"]["ip"][n - 1]
            self.layer += [Layer(dims[n], dims[n - 1], dims[n + 1], eta, params, self.act, bias, bias_val)]
        eta["up"] = params["eta"]["up"][-1]
        eta["pi"] = params["eta"]["pi"][-1]
        eta["ip"] = params["eta"]["ip"][-1]
        self.layer += [OutputLayer(dims[-1], dims[-2], eta, params, self.act, bias, bias_val)]
        print("feedback-couplings: lambda_out = %f, lambda_inter = %f, lambda_hidden = %f"
              % (params["gsom"] / (params["gl"] + params["gb"] + params["gsom"]),
                 params["gsom"] / (params["gl"] + params["gd"] + params["gsom"]),
                 params["ga"] / (params["gl"] + params["gb"] + params["ga"])))


    def update(self, r_in, u_target, n_passes=2, n_exposures=10, learning_on=True, records=None):
        for k in range(n_exposures):
            #update potentials
            for i in range(n_passes):
                self.layer[0].update(r_in, None if i==0 else self.layer[1].u_pyr["soma"], dir_up=True)
                for n in range(1, len(self.layer) - 1):
                    self.layer[n].update(self.act(self.layer[n - 1].u_pyr["soma"]), None if i==0 else self.layer[n + 1].u_pyr["soma"], dir_up=True)
                self.layer[-1].update(self.act(self.layer[-2].u_pyr["soma"]), u_target)
                for n in range(len(self.layer) - 1, 1):
                    self.layer[n].update(self.act(self.layer[n - 1].u_pyr["soma"]), self.layer[n + 1].u_pyr["soma"], dir_up=False)
                self.layer[0].update(r_in, self.layer[1].u_pyr["soma"], dir_up=False)

            #update weights
            if learning_on:
                self.layer[0].update_weights(r_in, n_exposures=n_exposures)
                for n in range(1, len(self.layer)):
                    self.layer[n].update_weights(self.act(self.layer[n - 1].u_pyr["soma"]), n_exposures=n_exposures)

        for i, layer in enumerate(self.layer):
            if records is not None and not records == []:
                for _, r in records[i].items():
                    r.record()

    def run(self, in_seq, trgt_seq=None, reset_weights=False, val_len=0, n_passes=2, n_exposures=10, metric=None,
            rec_pots=None, rec_dn=1, learning_off=False, info_update=100):
        #### prepare run
        print("each input pattern is presented %d times -> dT/tau_w: %.3f"%(n_exposures, (self.params["t_pattern"] - self.params["learning_lag"]) /(self.params["tau_w"] * n_exposures)))
        print("effective learning rate multiplier: %f"%((self.params["t_pattern"] - self.params["learning_lag"]) / n_exposures))
        assert isinstance(rec_dn, int)
        rec_len = int(np.ceil(len(in_seq)/ rec_dn))
        records = []

        # sequence of outputs generated by the network
        out_seq = np.zeros((len(in_seq), self.params["dims"][-1]), dtype=dtype)

        #store validation results
        if val_len > 0:
            val_res = []  # [mse of val result, metric of val result]

        if trgt_seq is not None:
            if len(trgt_seq) != len(in_seq):
                raise Exception("input and target sequence mismatch")

        # reset/initialize and add trackers for potentials that should be recorded
        for i in range(len(self.layer)):
            l = self.layer[i]
            l.reset(reset_weights)
            if rec_pots is None:
                continue
            rp = rec_pots[i]
            rcs = {}
            if "pyr_soma" in rp:
                rcs["pyr_soma"] = pyral.Tracker(rec_len, l.u_pyr["soma"], rec_dn)
            if "pyr_basal" in rp:
                rcs["pyr_basal"] = pyral.Tracker(rec_len, l.u_pyr["basal"], rec_dn)
            if "pyr_apical" in rp:
                rcs["pyr_apical"] = pyral.Tracker(rec_len, l.u_pyr["apical"], rec_dn)
            if "inn_dendrite" in rp:
                rcs["inn_dendrite"] = pyral.Tracker(rec_len, l.u_inn["dendrite"], rec_dn)
            if "inn_soma" in rp:
                rcs["inn_soma"] = pyral.Tracker(rec_len, l.u_inn["soma"], rec_dn)
            if "W_up" in rp:
                rcs["W_up"] = pyral.Tracker(rec_len, l.W_up, rec_dn)
            if "W_down" in rp:
                rcs["W_down"] = pyral.Tracker(rec_len, l.W_down, rec_dn)
            if "W_ip" in rp:
                rcs["W_ip"] = pyral.Tracker(rec_len, l.W_ip, rec_dn)
            if "W_pi" in rp:
                rcs["W_pi"] = pyral.Tracker(rec_len, l.W_pi, rec_dn)
            records += [rcs]


        ####simulate

        start = time.time()
        val_idx = -1
        for i in range(len(in_seq)):
            nudging_on = trgt_seq[i, -1] if trgt_seq is not None else False
            if not nudging_on and val_len > 0:
                val_idx += 1
            if trgt_seq is not None:
                u_trgt = trgt_seq[i, :-1]
                self.update(in_seq[i], u_trgt if nudging_on else None, n_passes, n_exposures, learning_on=not learning_off, records=records)
                out_seq[i] = self.layer[-1].u_pyr["soma"]
            else:
                self.update(in_seq[i], None, n_passes, n_exposures, learning_on=not learning_off, records=records)
                out_seq[i] = self.layer[-1].u_pyr["soma"]

            # print validation results if finished
            if val_idx >= 0 and val_idx == val_len - 1:
                print("---Validating on %d patterns---" % (val_len))
                pred = out_seq[i - val_len + 1:i + 1]
                true = trgt_seq[i - val_len + 1:i + 1, :-1]
                mse = np.mean((pred - true) ** 2)
                print("mean squared error: %f" % (mse))
                vres = [mse, 0]
                if metric is not None:
                    name, mres = metric(pred, true)
                    print("%s: %f" % (name, mres))
                    vres[1] = mres
                val_res += [vres]
                val_idx = -1

            # print some info
            if i > 0 and i % info_update == 0:
                print("%d/%d input patterns done. About %s left." % (i, len(in_seq), pyral.time_str((len(in_seq) - i - 1) * (time.time() - start) / info_update)))
                start = time.time()

        # finalize recordings
        for rcs in records:
            for _, r in rcs.items(): r.finalize()

        ret = []
        if rec_pots is not None:
            ret += [records]
        ret += [out_seq]
        if val_len > 0:
            ret += [np.array(val_res, dtype=dtype)]
        return tuple(ret) if len(ret) > 1 else ret[0]


    def train(self, X_train, Y_train, X_val, Y_val, n_epochs, val_len, n_out, classify, u_high=1.0,
              u_low=0.1, n_passes=2, n_exposures=10, rec_pots=None, rec_dn=1, vals_per_epoch=1, reset_weights=False,
              info_update=1000, metric=None):

        assert len(X_train) > vals_per_epoch
        assert len(X_train) == len(Y_train)
        assert len(X_val) == len(Y_val)

        n_features = X_train.shape[1]

        assert n_features == X_val.shape[1]
        assert n_features == self.dims[0]
        assert n_out == self.dims[-1]

        if classify:
            assert len(Y_train.shape) == 1
            assert len(Y_val.shape) == 1
        else:
            assert (len(Y_train.shape) == 1 and n_out == 1) or Y_train.shape[1] == n_out
            assert (len(Y_val.shape) == 1 and n_out == 1) or Y_val.shape[1] == n_out

        len_split_train = round(len(X_train) / vals_per_epoch)  # validation after each split
        vals_per_epoch = len(X_train) // len_split_train
        print("%d validations per epoch" % (vals_per_epoch))

        len_per_ep = vals_per_epoch * val_len + len(X_train)
        length = len_per_ep * n_epochs

        r_in_seq = np.zeros((length, n_features))
        val_res = np.zeros((vals_per_epoch * n_epochs,
                            3),
                           dtype=dtype)  # [number of training patterns seen, mse of val result, metric of val result]

        if classify:
            target_seq = np.ones((length, n_out), dtype=dtype) * u_low
        else:
            target_seq = np.zeros((length, n_out), dtype=dtype)

        nudging_on = np.ones((length, 1), dtype=dtype)
        val_idc = np.zeros((len(val_res), val_len))  # indices of validation patterns

        for n in range(n_epochs):
            perm_train = np.random.permutation(len(X_train))
            left = n * len_per_ep
            left_tr = 0
            for k in range(vals_per_epoch):
                if k == vals_per_epoch - 1:
                    right_tr = len(X_train)
                else:
                    right_tr = left_tr + len_split_train
                right = left + right_tr - left_tr
                r_in_seq[left: right] = X_train[perm_train[left_tr:right_tr]]
                if classify:
                    target_seq[np.arange(left, right), 1 * Y_train[
                        perm_train[left_tr:right_tr]]] = u_high  # enforce Y_train is an integer array!
                else:
                    target_seq[left:right] = Y_train[perm_train[left_tr:right_tr]]
                perm_val = np.random.permutation(len(X_val))[:val_len]
                left = right
                right = left + val_len
                r_in_seq[left: right] = X_val[perm_val]
                if classify:
                    target_seq[
                        np.arange(left, right), 1 * Y_val[perm_val]] = u_high  # enforce Y_val is an integer array!
                else:
                    target_seq[left:right] = Y_val[perm_val]
                nudging_on[left: right, 0] = False
                val_res[vals_per_epoch * n + k, 0] = right_tr + n * len(X_train)
                val_idc[vals_per_epoch * n + k] = np.arange(left, right)
                left = right
                left_tr = right_tr

        target_seq = np.hstack((target_seq, nudging_on))

        ret = self.run(r_in_seq, trgt_seq=target_seq, n_passes=n_passes, n_exposures=n_exposures,
                       reset_weights=reset_weights, val_len=val_len, metric=metric,
                       rec_pots=rec_pots, rec_dn=rec_dn, info_update=info_update)
        val_res[:, 1:] = ret[-1] #valres

        return ret[:-1] + tuple([val_res])

    def train_long(self, X_train, Y_train, X_val, Y_val, n_epochs, val_len, n_out, classify, u_high=1.0,
              u_low=0.1, n_passes=2, n_exposures=1, vals_per_epoch=1, info_update=1000, metric=None):

        assert len(X_train) > vals_per_epoch
        assert len(X_train) == len(Y_train)
        assert len(X_val) == len(Y_val)

        n_features = X_train.shape[1]

        assert n_features == X_val.shape[1]
        assert n_features == self.dims[0]
        assert n_out == self.dims[-1]

        if classify:
            assert len(Y_train.shape) == 1
            assert len(Y_val.shape) == 1
        else:
            assert (len(Y_train.shape) == 1 and n_out == 1) or Y_train.shape[1] == n_out
            assert (len(Y_val.shape) == 1 and n_out == 1) or Y_val.shape[1] == n_out

        len_split_train = round(len(X_train) / vals_per_epoch)  # validation after each split
        vals_per_epoch = len(X_train) // len_split_train
        print("%d validations per epoch" % (vals_per_epoch))

        length = vals_per_epoch * val_len + len(X_train)

        r_in_seq = np.zeros((length, n_features))
        val_res = np.zeros((vals_per_epoch * n_epochs,
                            3),
                           dtype=dtype)  # [number of training patterns seen, mse of val result, metric of val result]

        nudging_on = np.ones((length, 1), dtype=dtype)
        val_idc = np.zeros((len(val_res), val_len))  # indices of validation patterns

        for n in range(n_epochs):
            if classify:
                target_seq = np.ones((length, n_out), dtype=dtype) * u_low
            else:
                target_seq = np.zeros((length, n_out), dtype=dtype)
            perm_train = np.random.permutation(len(X_train))
            left = 0
            left_tr = 0
            for k in range(vals_per_epoch):
                if k == vals_per_epoch - 1:
                    right_tr = len(X_train)
                else:
                    right_tr = left_tr + len_split_train
                right = left + right_tr - left_tr
                r_in_seq[left: right] = X_train[perm_train[left_tr:right_tr]]
                if classify:
                    target_seq[np.arange(left, right), 1 * Y_train[
                        perm_train[left_tr:right_tr]]] = u_high  # enforce Y_train is an integer array!
                else:
                    target_seq[left:right] = Y_train[perm_train[left_tr:right_tr]]
                perm_val = np.random.permutation(len(X_val))[:val_len]
                left = right
                right = left + val_len
                r_in_seq[left: right] = X_val[perm_val]
                if classify:
                    target_seq[
                        np.arange(left, right), 1 * Y_val[perm_val]] = u_high  # enforce Y_val is an integer array!
                else:
                    target_seq[left:right] = Y_val[perm_val]
                nudging_on[left: right, 0] = False
                val_res[vals_per_epoch * n + k, 0] = right_tr + n * len(X_train)
                val_idc[vals_per_epoch * n + k] = np.arange(left, right)
                left = right
                left_tr = right_tr

            target_seq = np.hstack((target_seq, nudging_on))

            print("%d/%d epochs:"%(n+1, n_epochs))
            ret = self.run(r_in_seq, trgt_seq=target_seq, n_passes=n_passes, n_exposures=n_exposures,
                           reset_weights=False, val_len=val_len, metric=metric, info_update=info_update)
            val_res[n*vals_per_epoch:(n+1)*vals_per_epoch, 1:] = ret[-1] #valres

        return ret[:-1] + tuple([val_res])


def seq_to_trace(seq, dT):
    T = np.zeros(len(seq) * 2)
    T[np.arange(len(seq)) * 2] = np.arange(len(seq))*dT
    T[np.arange(len(seq)) * 2 + 1] = np.arange(1, len(seq)+1)*dT
    trace = np.zeros((len(T), seq.shape[1]))
    trace[np.arange(len(seq)) * 2] = seq
    trace[np.arange(len(seq)) * 2 + 1] = seq
    return T, trace
'''
def test(n_runs, n_passes):
    params = {"dims": [4, 120, 2], "dt": 0.1, "gl": 0.1, "gb": 1.0, "ga": 0.8, "gd": 1.0, "gsom": 0.8,
              "eta": {"up": [0.002, 0.001], "pi": [0.001, 0], "ip": [0.002, 0]},
              "init_weights": {"up": 0.5, "down": 1, "pi": 1, "ip": 1}, "tau_w": 30, "noise": 0, "t_pattern": 100,
              "out_lag": 98, "tau_0": 3}

    for n_p in n_passes:
        dev = []
        for i in range(n_runs):
            net = pyral.Net(params)
            net.reflect()
            symNet = Net(params)
            symNet.layer[0].W_up = net.layer[0].W_up.copy()
            symNet.layer[0].W_ip = net.layer[0].W_ip.copy()
            symNet.layer[0].W_pi = net.layer[0].W_pi.copy()
            symNet.layer[0].W_down = net.layer[0].W_down.copy()
            symNet.layer[1].W_up = net.layer[1].W_up.copy()

            r_in_seq = np.random.sample((400, 4))
            target_seq = np.random.sample((400, 2))
            target_seq = np.hstack((target_seq, np.ones((len(target_seq), 1))))

            rec_pots = [["pyr_soma", "pyr_apical", "pyr_basal", "inn_soma", "W_up", "W_ip", "W_pi"],
                        ["pyr_soma", "W_up"]]
            records, T, r_in, out_seq, u_target = net.run(r_in_seq, rec_pots, 100, trgt_seq=target_seq, info_update=100)
            _, out_seq_sym = symNet.run(r_in_seq, None, trgt_seq=target_seq, n_passes=n_p, info_update=100, n_exposures=5)
            dev += [np.mean((out_seq-out_seq_sym)**2)]
            t, trace = seq_to_trace(out_seq_sym, 100)
            plt.figure(figsize=(12, 10))
            plt.plot(T, records[1]["pyr_soma"].data, label="net dt=%.3f"%(params["dt"]), ls="--")
            plt.plot(t, trace)
            plt.legend()
            plt.title("%d passes"%(n_p))
            if np.any(out_seq > 100):
                k = np.argmax(np.max(records[0]["pyr_soma"].data, axis=0))
                plt.plot(T, records[0]["pyr_soma"].data[:, k], label="max upyr 0")
                plt.plot(T, records[0]["pyr_basal"].data[:, k], label="max basal 0")
                plt.plot(T, records[0]["pyr_apical"].data[:, k], label="max apical 0")
                plt.savefig("plots/pyralnet vs pyralan - %d passes - %d" % (n_p, i))
                plt.legend()
                #plt.clf()
                plt.show()
            else:
                plt.savefig("plots/pyralnet vs pyralan - %d passes - %d" % (n_p, i))
                plt.clf()
        print("Average squared error for %d passes: %f, std: %f"%(n_p, np.mean(dev), np.std(dev)))


def mimic_test(N=1000, n_epochs=50, N_in=2, N_hidden=3, N_out=2):
    # test task: Learn to mimic simple forward network
    W_21 = np.random.sample((N_out, N_hidden))*2-1
    W_10 = np.random.sample((N_hidden, N_in)) * 2 - 1

    # training set
    X_train = np.random.sample((N, N_in))
    act = pyral.soft_relu
    act = soft_relu
    ga, gsom = 0.8, 0.8
    gb , gd = 1, 1
    gl = 0.1
    teacher = lambda r_in: gb/(gl+gb)*np.matmul(W_21, act(gb/(gl+gb+ga)*np.matmul(W_10, r_in.T))).T
    Y_train = teacher(X_train)

    params = {"dims": [N_in, N_hidden, N_out], "dt": 0.1, "gl": 0.1, "gb": 1.0, "ga": 0.8, "gd": 1.0, "gsom": 0.8,
               "eta": {"up": [0.02, 0.01], "pi": [0.02, 0], "ip": [0.02, 0]},
              "init_weights": {"up": 0.1, "down": 1, "pi": 1, "ip": 1}, "tau_w": 30, "noise": 0, "t_pattern": 100,
              "out_lag": 3*10, "tau_0": 3}
    net = Net(params, act=act)

    rec_pots = [["pyr_soma", "pyr_apical", "inn_soma", "W_up", "W_ip", "W_pi"],
                ["pyr_soma", "W_up"]]
    rec_dn = (N*n_epochs)//100
    n_trials = (N*n_epochs)
    records, out_seq, test_err = net.train(X_train, Y_train, n_epochs=n_epochs, n_features=N_in,
                                           n_out=N_out, classify=False, rec_pots=rec_pots, rec_dn=rec_dn,
                                           n_exposures=10, info_update=500, test_intv=50)

    # plot exponential moving average of test error
    plt.title("Test error during training")
    plt.semilogy(pyral.ewma(test_err, int(len(test_err) // 20)))
    plt.xlabel("test trial")
    plt.ylabel("mean squared test error")
    plt.savefig("plots/PyraLAN test/test error during training.png")
    plt.show()

    #does self-predicting state emerge
    x = np.arange(len(records[0]["W_ip"].data))*rec_dn
    plt.title("Lateral weights convergence")
    plt.semilogy(x, np.sum((records[0]["W_ip"].data - records[1]["W_up"].data)**2, axis=(1,2)), label="$||W^{(0)}_{ip}-W^{(1)}_{up}||^2$")
    plt.semilogy(x, np.sum((records[0]["W_pi"].data + net.layer[0].W_down) ** 2, axis=(1, 2)), label="$||W^{(0)}_{pi}+W^{(0)}_{down}||^2$")
    plt.xlabel("trial")
    plt.ylabel("squared error")
    plt.legend()
    plt.savefig("plots/PyraLAN test/lateral weights convergence.png")
    plt.show()

    plt.figure(figsize=(12,10))
    plt.title("Forward Weights Evolution")
    for i in range(N_in):
        for j in range(N_hidden):
            plt.plot(x, records[0]["W_up"].data[:, j, i], label="$W^{(0)}_{up; %d->%d}$"%(i,j), lw=1)
            plt.plot([int(n_trials * 0.98), n_trials], [W_10[j, i], W_10[j, i]], c="r", lw=1.2)
    for i in range(N_hidden):
        for j in range(N_out):
            plt.plot(x, records[1]["W_up"].data[:, j, i], label="$W^{(1)}_{up; %d->%d}$" % (i,j), ls="--", lw=1)
            plt.plot([int(n_trials*0.98), n_trials], [W_21[j, i], W_21[j,i]], c="r", lw=1.2, ls=":")
    plt.xlabel("trial")
    plt.ylabel("weight")
    plt.xlim([-1, n_trials])
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig("plots/PyraLAN test/forward weights evolution.png")
    plt.show()

    # test
    r_in_seq = np.random.sample((10, N_in))
    target_seq = np.hstack((teacher(r_in_seq), np.zeros((len(r_in_seq), 1))))
    rec_test, out_seq_test = net.run(r_in_seq, rec_pots, rec_dn=1, trgt_seq=target_seq, test_if_not_nudging=True)

    _, trace_out_seq = seq_to_trace(out_seq_test, 1)
    t, trace_trgt_seq = seq_to_trace(target_seq[:, :-1], 1)
    plt.title("Test run")
    plt.plot(t, trace_trgt_seq[:, 0], label="$u_{target}[0]$")
    plt.plot(t, trace_out_seq[:, 0], label="$u_{out}[0]$")
    plt.plot(t, trace_trgt_seq[:, 1], label="$u_{target}[1]$")
    plt.plot(t, trace_out_seq[:, 1], label="$u_{out}[1]$")
    plt.xlabel("pattern")
    plt.ylabel("soma potentials")
    plt.legend()
    plt.savefig("plots/PyraLAN test/test run output.png")
    plt.show()


def bars_task(square_size=3, N_train=3000, N_test=300, n_epochs=2, N_hidden=30):
    X_train, Y_train = Dataset.BarsDataset(square_size, samples_per_class=round(N_train / square_size))[:]
    X_test, Y_test = Dataset.BarsDataset(square_size, samples_per_class=round(N_test / square_size))[:]
    params = {"dims": [square_size**2, N_hidden, 3], "dt": 0.1, "gl": 0.1, "gb": 1.0, "ga": 0.8, "gd": 1.0, "gsom": 0.8,
              "eta": {"up": [0.02, 0.01], "pi": [0.02, 0], "ip": [0.02, 0]},
              "init_weights": {"up": 1, "down": 1, "pi": 1, "ip": 1}, "tau_w": 30, "noise": 0, "t_pattern": 100,
              "out_lag": 100, "tau_0": 3}
    params_sim = {"dims": [square_size ** 2, N_hidden, 3], "dt": 0.1, "gl": 0.1, "gb": 1.0, "ga": 0.8, "gd": 1.0,
              "gsom": 0.8,
              "eta": {"up": [0.02, 0.01], "pi": [0.02, 0], "ip": [0.02, 0]},
              "init_weights": {"up": 1, "down": 1, "pi": 1, "ip": 1}, "tau_w": 30, "noise": 0, "t_pattern": 110,
              "out_lag": 100, "tau_0": 3}
    simNet = pyral.Net(params_sim, seed=0)
    simNet.reflect()
    rec_pots = [["W_up"], ["W_up"]]
    recs_sim, T, _, out_seq_sim, _, test_err_sim = simNet.train(X_train, Y_train, n_epochs=n_epochs,
                                                         n_features=square_size ** 2,
                                                         n_out=3, classify=True, u_high=1.0, u_low=0.1, rec_dt=110, rec_pots=rec_pots)
    stsNet = Net(params, seed=0)
    stsNet.reflect()
    recs_sts, out_seq_sts, test_err_sts = stsNet.train(X_train, Y_train, u_high=1.0, u_low=0.1, n_epochs=n_epochs, n_features=square_size**2,
                                                            n_out=3, classify=True, n_exposures=1000, rec_pots=rec_pots)
    # plot exponential moving average of test error
    plt.title("Test error during training")
    plt.semilogy(pyral.ewma(test_err_sim, int(len(test_err_sim) // 20)), label="sim test error")
    plt.semilogy(pyral.ewma(test_err_sts, int(len(test_err_sts) // 20)), label="steady state test error")
    plt.xlabel("test trial")
    plt.ylabel("mean squared test error")
    plt.legend()
    plt.show()

    # plot differences between simulation
    plt.title("Devation between simulation and steady-state network")
    plt.plot(np.sum((out_seq_sim-out_seq_sts)**2, axis=1), label="$||out^{sim}-out^{sts}||^2$")
    plt.xlabel("trial")
    plt.ylabel("squared deviation")
    plt.legend()
    plt.show()

    plt.title("Devation between simulation and steady-state network")
    plt.plot(np.sum((recs_sim[0]["W_up"].data - recs_sts[0]["W_up"].data) ** 2, axis=(1,2)), label="$||W_{up}^{0,sim}-W_{up}^{0, sts}||^2$")
    plt.plot(np.sum((recs_sim[0]["W_up"].data) ** 2, axis=(1,2)), label="$W_{up}^{0,sim}2$")
    plt.plot(np.sum((recs_sts[0]["W_up"].data) ** 2, axis=(1,2)), label="$W_{up}^{0,ssts}2$")
    #plt.plot(np.sum((recs_sim[1]["W_up"].data - recs_sts[1]["W_up"].data) ** 2, axis=(1,2)), label="$||W_{up}^{1,sim}-W_{up}^{1, sts}||^2$")
    plt.xlabel("trial")
    plt.ylabel("squared deviation")
    plt.legend()
    plt.show()

    #test run
    _, out_seq_test = stsNet.run(X_test)
    y_pred = np.argmax(out_seq_test, axis=1)
    acc = np.sum(y_pred == Y_test) / len(Y_test)
    print("test set accuracy: %f"%(acc))
'''

def yinyang_task(N_train=6000, N_test=600, n_epochs=45, N_hidden=120):
    Path("plots/SteadNet/yinyang_task").mkdir(parents=True, exist_ok=True)

    X_train, Y_train = Dataset.YinYangDataset(size=N_train, flipped_coords=True, seed=None)[:]
    X_val, Y_val = Dataset.YinYangDataset(size=N_test, flipped_coords=True, seed=None)[:]
    X_test, Y_test = Dataset.YinYangDataset(size=N_test, flipped_coords=True, seed=None)[:]
    params = {"dims": [4, N_hidden, 3], "dt": 0.1, "gl": 0.1, "gb": 1.0, "ga": 0.28, "gd": 1.0,
              "gsom": 0.34,
              "eta": {"up": [6.1, 0.00012], "pi": [0, 0], "ip": [0.00024, 0]},
              "bias": {"on": True, "val": 0.5},
              "init_weights": {"up": 0.1, "down": 1, "pi": 1, "ip": 1}, "tau_w": 30, "noise": 0, "t_pattern": 100,
              "out_lag": 80, "tau_0": 3, "learning_lag": 20}
    net = Net(params, act=pyral.sigmoid)
    net.reflect()

    out_seq, val_res = net.train(X_train, Y_train, X_val, Y_val, n_epochs=n_epochs, val_len=40, vals_per_epoch=15,
                                                      n_out=3, classify=True, u_high=1.0, u_low=0.1, metric=pyral.accuracy, n_exposures=1)

    # plot exponential moving average of validation error
    plt.title("Validation error during training")
    plt.semilogy(val_res[:, 0], pyral.ewma(val_res[:, 1], round(len(val_res) / 10)), label="mse")
    plt.xlabel("trial")
    plt.ylabel("mean squared error")
    ax2 = plt.gca().twinx()
    ax2.plot(val_res[:, 0], pyral.ewma(val_res[:, 2], round(len(val_res) / 10)), c="g", label="accuracy")
    ax2.set_ylabel("accuracy")
    plt.savefig("plots/SteadNet/yinyang_task/validation and accuracy during training.png")
    plt.show()

    # test run
    out_seq_test = net.run(X_test, learning_off=True, n_exposures=1)
    y_pred = np.argmax(out_seq_test, axis=1)
    acc = np.sum(y_pred == Y_test) / len(Y_test)
    print("test set accuracy: %f"%(acc))
    Dataset.plot_yy(X_test, y_pred)
    plt.title("test acc = %f"%(acc))
    plt.savefig("plots/SteadNet/yinyang_task/test result.png")
    plt.show()


def mnist_task():
    Path("plots/SteadNet/mnist_task").mkdir(parents=True, exist_ok=True)
    X_train = mnist.train_images().reshape((-1, 28**2))
    Y_train = mnist.train_labels()
    rand_ind = np.random.permutation(len(X_train))
    X_val = X_train[rand_ind[:5000]]
    Y_val = Y_train[rand_ind[:5000]]
    X_train = X_train[rand_ind[5000:]]
    Y_train = Y_train[rand_ind[5000:]]
    X_test = mnist.test_images().reshape((-1, 28 ** 2))
    Y_test = mnist.test_labels()

    params = {"dims": [28**2, 500, 500, 10], "dt": 0.1, "gl": 0.1, "gb": 1.0, "ga": 0.471, "gd": 1.0,
              "gsom": 0.12,
              "eta": {"up": [0.001111, 0.0003333, 0.0001], "pi": [0, 0, 0], "ip": [0.002222, 0.0006666]},
              "bias": {"on": False, "val": 0.0},
              "init_weights": {"up": 0.05, "down": 1, "pi": 1, "ip": 0.05}, "tau_w": 30, "noise": 0, "t_pattern": 100,
              "out_lag": 80, "tau_0": 3, "learning_lag": 00}
    net = Net(params, act=pyral.sigmoid_stable)
    net.reflect()

    out_seq, val_res = net.train_long(X_train, Y_train, X_val, Y_val, n_epochs=200, val_len=5000, vals_per_epoch=1,
                                 n_out=10, classify=True, metric=pyral.accuracy, n_exposures=1)

    # plot exponential moving average of validation error
    plt.title("Validation error during training")
    plt.semilogy(val_res[:, 0], pyral.ewma(val_res[:, 1], round(len(val_res) / 10)), label="mse")
    plt.xlabel("trial")
    plt.ylabel("mean squared error")
    ax2 = plt.gca().twinx()
    ax2.plot(val_res[:, 0], pyral.ewma(val_res[:, 2], round(len(val_res) / 10)), c="g", label="accuracy")
    ax2.set_ylabel("accuracy")
    plt.savefig("plots/SteadNet/yinyang_task/validation and accuracy during training.png")
    plt.show()

    # test run
    out_seq_test = net.run(X_test, learning_off=True, n_exposures=1)
    y_pred = np.argmax(out_seq_test, axis=1)
    acc = np.sum(y_pred == Y_test) / len(Y_test)
    print("test set accuracy: %f" % (acc))


def run_yinyang(params, name, dir):
    X_train, Y_train = Dataset.YinYangDataset(bottom_left=0, top_right=1, size=params["N_train"], flipped_coords=True)[:]
    X_val, Y_val = Dataset.YinYangDataset(bottom_left=0, top_right=1, size=params["N_val"], flipped_coords=True)[:]
    X_test, Y_test = Dataset.YinYangDataset(bottom_left=0, top_right=1, size=params["N_test"], flipped_coords=True)[:]
    if params["model"]["act"] == "sigmoid":
        act = pyral.sigmoid
    elif params["model"]["act"] == "softReLU":
        act = pyral.soft_relu
    net = Net(params["model"], act=act)
    net.reflect()

    out_seq, val_res = net.train(X_train, Y_train, X_val, Y_val, n_epochs=params["N_epochs"], val_len=params["val_len"], vals_per_epoch=params["vals_per_epoch"],
                                 n_out=3, classify=True, u_high=1.0, u_low=0.1, metric=pyral.accuracy, n_exposures=params["N_exposures"])

    f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 2]}, figsize=(11,5))

    # plot exponential moving average of validation error
    a0.set_title("Validation error during training %s" % (name))
    a0.semilogy(val_res[:, 0], pyral.ewma(val_res[:, 1], round(len(val_res) / 10)), label="mse")
    a0.set_xlabel("trial")
    a0.set_ylabel("mean squared error")
    ax2 = a0.twinx()
    ax2.plot(val_res[:, 0], pyral.ewma(val_res[:, 2], round(len(val_res) / 10)), c="g", label="accuracy")
    ax2.set_ylabel("accuracy")

    # test run
    out_seq_test = net.run(X_test, learning_off=True, n_exposures=1)
    y_pred = np.argmax(out_seq_test, axis=1)
    acc = np.sum(y_pred == Y_test) / len(Y_test)
    print("test set accuracy : %f " % (acc))
    Dataset.plot_yy(X_test, y_pred, ax=a1)
    a1.set_title("test accuracy = %f, %s" % (acc, name))

    plt.tight_layout()
    plt.savefig(dir+"result_%s.png" % (name))

    with open(dir+"results.txt", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write("%s\t\t%f\t\t%f\n"%(name, val_res[-1, 1], acc))
        fcntl.flock(f, fcntl.LOCK_UN)


mnist_task()
exit(0)

import argparse, json

parser = argparse.ArgumentParser()
parser.add_argument('task', help='task')
parser.add_argument('--config', help='config file')
parser.add_argument('--dir', help='store results here')

args = parser.parse_args()

with open(args.config) as json_file:
    params = json.load(json_file)

if args.task == "yinyang":
    run_yinyang(params, params["name"], args.dir)
else:
    raise Exception("task not known!")
