import matplotlib.pyplot as plt
import numpy as np
import time
import Dataset
from pathlib import Path
import fcntl


dtype = np.float32

class Layer:
    def __init__(self, N_pyr, N_in, N_next, eta, params, act, bias_on_pyr, bias_on_inter, bias_val):
        self.u_pyr = {"basal": np.zeros(N_pyr, dtype=dtype), "apical": np.zeros(N_pyr, dtype=dtype), "soma": np.zeros(N_pyr, dtype=dtype)}
        self.u_inn = {"dendrite": np.zeros(N_next, dtype=dtype), "soma": np.zeros(N_next, dtype=dtype)}

        self.W_up = (np.random.sample((N_pyr, N_in+bias_on_pyr)).astype(dtype) - 0.5) * 2 * params["init_weights"]["up"]
        self.W_down = (np.random.sample((N_pyr, N_next)).astype(dtype) - 0.5) * 2 * params["init_weights"]["down"]
        self.W_pi = (np.random.sample((N_pyr, N_next)).astype(dtype) - 0.5) * 2 * params["init_weights"]["pi"]
        self.W_ip = (np.random.sample((N_next, N_pyr+bias_on_inter)).astype(dtype) - 0.5) * 2 * params["init_weights"]["ip"]

        self.bias_on_pyr = bias_on_pyr
        self.bias_on_inter = bias_on_inter
        self.bias_val = bias_val

        self.Delta_up = np.zeros((N_pyr, N_in+bias_on_pyr), dtype=dtype)
        self.Delta_pi = np.zeros((N_pyr, N_next), dtype=dtype)
        self.Delta_ip = np.zeros((N_next, N_pyr+bias_on_inter), dtype=dtype)

        self.set_params(params, eta)

        self.act = act

    def set_params(self, params, eta):
        self.gl = params["gl"]
        self.gb = params["gb"]
        self.ga = params["ga"]
        self.gsom = params["gsom"]
        self.gd = params["gd"]

        self.eta = eta.copy()
        self.tau_w = params["tau_w"]

        self.noise = params["noise"]

        self.dt = params["dt"]

        self.params = params

    def update(self, r_in, u_next, learning_on, noise_on=True):

        #### rates
        r_pyr = np.zeros(self.u_pyr["soma"].shape[0] + self.bias_on_inter, dtype=dtype)
        r_in_buf = np.zeros(r_in.shape[0] + self.bias_on_pyr, dtype=dtype)
        r_inn = self.act(self.u_inn["soma"])
        r_next = self.act(u_next)
        r_in_buf[: len(r_in)] = r_in
        r_pyr[: len(self.u_pyr["soma"])] = self.act(self.u_pyr["soma"])
        if self.bias_on_pyr:
            r_in_buf[-1] = self.bias_val
        if self.bias_on_inter:
            r_pyr[-1] = self.bias_val

        ####compute dendritic potentials at current time

        # pyramidial neurons
        self.u_pyr["basal"][:] = np.matmul(self.W_up, r_in_buf)  # [:] to enforce rhs to be copied to lhs
        self.u_pyr["apical"][:] = np.matmul(self.W_down, r_next) + np.matmul(self.W_pi, r_inn)

        # lateral interneurons
        self.u_inn["dendrite"][:] = np.matmul(self.W_ip, r_pyr)

        ####compute changes

        u_p = self.u_pyr["soma"]
        u_i = self.u_inn["soma"]

        self.du_pyr = self.dt * (-self.gl * u_p + self.gb * (self.u_pyr["basal"] - u_p) + self.ga * (
                self.u_pyr["apical"] - u_p) + noise_on * self.noise * np.random.normal(size=1))
        self.du_inn = self.dt * (-self.gl * u_i + self.gd * (self.u_inn["dendrite"] - u_i) + self.gsom * (
                u_next - u_i) + noise_on * self.noise * np.random.normal(size=1))

        if not learning_on:
            return
        # weight updates (lowpass weight changes)
        gtot = self.gl + self.gb + self.ga
        dDelta_up = self.dt / self.tau_w * (- self.Delta_up + np.outer(
            self.act(self.u_pyr["soma"]) - self.act(self.gb / gtot * self.u_pyr["basal"]), r_in_buf))
        dDelta_ip = self.dt / self.tau_w * (- self.Delta_ip + np.outer(
            self.act(self.u_inn["soma"]) - self.act(self.gd / (self.gl + self.gd) * self.u_inn["dendrite"]), r_pyr))
        dDelta_pi = self.dt / self.tau_w * (- self.Delta_pi + np.outer(-self.u_pyr["apical"], r_inn))

        self.W_up += self.dt * self.eta["up"] * self.Delta_up
        self.W_ip += self.dt * self.eta["ip"] * self.Delta_ip
        self.W_pi += self.dt * self.eta["pi"] * self.Delta_pi

        # apply Deltas
        self.Delta_up += dDelta_up
        self.Delta_ip += dDelta_ip
        self.Delta_pi += dDelta_pi

    def apply(self):
        # apply changes to soma potential
        self.u_pyr["soma"] += self.du_pyr
        self.u_inn["soma"] += self.du_inn

    def reset(self, reset_weights=True):
        self.u_pyr["basal"].fill(0)
        self.u_pyr["soma"].fill(0)
        self.u_pyr["apical"].fill(0)
        self.u_inn["dendrite"].fill(0)
        self.u_inn["soma"].fill(0)

        if reset_weights:
            self.W_up = (np.rand_like(self.W_up) - 0.5) * 2 * self.params["init_weights"]["up"]
            self.W_down = (np.rand_like(self.W_down) - 0.5) * 2 * self.params["init_weights"]["down"]
            self.W_pi = (np.rand_like(self.W_pi) - 0.5) * 2 * self.params["init_weights"]["pi"]
            self.W_ip = (np.rand_like(self.W_ip) - 0.5) * 2 * self.params["init_weights"]["ip"]

        self.Delta_up.fill(0)
        self.Delta_pi.fill(0)
        self.Delta_ip.fill(0)


class OutputLayer:

    def __init__(self, N_out, N_in, eta, params, act, bias, bias_val):
        self.u_pyr = {"basal": np.zeros(N_out, dtype=dtype), "soma": np.zeros(N_out, dtype=dtype)}

        self.W_up = (np.random.sample((N_out, N_in + bias)).astype(dtype) - 0.5) * 2 * params["init_weights"]["up"]

        self.Delta_up = np.zeros((N_out, N_in + bias), dtype=dtype)

        self.set_params(params, eta)

        self.act = act

        self.bias = bias
        self.bias_val = bias_val

    def set_params(self, params, eta):
        self.gl = params["gl"]
        self.gb = params["gb"]
        self.gsom = params["gsom"]
        self.ga = 0

        self.eta = eta.copy()
        self.tau_w = params["tau_w"]

        self.noise = params["noise"]

        self.dt = params["dt"]

        self.params = params

    def update(self, r_in, u_target, learning_on, noise_on=True):

        #### input rates
        r_in_buf = np.zeros(r_in.shape[0] + self.bias, dtype=dtype)
        if self.bias:
            r_in_buf[:-1] = r_in
            r_in_buf[-1] = self.bias_val
        else:
            r_in_buf = r_in

        #### compute dendritic potentials at current time

        self.u_pyr["basal"][:] = np.matmul(self.W_up, r_in_buf)

        #### compute changes

        self.du_pyr = self.dt * (-self.gl * self.u_pyr["soma"] + self.gb * (
                self.u_pyr["basal"] - self.u_pyr["soma"]) + noise_on * self.noise * np.random.normal(size=1))
        if u_target is not None:
            self.du_pyr += self.dt * self.gsom * (u_target - self.u_pyr["soma"])

        if not learning_on:
            return
        # weight updates (lowpass weight changes)
        gtot = self.gl + self.gb
        dDelta_up = self.dt / self.tau_w * (- self.Delta_up + np.outer(
            self.act(self.u_pyr["soma"]) - self.act(self.gb / gtot * self.u_pyr["basal"]), r_in_buf))

        self.W_up += self.dt * self.eta["up"] * self.Delta_up

        # apply Delta
        self.Delta_up += dDelta_up

    def apply(self):
        # apply changes to soma potential
        self.u_pyr["soma"] += self.du_pyr

    def reset(self, reset_weights=True):
        self.u_pyr["basal"].fill(0)
        self.u_pyr["soma"].fill(0)

        if reset_weights:
            self.W_up = (np.rand_like(self.W_up) - 0.5) * 2 * self.params["init_weights"]["up"]

        self.Delta_up.fill(0)


class Net:

    def __init__(self, params, act=None, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.params = params
        self.layer = []
        if act is None:
            self.act = soft_relu
        else:
            self.act = act
        dims = params["dims"]
        self.dims = dims
        bias_on_pyr = params["bias"]["pyr_on"]
        bias_on_inter = params["bias"]["inter_on"]
        bias_val = params["bias"]["val"]
        eta = {}
        for n in range(1, len(dims) - 1):
            eta["up"] = params["eta"]["up"][n - 1]
            eta["pi"] = params["eta"]["pi"][n - 1]
            eta["ip"] = params["eta"]["ip"][n - 1]
            self.layer += [Layer(dims[n], dims[n - 1], dims[n + 1], eta, params, self.act, bias_on_pyr, bias_on_inter, bias_val)]
        eta["up"] = params["eta"]["up"][-1]
        eta["pi"] = params["eta"]["pi"][-1]
        eta["ip"] = params["eta"]["ip"][-1]
        self.layer += [OutputLayer(dims[-1], dims[-2], eta, params, self.act, bias_on_pyr, bias_val)]
        print("feedback-couplings: lambda_out = %f, lambda_inter = %f, lambda_hidden = %f"
              % (params["gsom"] / (params["gl"] + params["gb"] + params["gsom"]),
                 params["gsom"] / (params["gl"] + params["gd"] + params["gsom"]),
                 params["ga"] / (params["gl"] + params["gb"] + params["ga"])))

    def reflect(self):
        for i in range(len(self.layer) - 1):
            l = self.layer[i]
            l_n = self.layer[i + 1]
            l.W_pi = - l.W_down.copy()
            cols = min(l.W_ip.shape[1], l_n.W_up.shape[1]) # due to biases the column length of W_ip and W_up (next level) might differ
            l.W_ip[:, :cols] = l_n.W_up[:, :cols].copy() * l_n.gb / (l_n.gl + l_n.ga + l_n.gb) * (l.gl + l.gd) / l.gd

    def dump_weights(self, file):
        weights = []
        for n in range(1, len(self.layer) - 1):
            l = self.layer[n]
            weights += [[l.W_up, l.W_pi, l.W_ip, l.W_down]]
        weights += [[self.layer[-1].W_up]]
        np.save(file, weights)

    def update_params(self, params):
        for key, item in params.items():
            if key in self.params:
                self.params[key] = item
                print("update %s: %s." % (key, str(item)))
        eta = {}
        for n, l in enumerate(self.layer):
            eta["up"] = self.params["eta"]["up"][n]
            eta["pi"] = self.params["eta"]["pi"][n]
            eta["ip"] = self.params["eta"]["ip"][n]
            l.set_params(self.params, eta)

    def update(self, r_in, u_target, learning_on=True, records=None, noise_on=True):
        self.layer[0].update(r_in, self.layer[1].u_pyr["soma"], learning_on, noise_on=noise_on)
        for n in range(1, len(self.layer) - 1):
            self.layer[n].update(self.act(self.layer[n - 1].u_pyr["soma"]), self.layer[n + 1].u_pyr["soma"],
                                 learning_on, noise_on=noise_on)
        self.layer[-1].update(self.act(self.layer[-2].u_pyr["soma"]), u_target, learning_on, noise_on=noise_on)

        for i, layer in enumerate(self.layer):
            layer.apply()
            if records is not None and not records == []:
                for _, r in records[i].items():
                    r.record()

    def run(self, in_seq, trgt_seq=None, reset_weights=False, val_len=0, metric=None, rec_pots=None, rec_dt=0.0, learning_off=False,
            info_update=100):
        #### prepare run
        # record signals with time resolution rec_dt -> compress actual data
        n_pattern = int(self.params["t_pattern"] / self.params["dt"])  # length of one input pattern
        compress_len = int(np.round(rec_dt / self.params["dt"]))  # number of samples to average over
        if rec_dt > 0:
            rec_len = int(np.ceil(len(in_seq) * self.params["t_pattern"] / rec_dt))  # number of averaged samples to record
        records = []

        n_out_wait = round(
            self.params["out_lag"] / self.params["dt"])  # steps to wait per pattern before filling output buffer
        n_learning_wait = round(
            self.params["learning_lag"] / self.params["dt"])  # steps to wait per pattern before enabling learning
        if n_out_wait >= n_pattern:
            raise Exception("output lag to big!")
        if n_learning_wait >= n_pattern:
            raise Exception("learning lag to big!")

        r_in = in_seq[0].copy()  # current input rates
        if trgt_seq is not None:
            if len(trgt_seq) != len(in_seq):
                raise Exception("input and target sequence mismatch")
            u_trgt = trgt_seq[0, :-1].copy()  # current target potentials

        out_seq = np.zeros((len(in_seq), self.params["dims"][-1]), dtype=dtype)  # sequence of outputs generated by the network

        #store validation results
        if val_len > 0:
            val_res = []  # [mse of val result, metric of val result]

        # reset/initialize and add trackers for potentials that should be recorded
        for i in range(len(self.layer)):
            l = self.layer[i]
            l.reset(reset_weights)
            if rec_pots is None:
                continue
            rp = rec_pots[i]
            rcs = {}
            if "pyr_soma" in rp:
                rcs["pyr_soma"] = Tracker(rec_len, l.u_pyr["soma"], compress_len)
            if "pyr_basal" in rp:
                rcs["pyr_basal"] = Tracker(rec_len, l.u_pyr["basal"], compress_len)
            if "pyr_apical" in rp:
                rcs["pyr_apical"] = Tracker(rec_len, l.u_pyr["apical"], compress_len)
            if "inn_dendrite" in rp:
                rcs["inn_dendrite"] = Tracker(rec_len, l.u_inn["dendrite"], compress_len)
            if "inn_soma" in rp:
                rcs["inn_soma"] = Tracker(rec_len, l.u_inn["soma"], compress_len)
            if "W_up" in rp:
                rcs["W_up"] = Tracker(rec_len, l.W_up, compress_len)
            if "W_down" in rp:
                rcs["W_down"] = Tracker(rec_len, l.W_down, compress_len)
            if "W_ip" in rp:
                rcs["W_ip"] = Tracker(rec_len, l.W_ip, compress_len)
            if "W_pi" in rp:
                rcs["W_pi"] = Tracker(rec_len, l.W_pi, compress_len)
            records += [rcs]

        # init trackers for input rates signal and target potentials signal
        if rec_dt>0:
            r_in_trc = Tracker(rec_len, r_in, compress_len)
            u_trgt_trc = None
            if trgt_seq is not None:
                u_trgt_trc = Tracker(rec_len, u_trgt, compress_len)

        ####simulate

        start = time.time()
        val_idx = -1
        for seq_idx in range(len(in_seq)):
            nudging_on = trgt_seq[seq_idx, -1] if trgt_seq is not None else False
            if not nudging_on and val_len > 0:
                val_idx += 1
            for i in range(n_pattern):
                # lowpass input rates
                r_in[:] += self.params["dt"] / self.params["tau_0"] * (in_seq[seq_idx] - r_in)
                if rec_dt>0:
                    r_in_trc.record()
                learning_on = i > n_learning_wait and not learning_off

                # lowpass target potentials and update network
                if trgt_seq is not None:
                    u_trgt[:] += self.params["dt"] / self.params["tau_0"] * (trgt_seq[seq_idx, :-1] - u_trgt)
                    if rec_dt > 0:
                        u_trgt_trc.record()
                    l_on = learning_on and val_idx < 0
                    self.update(r_in, u_trgt if nudging_on else None, records=records, learning_on=l_on,
                                noise_on=val_idx < 0)
                    if i >= n_out_wait:
                        out_seq[seq_idx] += self.layer[-1].u_pyr["soma"] / (n_pattern - n_out_wait)
                else:
                    self.update(r_in, None, records=records, learning_on=learning_on)
                    if i >= n_out_wait:
                        out_seq[seq_idx] += self.layer[-1].u_pyr["soma"] / (n_pattern - n_out_wait)

            # print validation results if finished
            if val_idx >= 0 and val_idx == val_len - 1:
                print("---Validating on %d patterns---" % (val_len))
                pred = out_seq[seq_idx - val_len + 1:seq_idx + 1]
                true = trgt_seq[seq_idx - val_len + 1:seq_idx + 1, :-1]
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
            if seq_idx > 0 and seq_idx % info_update == 0:
                print("%d/%d input patterns done. About %s left." % (
                    seq_idx, len(in_seq), time_str((len(in_seq) - seq_idx - 1) * (time.time() - start) / info_update)))
                start = time.time()

        # finalize recordings
        for rcs in records:
            for _, r in rcs.items(): r.finalize()
        if rec_dt > 0:
            r_in_trc.finalize()
            if trgt_seq is not None:
                u_trgt_trc.finalize()

        # return records (with res rec_dt), a time signal (rec_dt), the input rates signal (rec_dt), target pot signal (rec_dt) and the output sequence
        ret = []
        if rec_pots is not None:
            ret += [records]
        if rec_dt > 0:
            ret += [np.linspace(0, rec_len * rec_dt, rec_len), r_in_trc.data]
            if trgt_seq is not None:
                ret += [u_trgt_trc.data]
        ret += [out_seq]
        if val_len > 0:
            ret += [np.array(val_res, dtype=dtype)]
        return tuple(ret) if len(ret)>1 else ret[0]

    def train(self, X_train, Y_train, X_val, Y_val, n_epochs, val_len, n_out, classify, u_high=1.0,
              u_low=0.1, rec_pots=None, rec_dt=0.0,
              vals_per_epoch=1, reset_weights=False, info_update=100, metric=None):

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
            assert (len(Y_train.shape) == 1 and n_out==1) or Y_train.shape[1] == n_out
            assert (len(Y_val.shape) == 1 and n_out==1) or Y_val.shape[1] == n_out

        len_split_train = round(len(X_train) / vals_per_epoch)  # validation after each split
        vals_per_epoch = len(X_train) // len_split_train
        print("%d validations per epoch" % (vals_per_epoch))

        len_per_ep = vals_per_epoch * val_len + len(X_train)
        length = len_per_ep * n_epochs

        r_in_seq = np.zeros((length, n_features))
        val_res = np.zeros((vals_per_epoch * n_epochs,
                            3), dtype=dtype)  # [number of training patterns seen, mse of val result, metric of val result]

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
                    target_seq[np.arange(left,right), 1*Y_train[perm_train[left_tr:right_tr]]] = u_high # enforce Y_train is an integer array!
                else:
                    target_seq[left:right] = Y_train[perm_train[left_tr:right_tr]]
                perm_val = np.random.permutation(len(X_val))[:val_len]
                left = right
                right = left + val_len
                r_in_seq[left: right] = X_val[perm_val]
                if classify:
                    target_seq[np.arange(left,right), 1*Y_val[perm_val]] = u_high # enforce Y_val is an integer array!
                else:
                    target_seq[left:right] = Y_val[perm_val]
                nudging_on[left: right, 0] = False
                val_res[vals_per_epoch * n + k, 0] = right_tr + n * len(X_train)
                val_idc[vals_per_epoch * n + k] = np.arange(left, right)
                left = right
                left_tr = right_tr

        target_seq = np.hstack((target_seq, nudging_on))

        ret = self.run(r_in_seq, trgt_seq=target_seq, reset_weights=reset_weights, val_len=val_len, metric=metric,
                       rec_pots=rec_pots, rec_dt=rec_dt, info_update=info_update)
        val_res[:, 1:] = ret[-1] #valres

        return ret[:-1] + tuple([val_res])


class Tracker:
    '''tracks and records changes in target array. Records length*compress_len samples, compressed (averaged) into length samples'''

    def __init__(self, length, target, compress_len):
        self.target = target
        self.data = np.zeros(tuple([length]) + target.shape, dtype=np.float32)
        self.index = 0
        self.buffer = np.zeros(target.shape)
        self.din = compress_len

    def record(self):
        self.buffer += self.target
        if (self.index + 1) % self.din == 0:
            self.data[int(self.index / self.din), :] = self.buffer / self.din
            self.buffer.fill(0)
        self.index += 1

    def finalize(self):
        '''fill last data point with average of remaining target data in buffer.'''
        n_buffer = self.index % self.din
        if n_buffer > 0:
            self.data[int(self.index / self.din), :] = self.buffer / n_buffer


def soft_relu(x, thresh=15):
    res = x.copy()
    ind = np.abs(x) < thresh
    res[x < -thresh] = 0
    res[ind] = np.log(1 + np.exp(x[ind]))
    return res

def sigmoid(x):
    return 1/(1+np.exp(-x))

def time_str(sec):
    string = ""
    h = int(sec / 3600)
    if h > 0:
        string = str(h) + "h, "
    if int(sec / 60) > 0:
        m = int((sec - h * 3600) / 60)
        string += str(m) + "min and "
    string += str(int(sec % 60)) + "s"
    return string


def ewma(data, window):
    alpha = 2 / (window + 1.0)
    alpha_rev = 1 - alpha
    n = data.shape[0]
    pows = alpha_rev ** (np.arange(n + 1))
    scale_arr = 1 / pows[:-1]
    offset = data[0] * pows[1:]
    pw0 = alpha * alpha_rev ** (n - 1)
    mult = data * pw0 * scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums * scale_arr[::-1]
    return out

def accuracy(pred, true):
    pred_class = np.argmax(pred, axis=1)
    true_class = np.argmax(true, axis=1)
    acc = np.sum(pred_class==true_class)/len(pred)
    return "accuracy", acc

def mimic_task(N=1000, n_epochs=10, N_in=2, N_hidden=3, N_out=2):
    Path("plots/PyraLNet/mimic_task").mkdir(parents=True, exist_ok=True)

    # test task: Learn to mimic simple forward network
    W_21 = np.random.sample((N_out, N_hidden)) * 2 - 1
    W_10 = np.random.sample((N_hidden, N_in)) * 2 - 1

    # training set
    X_train = np.random.sample((N, N_in))
    X_val = np.random.sample((200, N_in))
    act = soft_relu
    ga, gsom = 0.8, 0.8
    gb , gd = 1, 1
    gl = 0.1
    teacher = lambda r_in: gb/(gl+gb)*np.matmul(W_21, act(gb/(gl+gb+ga)*np.matmul(W_10, r_in.T))).T
    Y_train = teacher(X_train)
    Y_val = teacher(X_val)

    params = {"dims": [N_in, N_hidden, N_out], "dt": 0.1, "gl": gl, "gb": gb, "ga": ga, "gd": gd, "gsom": gsom,
              "eta": {"up": [0.01, 0.005], "pi": [0.01, 0], "ip": [0.01, 0]},
              "bias": {"pyr_on": False, "inter_on": False, "val": 0.0},
              "init_weights": {"up": 1, "down": 1, "pi": 1, "ip": 1}, "tau_w": 30, "noise": 0, "t_pattern": 100,
              "out_lag": 3 * 10, "tau_0": 3, "learning_lag": 0}
    net = Net(params, act)

    rec_pots = [["pyr_soma", "pyr_apical", "inn_soma", "W_up", "W_ip", "W_pi"],
                ["pyr_soma", "W_up"]]
    records, T, r_in, u_trgt, out_seq, val_res = net.train(X_train, Y_train, X_val, Y_val, n_epochs=n_epochs,
                                                           val_len=8,
                                                           n_out=N_out, classify=False, vals_per_epoch=10,
                                                           rec_pots=rec_pots, rec_dt=0.5*N * n_epochs)

    # plot exponential moving average of validation error
    plt.title("Validation error during training")
    plt.semilogy(val_res[:, 0], ewma(val_res[:, 1], round(len(val_res) / 10)))
    plt.xlabel("trial")
    plt.ylabel("mean squared error")
    plt.savefig("plots/PyraLNet/mimic_task/validation error during training.png")
    plt.show()

    # does self-predicting state emerge
    plt.title("Lateral Weights Convergence")
    plt.semilogy(T / 100, np.sum((records[0]["W_ip"].data - records[1]["W_up"].data) ** 2, axis=(1, 2)),
                 label="$||W^{(0)}_{ip}-W^{(1)}_{up}||^2$")
    plt.semilogy(T / 100, np.sum((records[0]["W_pi"].data + net.layer[0].W_down) ** 2, axis=(1, 2)),
                 label="$||W^{(0)}_{pi}+W^{(0)}_{down}||^2$")
    plt.xlabel("trial")
    plt.ylabel("squared error")
    plt.legend()
    plt.savefig("plots/PyraLNet/mimic_task/lateral weights convergence.png")
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.title("Forward Weights Evolution")
    T_max = T[-1] / 100
    for i in range(N_in):
        for j in range(N_hidden):
            plt.plot(T / 100, records[0]["W_up"].data[:, j, i], label="$W^{(0)}_{up; %d->%d}$" % (i, j), lw=1)
            plt.plot([T_max * 0.98, T_max], [W_10[j, i], W_10[j, i]], c="r", lw=1.2)
    for i in range(N_hidden):
        for j in range(N_out):
            plt.plot(T / 100, records[1]["W_up"].data[:, j, i], label="$W^{(1)}_{up; %d->%d}$" % (i, j), ls="--", lw=1)
            plt.plot([T_max * 0.98, T_max], [W_21[j, i], W_21[j, i]], c="r", lw=1.2, ls=":")
    plt.xlabel("trial")
    plt.ylabel("weight")
    plt.xlim([-1, T_max * 1.05])
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig("plots/PyraLNet/mimic_task/forward weights evolution.png")
    plt.show()

    # test
    print("-----Test run-----")
    r_in_seq = np.random.sample((10, N_in))
    target_seq = np.hstack((teacher(r_in_seq), np.zeros((len(r_in_seq), 1))))
    rec_test, T, r_in_test, u_target_test, out_seq_test, val_res_test = net.run(r_in_seq, trgt_seq=target_seq,
                                                                  val_len=1, rec_pots=rec_pots, rec_dt=1)

    plt.title("Test run")
    plt.plot(T, u_target_test[:, 0], label="$u_{target}[0]$")
    plt.plot(T, rec_test[-1]["pyr_soma"].data[:, 0], label="$u_{out}[0]$")
    plt.plot(T, u_target_test[:, 1], label="$u_{target}[1]$")
    plt.plot(T, rec_test[-1]["pyr_soma"].data[:, 1], label="$u_{out}[1]$")
    plt.xlabel("time / ms")
    plt.ylabel("soma potentials")
    plt.legend()
    plt.savefig("plots/PyraLNet/mimic_task/test run output.png")
    plt.show()



def sine_task(N_train=1000, N_test=500, n_epochs=10, N_hidden=40, bias_pyr=False, bias_inter=False):
    Path('plots/PyraLNet/sine_task').mkdir(parents=True, exist_ok=True)

    X_train, Y_train = Dataset.SineDataset(size=N_train, flipped_coords=True, seed=42)[:]
    X_val, Y_val = Dataset.SineDataset(size=N_test, flipped_coords=True, seed=None)[:]
    X_test, Y_test = Dataset.SineDataset(bottom_left=0, top_right=1, size=N_test, flipped_coords=True, seed=None)[:]
    params = {"dims": [4, N_hidden, 2], "dt": 0.1, "gl": 0.1, "gb": 1.0, "ga": 0.8, "gd": 1.0,
              "gsom": 0.8,
              "eta": {"up": [0.2, 0.002], "pi": [0, 0], "ip": [2*0.002, 0]},
              "bias": {"pyr_on": bias_pyr, "inter_on": bias_inter, "val": 0.5},
              "init_weights": {"up": 0.1, "down": 1, "pi": 1, "ip": 0.1}, "tau_w": 30, "noise": 0, "t_pattern": 100,
              "out_lag": 80, "tau_0": 3, "learning_lag": 0}
    net = Net(params, act=sigmoid)
    net.reflect()

    out_seq, val_res = net.train(X_train, Y_train, X_val, Y_val, n_epochs=n_epochs, val_len=40, vals_per_epoch=5,
                                 n_out=2, classify=True, u_high=1.0, u_low=0.1, metric=accuracy)

    # plot exponential moving average of validation error
    plt.title("Validation error during training")
    plt.semilogy(val_res[:, 0], ewma(val_res[:, 1], round(len(val_res) / 10)), label="mse")
    plt.xlabel("trial")
    plt.ylabel("mean squared error")
    ax2 = plt.gca().twinx()
    ax2.plot(val_res[:, 0], ewma(val_res[:, 2], round(len(val_res) / 10)), c="g", label="accuracy")
    ax2.set_ylabel("accuracy")
    plt.savefig("plots/PyraLNet/sine_task/validation and accuracy during training (pyr, inter)=(%d,%d).png"%(bias_pyr, bias_inter))
    plt.clf()

    # test run
    out_seq_test = net.run(X_test)
    y_pred = np.argmax(out_seq_test, axis=1)
    acc = np.sum(y_pred == Y_test) / len(Y_test)
    print("test set accuracy: %f" % (acc))
    Dataset.plot_sine(X_test, y_pred)
    plt.title("accuracy %f"%(acc))
    plt.savefig("plots/PyraLNet/sine_task/test (pyr, inter)=(%d,%d).png"%(bias_pyr, bias_inter))
    plt.clf()


def bars_task(square_size=3, N_train=3000, N_test=300, n_epochs=2, N_hidden=30):
    Path("plots/PyraLNet/bars_task").mkdir(parents=True, exist_ok=True)

    X_train, Y_train = Dataset.BarsDataset(square_size, samples_per_class=round(N_train / square_size), seed=42)[:]
    X_val, Y_val = Dataset.BarsDataset(square_size, samples_per_class=round(N_test / square_size), seed=None)[:]
    X_test, Y_test = Dataset.BarsDataset(square_size, samples_per_class=round(N_test / square_size), seed=None)[:]
    params = {"dims": [square_size ** 2, N_hidden, 3], "dt": 0.1, "gl": 0.1, "gb": 1.0, "ga": 0.8, "gd": 1.0,
              "gsom": 0.8,
              "eta": {"up": [0.01, 0.005], "pi": [0.01, 0], "ip": [0.01, 0]},
              "bias": {"pyr_on": False, "inter_on": False, "val": 0.0},
              "init_weights": {"up": 0.4, "down": 1, "pi": 0.5, "ip": 0.5}, "tau_w": 30, "noise": 0, "t_pattern": 100,
              "out_lag": 80, "tau_0": 3, "learning_lag":0}
    net = Net(params)

    #self-predicting state
    print("----learning self-predicting state----")
    rec_pots = [["W_up", "W_ip", "W_pi"], ["W_up"]]
    records, T, r_in, out_seq = net.run(X_train[np.random.permutation(N_train)], rec_pots=rec_pots, rec_dt=100)

    plt.title("Lateral Weights Convergence")
    plt.semilogy(T / 100, np.sum((records[0]["W_ip"].data - records[1]["W_up"].data) ** 2, axis=(1, 2)),
                 label="$||W^{(0)}_{ip}-W^{(1)}_{up}||^2$")
    plt.semilogy(T / 100, np.sum((records[0]["W_pi"].data + net.layer[0].W_down) ** 2, axis=(1, 2)),
                 label="$||W^{(0)}_{pi}+W^{(0)}_{down}||^2$")
    plt.xlabel("trial")
    plt.ylabel("squared error")
    plt.legend()
    plt.savefig("plots/PyraLNet/bar_task/lateral weights convergence.png")
    plt.show()


    #training
    print("----training----")
    records, T, r_in, u_trgt, out_seq, val_res = net.train(X_train, Y_train, X_val, Y_val, n_epochs=n_epochs, val_len=20, vals_per_epoch=15,
                                 n_out=3, classify=True, u_high=1.0, u_low=0.1, metric=accuracy, rec_pots=rec_pots, rec_dt=400)

    # plot exponential moving average of validation error
    plt.title("Validation error during training")
    plt.semilogy(val_res[:, 0], ewma(val_res[:, 1], round(len(val_res) / 10)), label="mse")
    plt.xlabel("trial")
    plt.ylabel("mean squared error")
    ax2 = plt.gca().twinx()
    ax2.plot(val_res[:, 0], ewma(val_res[:, 2], round(len(val_res) / 10)), c="g", label="accuracy")
    ax2.set_ylabel("accuracy")
    plt.savefig("plots/PyraLNet/bar_task/validation and accuracy during training.png")
    plt.show()

    plt.title("Lateral Weights Convergence")
    plt.semilogy(T / 100, np.sum((records[0]["W_ip"].data - records[1]["W_up"].data) ** 2, axis=(1, 2)),
                 label="$||W^{(0)}_{ip}-W^{(1)}_{up}||^2$")
    plt.semilogy(T / 100, np.sum((records[0]["W_pi"].data + net.layer[0].W_down) ** 2, axis=(1, 2)),
                 label="$||W^{(0)}_{pi}+W^{(0)}_{down}||^2$")
    plt.xlabel("trial")
    plt.ylabel("squared error")
    plt.legend()
    plt.savefig("plots/PyraLNet/bar_task/lateral weights convergence during training.png")
    plt.show()


    # test run
    print("----testing----")
    out_seq_test = net.run(X_test)
    y_pred = np.argmax(out_seq_test, axis=1)
    acc = np.sum(y_pred == Y_test) / len(Y_test)
    print("test set accuracy: %f" % (acc))



def yinyang_task(N_train=6000, N_test=600, n_epochs=45, N_hidden=120, mul=0.2):
    Path("plots/PyraLNet/yinyang_task").mkdir(parents=True, exist_ok=True)

    X_train, Y_train = Dataset.YinYangDataset(size=N_train, flipped_coords=True, seed=42)[:]
    X_val, Y_val = Dataset.YinYangDataset(size=N_test, flipped_coords=True, seed=None)[:]
    X_test, Y_test = Dataset.YinYangDataset(size=N_test, flipped_coords=True, seed=None)[:]
    params = {"dims": [4, N_hidden, 3], "dt": 0.1, "gl": 0.1, "gb": 1.0, "ga": 0.8, "gd": 1.0,
              "gsom": 0.1,  ###################changed!
              "eta": {"up": [mul * 0.03, mul * 0.01], "pi": [0, 0], "ip": [mul * 0.02, 0]},
              "bias": {"pyr_on": False, "inter_on": False, "val": 0.0},
              "init_weights": {"up": 0.1, "down": 1, "pi": 1, "ip": 1}, "tau_w": 30, "noise": 0, "t_pattern": 100,
              "out_lag": 80, "tau_0": 3, "learning_lag": 0}
    net = Net(params, act=sigmoid)
    net.reflect()

    out_seq, val_res = net.train(X_train, Y_train, X_val, Y_val, n_epochs=n_epochs, val_len=30, vals_per_epoch=15,
                                                      n_out=3, classify=True, u_high=1.0, u_low=0.1, metric=accuracy)

    # plot exponential moving average of validation error
    plt.title("Validation error during training")
    plt.semilogy(val_res[:, 0], ewma(val_res[:, 1], round(len(val_res) / 10)), label="mse")
    plt.xlabel("trial")
    plt.ylabel("mean squared error")
    ax2 = plt.gca().twinx()
    ax2.plot(val_res[:, 0], ewma(val_res[:, 2], round(len(val_res) / 10)), c="g", label="accuracy")
    ax2.set_ylabel("accuracy")
    plt.savefig("plots/PyraLNet/yinyang_task/validation and accuracy during training.png")
    plt.show()

    # test run
    out_seq_test = net.run(X_test)
    y_pred = np.argmax(out_seq_test, axis=1)
    acc = np.sum(y_pred == Y_test) / len(Y_test)
    print("test set accuracy: %f" % (acc))
    Dataset.plot_yy(X_test, y_pred)
    plt.title("accuracy %f" % (acc))
    plt.savefig("plots/PyraLNet/yinyang_task/test.png")
    plt.show()



# cluster version

def run_sine(params, name, dir):
    X_train, Y_train = Dataset.SineDataset(size=params["N_train"], flipped_coords=True, seed=params["seed"])[:]
    X_val, Y_val = Dataset.SineDataset(size=params["N_val"], flipped_coords=True, seed=None)[:]
    X_test, Y_test = Dataset.SineDataset(size=params["N_test"], flipped_coords=True, seed=None)[:]
    if params["model"]["act"] == "sigmoid":
        act = sigmoid
    elif params["model"]["act"] == "softReLU":
        act = soft_relu
    net = Net(params["model"], act=act, seed=None)

    if params["init_sps"]:
        net.reflect()

    rec_pots = [["W_up", "W_ip", "W_pi"], ["W_up"]]
    if params["track_sps"]:
        # self-predicting state
        print("----learning self-predicting state----")
        records, T, r_in, out_seq = net.run(X_train, rec_pots=rec_pots, rec_dt=100)

        plt.title("Lateral Weights Convergence pre-training")
        plt.semilogy(T / 100, np.sum((records[0]["W_ip"].data - records[1]["W_up"].data) ** 2, axis=(1, 2)),
                     label="$||W^{(0)}_{ip}-W^{(1)}_{up}||^2$")
        plt.semilogy(T / 100, np.sum((records[0]["W_pi"].data + net.layer[0].W_down) ** 2, axis=(1, 2)),
                     label="$||W^{(0)}_{pi}+W^{(0)}_{down}||^2$")
        plt.xlabel("trial")
        plt.ylabel("squared error")
        plt.legend()
        plt.savefig(dir + "lat_weights_conv_pre_%s.png" % (name))
        plt.clf()

    records, T, r_in, u_trgt, out_seq, val_res = net.train(X_train, Y_train, X_val, Y_val, n_epochs=params["N_epochs"],
                                                           val_len=params["val_len"],
                                                           vals_per_epoch=params["vals_per_epoch"],
                                                           n_out=2, classify=True, u_high=1.0, u_low=0.1,
                                                           metric=accuracy, rec_pots=rec_pots, rec_dt=1000)

    if params["track_sps"]:
        plt.title("Lateral Weights Convergence during training")
        plt.semilogy(T / 100, np.sum((records[0]["W_ip"].data - records[1]["W_up"].data) ** 2, axis=(1, 2)),
                     label="$||W^{(0)}_{ip}-W^{(1)}_{up}||^2$")
        plt.semilogy(T / 100, np.sum((records[0]["W_pi"].data + net.layer[0].W_down) ** 2, axis=(1, 2)),
                     label="$||W^{(0)}_{pi}+W^{(0)}_{down}||^2$")
        plt.xlabel("trial")
        plt.ylabel("squared error")
        plt.legend()
        plt.savefig(dir + "lat_weights_conv_during_%s.png" % (name))
        plt.clf()

    f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 2]}, figsize=(11, 5))

    # plot exponential moving average of validation error
    a0.set_title("Validation error during training %s" % (name))
    a0.semilogy(val_res[:, 0], ewma(val_res[:, 1], round(len(val_res) / 10)), label="mse")
    a0.set_xlabel("trial")
    a0.set_ylabel("mean squared error")
    ax2 = a0.twinx()
    ax2.plot(val_res[:, 0], ewma(val_res[:, 2], round(len(val_res) / 10)), c="g", label="accuracy")
    ax2.set_ylabel("accuracy")

    # validate on full validation set
    out_seq_test = net.run(X_val, learning_off=True)
    y_pred = np.argmax(out_seq_test, axis=1)
    acc_val = np.sum(y_pred == Y_val) / len(Y_val)
    print("validation set accuracy : %f " % (acc_val))

    # test run
    out_seq_test = net.run(X_test, learning_off=True)
    y_pred = np.argmax(out_seq_test, axis=1)
    acc = np.sum(y_pred == Y_test) / len(Y_test)
    print("test set accuracy : %f " % (acc))
    Dataset.plot_yy(X_test, y_pred, ax=a1)
    a1.set_title("test accuracy = %f" % (acc))

    plt.tight_layout()
    plt.savefig(dir + "result_%s.png" % (name))

    # save network state
    net.dump_weights(dir + "weights_%s.npy" % (name))

    with open(dir + "results.txt", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write("%s\t\t\t%f\t\t%f\t\t%f\n" % (name, val_res[-1, 1], acc_val, acc))
        fcntl.flock(f, fcntl.LOCK_UN)



def run_yinyang(params, name, dir):
    X_train, Y_train = Dataset.YinYangDataset(bottom_left=0, top_right=1, size=params["N_train"], flipped_coords=True, seed=params["seed"])[:]
    X_val, Y_val = Dataset.YinYangDataset(bottom_left=0, top_right=1, size=params["N_val"], flipped_coords=True, seed=None)[:]
    X_test, Y_test = Dataset.YinYangDataset(bottom_left=0, top_right=1, size=params["N_test"], flipped_coords=True, seed=None)[:]
    if params["model"]["act"] == "sigmoid":
        act = sigmoid
    elif params["model"]["act"] == "softReLU":
        act = soft_relu
    net = Net(params["model"], act=act, seed=None)

    if params["init_sps"]:
        net.reflect()

    rec_pots = [["W_up", "W_ip", "W_pi"], ["W_up"]]
    if params["track_sps"]:
        #self-predicting state
        print("----learning self-predicting state----")
        records, T, r_in, out_seq = net.run(X_train, rec_pots=rec_pots, rec_dt=100)

        plt.title("Lateral Weights Convergence pre-training")
        plt.semilogy(T / 100, np.sum((records[0]["W_ip"].data - records[1]["W_up"].data) ** 2, axis=(1, 2)),
                     label="$||W^{(0)}_{ip}-W^{(1)}_{up}||^2$")
        plt.semilogy(T / 100, np.sum((records[0]["W_pi"].data + net.layer[0].W_down) ** 2, axis=(1, 2)),
                     label="$||W^{(0)}_{pi}+W^{(0)}_{down}||^2$")
        plt.xlabel("trial")
        plt.ylabel("squared error")
        plt.legend()
        plt.savefig(dir+"lat_weights_conv_pre_%s.png" % (name))
        plt.clf()

    records, T, r_in, u_trgt, out_seq, val_res = net.train(X_train, Y_train, X_val, Y_val, n_epochs=params["N_epochs"], val_len=params["val_len"], vals_per_epoch=params["vals_per_epoch"],
                                 n_out=3, classify=True, u_high=1.0, u_low=0.1, metric=accuracy, rec_pots=rec_pots, rec_dt=1000)

    if params["track_sps"]:
        plt.title("Lateral Weights Convergence during training")
        plt.semilogy(T / 100, np.sum((records[0]["W_ip"].data - records[1]["W_up"].data) ** 2, axis=(1, 2)),
                     label="$||W^{(0)}_{ip}-W^{(1)}_{up}||^2$")
        plt.semilogy(T / 100, np.sum((records[0]["W_pi"].data + net.layer[0].W_down) ** 2, axis=(1, 2)),
                     label="$||W^{(0)}_{pi}+W^{(0)}_{down}||^2$")
        plt.xlabel("trial")
        plt.ylabel("squared error")
        plt.legend()
        plt.savefig(dir + "lat_weights_conv_during_%s.png" % (name))
        plt.clf()
    

    f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 2]}, figsize=(11,5))

    # plot exponential moving average of validation error
    a0.set_title("Validation error during training %s" % (name))
    a0.semilogy(val_res[:, 0], ewma(val_res[:, 1], round(len(val_res) / 10)), label="mse")
    a0.set_xlabel("trial")
    a0.set_ylabel("mean squared error")
    ax2 = a0.twinx()
    ax2.plot(val_res[:, 0], ewma(val_res[:, 2], round(len(val_res) / 10)), c="g", label="accuracy")
    ax2.set_ylabel("accuracy")

    # validate on full validation set
    out_seq_test = net.run(X_val, learning_off=True)
    y_pred = np.argmax(out_seq_test, axis=1)
    acc_val = np.sum(y_pred == Y_val) / len(Y_val)
    print("validation set accuracy : %f " % (acc_val))

    # test run
    out_seq_test = net.run(X_test, learning_off=True)
    y_pred = np.argmax(out_seq_test, axis=1)
    acc = np.sum(y_pred == Y_test) / len(Y_test)
    print("test set accuracy : %f " % (acc))
    Dataset.plot_yy(X_test, y_pred, ax=a1)
    a1.set_title("test accuracy = %f" % (acc))

    plt.tight_layout()
    plt.savefig(dir+"result_%s.png" % (name))

    # save network state
    net.dump_weights(dir + "weights_%s.npy" % (name))

    with open(dir+"results.txt", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write("%s\t\t\t%f\t\t%f\t\t%f\n"%(name, val_res[-1, 1], acc_val, acc))
        fcntl.flock(f, fcntl.LOCK_UN)


if __name__=="__main__":
    sine_task(bias_pyr=False, bias_inter=False)
    sine_task(bias_pyr=True, bias_inter=False)
    sine_task(bias_pyr=True, bias_inter=True)

'''

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
    elif args.task == "sine":
        run_sine(params, params["name"], args.dir)
    else:
        raise Exception("task not known!")
'''