import numpy as np


class TimeSVDpp:
    def __init__(self, n_factors=20, n_epochs=20, learning_rate=0.005, reg=0.02,
                 beta=0.4, n_time_bins=30, random_state=42, verbose=False):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.reg = reg
        self.beta = beta
        self.n_time_bins = n_time_bins
        self.random_state = random_state
        self.verbose = verbose
        self.mu_ = 0.0
        self.bu_ = None
        self.bi_ = None
        self.p_ = None
        self.q_ = None
        self.y_ = None
        self.alpha_u_ = None
        self.user_bin_bias_ = None
        self.item_bin_bias_ = None
        self.user_bin_p_ = None
        self.user_index_ = None
        self.item_index_ = None
        self.user_mean_time_ = None
        self.time_bin_edges_ = None

    def _bin(self, ts):
        idx = np.searchsorted(self.time_bin_edges_, ts, side="right") - 1
        return np.clip(idx, 0, self.n_time_bins - 1)

    def _dev(self, u, ts):
        t_mean = self.user_mean_time_[u]
        diff = ts - t_mean
        return np.sign(diff) * np.abs(diff) ** self.beta

    def fit(self, train_df, n_users=None, n_items=None):
        rng = np.random.default_rng(self.random_state)
        df = train_df
        users = df["user_id"].unique()
        items = df["item_id"].unique()
        self.user_index_ = {u: i for i, u in enumerate(users)}
        self.item_index_ = {it: i for i, it in enumerate(items)}
        n_u, n_i = len(users), len(items)

        u_arr = df["user_id"].map(self.user_index_).to_numpy()
        i_arr = df["item_id"].map(self.item_index_).to_numpy()
        r_arr = df["rating"].to_numpy(dtype=np.float32)
        t_arr = df["timestamp"].to_numpy(dtype=np.float64)

        self.mu_ = float(r_arr.mean())
        self.bu_ = np.zeros(n_u, dtype=np.float32)
        self.bi_ = np.zeros(n_i, dtype=np.float32)
        self.p_ = rng.normal(0, 0.01, size=(n_u, self.n_factors)).astype(np.float32)
        self.q_ = rng.normal(0, 0.01, size=(n_i, self.n_factors)).astype(np.float32)
        self.y_ = rng.normal(0, 0.01, size=(n_i, self.n_factors)).astype(np.float32)
        self.alpha_u_ = np.zeros(n_u, dtype=np.float32)
        self.user_bin_bias_ = np.zeros((n_u, self.n_time_bins), dtype=np.float32)
        self.item_bin_bias_ = np.zeros((n_i, self.n_time_bins), dtype=np.float32)
        self.user_bin_p_ = np.zeros((n_u, self.n_time_bins, self.n_factors), dtype=np.float32)

        t_min, t_max = t_arr.min(), t_arr.max()
        self.time_bin_edges_ = np.linspace(t_min, t_max, self.n_time_bins + 1)
        self.time_bin_edges_[-1] += 1

        self.user_mean_time_ = np.zeros(n_u, dtype=np.float64)
        for u in range(n_u):
            self.user_mean_time_[u] = t_arr[u_arr == u].mean()

        user_items = [[] for _ in range(n_u)]
        for u, i in zip(u_arr, i_arr):
            user_items[u].append(i)
        user_items = [np.array(x, dtype=np.int64) for x in user_items]

        lr = self.learning_rate
        reg = self.reg

        for epoch in range(self.n_epochs):
            perm = rng.permutation(len(df))
            for k in perm:
                u = u_arr[k]; i = i_arr[k]; r = r_arr[k]; t = t_arr[k]
                b = self._bin(t)
                dev = self._dev(u, t)

                N_u = user_items[u]
                if len(N_u) > 0:
                    y_sum = self.y_[N_u].sum(axis=0) / np.sqrt(len(N_u))
                else:
                    y_sum = np.zeros(self.n_factors, dtype=np.float32)

                p_u = self.p_[u] + self.alpha_u_[u] * dev + self.user_bin_p_[u, b]
                q_i = self.q_[i]
                pred = (self.mu_ + self.bu_[u] + self.bi_[i]
                        + self.user_bin_bias_[u, b] + self.item_bin_bias_[i, b]
                        + q_i @ (p_u + y_sum))
                err = r - pred

                self.bu_[u] += lr * (err - reg * self.bu_[u])
                self.bi_[i] += lr * (err - reg * self.bi_[i])
                self.user_bin_bias_[u, b] += lr * (err - reg * self.user_bin_bias_[u, b])
                self.item_bin_bias_[i, b] += lr * (err - reg * self.item_bin_bias_[i, b])

                q_old = q_i.copy()
                self.q_[i] += lr * (err * (p_u + y_sum) - reg * self.q_[i])
                self.p_[u] += lr * (err * q_old - reg * self.p_[u])
                self.alpha_u_[u] += lr * (err * dev * (q_old @ np.ones(self.n_factors)) * 0 - reg * self.alpha_u_[u])
                self.alpha_u_[u] += lr * (err * (q_old @ (np.ones(self.n_factors) * dev)) * 0)
                self.user_bin_p_[u, b] += lr * (err * q_old - reg * self.user_bin_p_[u, b])

                if len(N_u) > 0:
                    self.y_[N_u] += lr * (err * q_old / np.sqrt(len(N_u)) - reg * self.y_[N_u])

            if self.verbose:
                print(f"[TimeSVDpp] epoch {epoch + 1}/{self.n_epochs}")
        return self

    def predict(self, user_ids, item_ids, timestamps=None):
        res = np.zeros(len(user_ids), dtype=np.float32)
        if timestamps is None:
            timestamps = np.full(len(user_ids), self.time_bin_edges_[-2], dtype=np.float64)
        for k, (u, i, t) in enumerate(zip(user_ids, item_ids, timestamps)):
            ju = self.user_index_.get(u)
            ji = self.item_index_.get(i)
            if ju is None or ji is None:
                res[k] = self.mu_
                continue
            b = self._bin(np.array([t]))[0]
            dev = self._dev(ju, t)
            user_items = np.array([], dtype=np.int64)
            y_sum = np.zeros(self.n_factors, dtype=np.float32)
            p_u = self.p_[ju] + self.alpha_u_[ju] * dev + self.user_bin_p_[ju, b]
            q_i = self.q_[ji]
            res[k] = (self.mu_ + self.bu_[ju] + self.bi_[ji]
                      + self.user_bin_bias_[ju, b] + self.item_bin_bias_[ji, b]
                      + q_i @ (p_u + y_sum))
        return res

    def recommend(self, user_ids, candidates, k=10, exclude_seen=True,
                  train_df=None, eval_time=None):
        item_ids = np.array(list(self.item_index_.keys()))
        item_pos = {it: idx for idx, it in enumerate(item_ids)}
        if candidates is not None:
            pool_idx = np.array([item_pos[it] for it in candidates if it in item_pos])
        else:
            pool_idx = np.arange(len(item_ids))
        pool_items = item_ids[pool_idx]
        kk = min(k, len(pool_idx))
        if eval_time is None:
            eval_time = self.time_bin_edges_[-2]

        seen_idx = {}
        user_item_idx = {}
        if train_df is not None:
            for u, grp in train_df.groupby("user_id"):
                idxs = grp["item_id"].map(item_pos).dropna().astype(int).to_numpy()
                user_item_idx[u] = idxs
                if exclude_seen:
                    seen_idx[u] = idxs

        result = {}
        b_eval = self._bin(np.array([eval_time]))[0]
        for u in user_ids:
            ju = self.user_index_.get(u)
            if ju is None:
                result[u] = pool_items[:kk].tolist()
                continue
            dev = self._dev(ju, eval_time)
            u_items = user_item_idx.get(u, np.array([], dtype=int))
            if len(u_items) > 0:
                y_sum = self.y_[u_items].sum(axis=0) / np.sqrt(len(u_items))
            else:
                y_sum = np.zeros(self.n_factors, dtype=np.float32)
            p_u = self.p_[ju] + self.alpha_u_[ju] * dev + self.user_bin_p_[ju, b_eval]
            base = self.mu_ + self.bu_[ju] + self.user_bin_bias_[ju, b_eval]
            s = base + self.bi_[pool_idx] + self.item_bin_bias_[pool_idx, b_eval] + \
                self.q_[pool_idx] @ (p_u + y_sum)
            if exclude_seen and u in seen_idx:
                s[np.isin(pool_idx, seen_idx[u])] = -np.inf
            top = np.argpartition(-s, kth=kk - 1)[:kk]
            top = top[np.argsort(-s[top])]
            result[u] = pool_items[top].tolist()
        return result