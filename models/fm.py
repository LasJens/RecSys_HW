import numpy as np
import torch
import torch.nn as nn


class _TorchFM(nn.Module):
    def __init__(self, n_features, n_factors):
        super().__init__()
        self.w0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.zeros(n_features))
        self.V = nn.Parameter(torch.randn(n_features, n_factors) * 0.01)

    def forward(self, x):
        w_sum = self.w0 + self.w[x].sum(dim=1)
        v = self.V[x]
        v_sum = v.sum(dim=1)
        inter = 0.5 * (v_sum ** 2 - (v ** 2).sum(dim=1)).sum(dim=1)
        return w_sum + inter


class FM:
    def __init__(self, n_factors=16, learning_rate=0.01, n_epochs=5,
                 reg_v=1e-5, batch_size=8192, random_state=42,
                 verbose=False, device="cpu"):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.reg_v = reg_v
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.device = device
        self.user_index_ = None
        self.item_index_ = None
        self.cat_values_ = None
        self.meta_cols_ = ["store", "main_category", "cat_top2", "cat_top3",
                          "price_bin", "has_price", "has_description"]
        self.offsets_ = None
        self.n_features_ = 0
        self.model_ = None

    def _build_cat_maps(self, meta_df):
        self.cat_values_ = {}
        for c in self.meta_cols_:
            vals = sorted(meta_df[c].astype(str).unique())
            self.cat_values_[c] = {v: i for i, v in enumerate(vals)}

    def _meta_index(self, meta_df):
        out = {}
        m = meta_df.set_index("item_id")
        for c in self.meta_cols_:
            cmap = self.cat_values_[c]
            unk = len(cmap)
            out[c] = m[c].astype(str).map(cmap).fillna(unk).astype(int)
        return out

    def _build_matrix(self, df, meta_df, fit=False):
        if fit:
            self.user_index_ = {u: i for i, u in enumerate(df["user_id"].unique())}
            self.item_index_ = {it: i for i, it in enumerate(df["item_id"].unique())}
            self._build_cat_maps(meta_df)
            n_u, n_i = len(self.user_index_), len(self.item_index_)
            offsets = [0, n_u, n_u + n_i]
            for c in self.meta_cols_:
                offsets.append(offsets[-1] + len(self.cat_values_[c]) + 1)
            self.offsets_ = offsets
            self.n_features_ = offsets[-1]
        n_i = len(self.item_index_)
        meta_idx = self._meta_index(meta_df)
        u_arr = df["user_id"].map(self.user_index_).fillna(-1).astype(int).to_numpy()
        i_arr = df["item_id"].map(self.item_index_).fillna(-1).astype(int).to_numpy()
        n = len(df)
        out = np.zeros((n, 2 + len(self.meta_cols_)), dtype=np.int64)
        out[:, 0] = u_arr
        out[:, 1] = self.offsets_[1] + i_arr
        for k, c in enumerate(self.meta_cols_):
            codes = meta_idx[c].reindex(df["item_id"].values).fillna(len(self.cat_values_[c])).astype(int).to_numpy()
            out[:, 2 + k] = self.offsets_[2 + k] + codes
        return out

    def fit(self, train_df, meta_df):
        X = self._build_matrix(train_df, meta_df, fit=True)
        y = train_df["rating"].to_numpy(dtype=np.float32)
        torch.manual_seed(self.random_state)
        self.model_ = _TorchFM(self.n_features_, self.n_factors).to(self.device)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate,
                               weight_decay=self.reg_v)
        Xt = torch.from_numpy(X).long().to(self.device)
        yt = torch.from_numpy(y).float().to(self.device)
        n = len(y)
        for epoch in range(self.n_epochs):
            perm = torch.randperm(n, device=self.device)
            total = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                pred = self.model_(Xt[idx])
                loss = ((pred - yt[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            if self.verbose:
                print(f"[FM] epoch {epoch+1}/{self.n_epochs} MSE={total/n:.4f}")
        return self

    def fm_recommend_batched(model, user_ids, candidates, meta_df, k=10,
                            exclude_seen=True, train_df=None, batch_users=2000):
        item_ids = np.array(list(model.item_index_.keys()))
        item_pos = {it: idx for idx, it in enumerate(item_ids)}
        pool_idx = np.array([item_pos[it] for it in candidates if it in item_pos])
        pool_items = item_ids[pool_idx]
        meta_idx = model._meta_index(meta_df)
        W = model.model_.w.detach().cpu().numpy()
        V = model.model_.V.detach().cpu().numpy()
        w0 = model.model_.w0.item()
        meta_codes = []
        for kk, c in enumerate(model.meta_cols_):
            codes = meta_idx[c].reindex(pool_items).fillna(len(model.cat_values_[c])).astype(int).to_numpy()
            meta_codes.append(codes + model.offsets_[2 + kk])
        item_idx = pool_idx + model.offsets_[1]
        w_pool = W[item_idx] + sum(W[mc] for mc in meta_codes)
        V_pool = V[item_idx] + sum(V[mc] for mc in meta_codes)
        C_pool = V[item_idx]**2 + sum(V[mc]**2 for mc in meta_codes)
        static = w0 + w_pool - 0.5 * C_pool.sum(axis=1) + 0.5 * (V_pool**2).sum(axis=1)
        seen = {}
        if exclude_seen and train_df is not None:
            for u, grp in train_df.groupby("user_id"):
                seen[u] = set(grp["item_id"].tolist())
        result = {}
        uids = list(user_ids)
        for start in range(0, len(uids), batch_users):
            batch = uids[start:start + batch_users]
            js = np.array([model.user_index_.get(u, -1) for u in batch])
            known = js >= 0
            V_u = np.zeros((len(batch), model.n_factors), dtype=np.float32)
            w_u = np.zeros(len(batch), dtype=np.float32)
            if known.any():
                V_u[known] = V[js[known]]
                w_u[known] = W[js[known]]
            scores = static[None, :] + w_u[:, None] + V_u @ V_pool.T
            for bi, u in enumerate(batch):
                if exclude_seen and u in seen:
                    seen_idx = np.array([item_pos[it] for it in seen[u] if it in item_pos])
                    if len(seen_idx) > 0:
                        scores[bi][np.isin(pool_idx, seen_idx)] = -np.inf
                kk = min(k, scores.shape[1])
                top = np.argpartition(-scores[bi], kth=kk - 1)[:kk]
                top = top[np.argsort(-scores[bi][top])]
                result[u] = pool_items[top].tolist()
        return result