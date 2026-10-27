import numpy as np
import scipy.sparse as sp


class EASE:
    def __init__(self, reg=200.0, verbose=False):
        self.reg = reg
        self.verbose = verbose
        self.user_index_ = None
        self.item_index_ = None
        self.B_ = None
        self.rating_matrix_ = None

    def fit(self, train_df, n_users=None, n_items=None):
        users = train_df["user_id"].unique()
        items = train_df["item_id"].unique()
        self.user_index_ = {u: i for i, u in enumerate(users)}
        self.item_index_ = {it: i for i, it in enumerate(items)}

        rows = train_df["user_id"].map(self.user_index_).to_numpy()
        cols = train_df["item_id"].map(self.item_index_).to_numpy()
        data = np.ones(len(train_df), dtype=np.float32)
        n_u, n_i = len(self.user_index_), len(self.item_index_)

        X = sp.csr_matrix((data, (rows, cols)), shape=(n_u, n_i), dtype=np.float32)
        self.rating_matrix_ = X

        G = (X.T @ X).toarray().astype(np.float64)
        G[np.diag_indices(n_i)] += self.reg
        P = np.linalg.inv(G)

        B = P / (-np.diag(P)[None, :])
        np.fill_diagonal(B, 0.0)
        self.B_ = B.astype(np.float32)

        if self.verbose:
            print(f"[EASE] users={n_u}, items={n_i}, nnz={X.nnz}")
        return self

    def _scores_matrix(self):
        return np.asarray(self.rating_matrix_ @ self.B_)

    def predict(self, user_ids, item_ids):
        S = self._scores_matrix()
        res = np.zeros(len(user_ids), dtype=np.float32)
        for idx, (u, i) in enumerate(zip(user_ids, item_ids)):
            ju = self.user_index_.get(u)
            ji = self.item_index_.get(i)
            if ju is None or ji is None:
                continue
            res[idx] = S[ju, ji]
        return res

    def recommend(self, user_ids, candidates, k=10, exclude_seen=True, train_df=None):
        S = self._scores_matrix()
        item_ids = np.array(list(self.item_index_.keys()))
        item_pos = {it: idx for idx, it in enumerate(item_ids)}

        seen = {}
        if exclude_seen and train_df is not None:
            for u, grp in train_df.groupby("user_id"):
                seen[u] = set(grp["item_id"].tolist())

        cand_set = set(candidates) if candidates is not None else None
        result = {}
        for u in user_ids:
            ju = self.user_index_.get(u)
            if ju is None:
                result[u] = list(item_ids[:k])
                continue
            s = S[ju].copy()
            if cand_set is not None:
                mask = np.array([it in cand_set for it in item_ids])
                s = np.where(mask, s, -np.inf)
            if exclude_seen and u in seen:
                for it in seen[u]:
                    j = item_pos.get(it)
                    if j is not None:
                        s[j] = -np.inf
            top = np.argpartition(-s, kth=min(k, len(s) - 1))[:k]
            top = top[np.argsort(-s[top])]
            result[u] = [item_ids[j] for j in top]
        return result