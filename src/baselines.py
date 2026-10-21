"""Label-free completion baselines fitted on train and tuned on validation only."""
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from evaluation import make_mask

METHODS = ('zero', 'mean', 'knn', 'ridge', 'mlp')


def _normalize(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Features must be a finite 2D matrix')
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norm, 1e-12)).astype(np.float32)


def _mse(pred, target):
    return float(np.mean((np.asarray(pred) - np.asarray(target)) ** 2))


class _MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, dim))

    def forward(self, x):
        return self.net(x)


class CompletionBaselines:
    def __init__(self, train_img, train_txt, val_img, val_txt, seed=42, device='cpu',
                 val_rate=0.3, val_seed=777, knn_k=(1, 3, 5, 10, 20),
                 ridge_alpha=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
                 mlp_hidden=512, mlp_epochs=100, mlp_patience=15, mlp_lr=0.001,
                 mlp_weight_decay=0.0001, mlp_batch_size=128):
        self.train = {'img': _normalize(train_img), 'txt': _normalize(train_txt)}
        self.val = {'img': _normalize(val_img), 'txt': _normalize(val_txt)}
        if self.train['img'].shape != self.train['txt'].shape:
            raise ValueError('Train image/text features must have the same shape')
        if self.val['img'].shape != self.val['txt'].shape:
            raise ValueError('Val image/text features must have the same shape')
        self.dim = self.train['img'].shape[1]
        self.device = torch.device(device)
        self.seed = int(seed)
        self.val_rate = float(val_rate)
        self.val_seed = int(val_seed)
        self.knn_k = tuple(sorted({int(k) for k in knn_k if int(k) >= 1}))
        self.ridge_alpha = tuple(float(a) for a in ridge_alpha)
        self.mlp_hidden = int(mlp_hidden)
        self.mlp_epochs = int(mlp_epochs)
        self.mlp_patience = int(mlp_patience)
        self.mlp_lr = float(mlp_lr)
        self.mlp_weight_decay = float(mlp_weight_decay)
        self.mlp_batch_size = int(mlp_batch_size)
        if not self.knn_k or not self.ridge_alpha or self.mlp_epochs < 1:
            raise ValueError('Baseline hyperparameter grids must be nonempty')
        self.models = {}
        self.choices = {}
        self.val_mse = {}
        self._fit_all()

    def _fit_all(self):
        val_mask = make_mask(len(self.val['img']), self.val_rate, 'random', self.val_seed)
        for direction in ('img2txt', 'txt2img'):
            src_key, tgt_key = ('img', 'txt') if direction == 'img2txt' else ('txt', 'img')
            target_col = 1 if direction == 'img2txt' else 0
            missing = val_mask[:, target_col] == 0
            if not missing.any():
                raise ValueError('Validation mask left no missing target for baseline tuning')
            source_train = self.train[src_key]
            target_train = self.train[tgt_key]
            source_val = self.val[src_key][missing]
            target_val = self.val[tgt_key][missing]
            self.models[direction] = {}
            self.choices[direction] = {}
            self.val_mse[direction] = {}

            self.models[direction]['zero'] = None
            self.val_mse[direction]['zero'] = _mse(np.zeros_like(target_val), target_val)

            mean_vec = _normalize(target_train.mean(axis=0, keepdims=True))[0]
            self.models[direction]['mean'] = mean_vec
            self.val_mse[direction]['mean'] = _mse(
                np.repeat(mean_vec[None], len(target_val), axis=0), target_val)

            best = None
            for k in self.knn_k:
                kk = min(k, len(source_train))
                model = NearestNeighbors(n_neighbors=kk, metric='cosine').fit(source_train)
                pred = self._knn_predict(model, source_val, target_train)
                score = _mse(pred, target_val)
                if best is None or score < best[0]:
                    best = (score, kk, model)
            self.val_mse[direction]['knn'] = best[0]
            self.choices[direction]['knn'] = best[1]
            self.models[direction]['knn'] = best[2]

            best = None
            for alpha in self.ridge_alpha:
                model = Ridge(alpha=alpha).fit(source_train, target_train)
                pred = _normalize(model.predict(source_val))
                score = _mse(pred, target_val)
                if best is None or score < best[0]:
                    best = (score, alpha, model)
            self.val_mse[direction]['ridge'] = best[0]
            self.choices[direction]['ridge'] = best[1]
            self.models[direction]['ridge'] = best[2]

            model, score = self._fit_mlp(source_train, target_train,
                                         self.val[src_key], self.val[tgt_key], direction)
            self.val_mse[direction]['mlp'] = score
            self.choices[direction]['mlp'] = self.mlp_epochs
            self.models[direction]['mlp'] = model

    def _knn_predict(self, model, source, target_train):
        _, indices = model.kneighbors(source)
        return _normalize(target_train[indices].mean(axis=1))

    def _fit_mlp(self, source_train, target_train, source_val, target_val, direction):
        offset = 0 if direction == 'img2txt' else 1
        torch.manual_seed(self.seed + 1000 + offset)
        model = _MLP(self.dim, self.mlp_hidden).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.mlp_lr,
                                      weight_decay=self.mlp_weight_decay)
        dataset = TensorDataset(torch.from_numpy(source_train), torch.from_numpy(target_train))
        loader = DataLoader(dataset, batch_size=self.mlp_batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(self.seed + 2000 + offset),
                            num_workers=0)
        vx = torch.from_numpy(source_val).to(self.device)
        vy = torch.from_numpy(target_val).to(self.device)
        best, best_state, stale = float('inf'), None, 0
        for _ in range(1, self.mlp_epochs + 1):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                loss = torch.nn.functional.mse_loss(model(xb), yb)
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite baseline training loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.inference_mode():
                vloss = torch.nn.functional.mse_loss(model(vx), vy).item()
            if vloss < best:
                best = vloss
                stale = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= self.mlp_patience:
                    break
        model.load_state_dict(best_state)
        model.eval()
        return model, best

    def predict(self, method, direction, source):
        if method not in METHODS:
            raise ValueError(f'Unknown baseline {method}')
        if direction not in ('img2txt', 'txt2img'):
            raise ValueError(f'Unknown direction {direction}')
        source = np.asarray(source, dtype=np.float32)
        if source.ndim != 2 or source.shape[1] != self.dim:
            raise ValueError('Source feature shape does not match baseline input')
        if method == 'zero':
            return np.zeros((len(source), self.dim), dtype=np.float32)
        if method == 'mean':
            return np.repeat(self.models[direction]['mean'][None], len(source), axis=0)
        if method == 'knn':
            return self._knn_predict(self.models[direction]['knn'], source,
                                     self.train['txt' if direction == 'img2txt' else 'img'])
        if method == 'ridge':
            return _normalize(self.models[direction]['ridge'].predict(source))
        model = self.models[direction]['mlp']
        with torch.inference_mode():
            pred = model(torch.from_numpy(source).to(self.device)).cpu().numpy()
        return _normalize(pred)

    def complete(self, method, img, txt, mask):
        if method not in METHODS:
            raise ValueError(f'Unknown baseline {method}')
        img = np.asarray(img, dtype=np.float32)
        txt = np.asarray(txt, dtype=np.float32)
        mask = np.asarray(mask)
        if img.shape != txt.shape or mask.shape != (len(img), 2):
            raise ValueError('Feature/mask shapes are inconsistent')
        out_img = np.where(mask[:, :1] == 1, img, 0.).astype(np.float32)
        out_txt = np.where(mask[:, 1:] == 1, txt, 0.).astype(np.float32)
        missing_img = mask[:, 0] == 0
        missing_txt = mask[:, 1] == 0
        if missing_img.any():
            out_img[missing_img] = self.predict(method, 'txt2img', txt[missing_img])
        if missing_txt.any():
            out_txt[missing_txt] = self.predict(method, 'img2txt', img[missing_txt])
        return out_img, out_txt

    def best_method(self):
        scores = {m: float(np.mean([self.val_mse[d][m] for d in ('img2txt', 'txt2img')]))
                  for m in METHODS}
        best = min(scores, key=scores.get)
        return best, scores

    def summary(self):
        return {'choices': self.choices, 'val_mse': self.val_mse,
                'best_method': self.best_method()[0]}
