import numpy as np

class HHM_Algorithm:
    def __init__(self, in_st_amount=None, ex_st_amount=None, mu=None, A=None, B=None, random_state=42, verbose=1):
        self.rng_ = np.random.default_rng(random_state)
        self.verbose_ = verbose

        if A is not None:
            self.transmat_ = A
            self.in_st_amount_ = self.transmat_.shape[0]
        else:
            self.in_st_amount_ = in_st_amount

        if B is not None:
            self.emissionprob_ = B
            self.ex_st_amount_ = self.emissionprob_.shape[1]
        else:
            self.ex_st_amount_ = ex_st_amount

        if mu is not None:
            self.startprob_ = mu

        self.E_ = np.arange(self.in_st_amount_)
        self.F_ = np.arange(self.ex_st_amount_)

        if A is None and B is None and mu is None:
            self.startprob_, self.transmat_, self.emissionprob_ = self.generate_est()

    def simulate(self, iter_nb, n_samples):
        external_states = np.zeros((n_samples, iter_nb), dtype='int')
        for r in range(n_samples):
            prob_distrib = np.copy(self.startprob_)
            for iter in range(iter_nb):
                internal_state = self._generate_state(prob_distrib) - 1
                prob_distrib = self.transmat_[internal_state]
                external_state = self._generate_state(self.emissionprob_[internal_state])
                external_states[r, iter] = external_state - 1
        return external_states

    def fit(self, output, max_iter, tol, reestimate=True):
        iter_n = 0
        stats = []
        log_prob = 0
        old_log_prob = np.inf

        while iter_n != max_iter and np.abs(log_prob - old_log_prob) > tol:
            _, norm_factors, alphas, betas = self.forward_and_backward(output)
            T = len(output)
            gammas = np.zeros((T - 1, self.in_st_amount_, self.in_st_amount_))
            digammas = np.zeros((T, self.in_st_amount_))

            for t in range(T - 1):
                gammas[t] = alphas[t, :, None] * self.transmat_ * self.emissionprob_[:, output[t + 1]] * betas[t + 1, :]
                digammas[t] = np.sum(gammas[t], axis=1)
                gammas[t] /= np.sum(gammas[t])

            digammas[T - 1] = alphas[T - 1]
            digammas[T - 1] /= np.sum(digammas[T - 1])

            if reestimate:
                self.startprob_ = digammas[0]
                for i in self.E_:
                    for j in self.E_:
                        self.transmat_[i, j] = np.sum(gammas[:, i, j]) / np.sum(digammas[:-1, i])

            for i in self.E_:
                self.emissionprob_[i, :] = np.dot(digammas[:, i], (output[:, None] == self.F_)) / np.sum(digammas[:, i])

            old_log_prob = log_prob
            log_prob = -np.sum(np.log(norm_factors))
            stats.append(log_prob)
            iter_n += 1

        return stats

    def forward_and_backward(self, output):
        T = len(output)
        alphas = np.zeros((T, self.in_st_amount_))
        betas = np.zeros((T, self.in_st_amount_))
        norm_factors = np.zeros(T, dtype=np.float64)

        for t in range(T):
            if t == 0:
                alphas[t] = self.startprob_
            else:
                alphas[t] = np.dot(alphas[t - 1], self.transmat_)
            alphas[t] *= self.emissionprob_[:, output[t]]
            norm_factors[t] = 1 / np.sum(alphas[t])
            alphas[t] *= norm_factors[t]

        for t in range(T - 1, -1, -1):
            if t == T - 1:
                betas[t] = 1
            else:
                betas[t] = np.dot(self.transmat_ * betas[t + 1], self.emissionprob_[:, output[t + 1]])
            betas[t] *= norm_factors[t]

        res_prob = np.exp(-np.sum(np.log(norm_factors)))
        return res_prob, norm_factors, alphas, betas

    def _generate_state(self, prob_distrib):
        u = self.rng_.random()
        state = np.digitize(u, np.cumsum(prob_distrib))
        return state

    def generate_est(self):
        mu_est = np.full(self.in_st_amount_, 1/self.in_st_amount_)
        noise = self.rng_.dirichlet(np.ones(self.in_st_amount_))
        mu_est += noise * 0.015
        mu_est /= np.sum(mu_est)

        A_est = np.ones((self.in_st_amount_, self.in_st_amount_)) / self.in_st_amount_
        noise = self.rng_.dirichlet(np.ones(self.in_st_amount_), size=self.in_st_amount_)
        A_est += noise * 0.015
        A_est /= np.sum(A_est, axis=1)

        B_est = np.ones((self.in_st_amount_, self.ex_st_amount_)) / self.ex_st_amount_
        noise = self.rng_.dirichlet(np.ones(self.ex_st_amount_), size=self.in_st_amount_)
        B_est += noise * 0.015
        B_est /= np.sum(B_est, axis=1).reshape(-1, 1)

        return mu_est, A_est, B_est

    def viterbi(self, sequence):
        old_settings = np.seterr(divide='ignore')
        deltas = np.zeros((len(sequence), self.in_st_amount_))
        deltas_argmax = np.zeros((len(sequence), self.in_st_amount_), dtype='int')

        for t in range(len(sequence)):
            for x in self.E_:
                if t == 0:
                    deltas[t, x] = np.log(self.startprob_[x]) + np.log(self.emissionprob_[x, t])
                else:
                    deltas[t, x] = np.max(deltas[t-1] + np.log(self.transmat_[:, x])) + np.log(self.emissionprob_[x, sequence[t]])
                    deltas_argmax[t, x] = np.argmax(deltas[t-1] + np.log(self.transmat_[:, x]))

        np.seterr(**old_settings)

        states = np.zeros(len(sequence), dtype='int')
        for t in range(len(sequence) - 1, -1, -1):
            states[t] = np.argmax(deltas[t]) if t == len(sequence) - 1 else deltas_argmax[t + 1, states[t + 1]]

        return states
