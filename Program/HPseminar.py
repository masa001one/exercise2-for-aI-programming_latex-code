from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cho_solve, cho_factor


# 想定する真のblack-box関数
def f(x):
    return np.cos(2 * x) + np.sin(3 * x) - 0.5 * x


# 課題1: ガウスカーネルの定義 (課題.pdfに載っているカーネル関数)
# 獲得関数の課題と異なり, カーネルパラメータker_var, length_scaleに応じた計算を行う
# n個の入力x1, m個の入力x2にに対し, カーネル行列を返す
# shape=(n,m)のカーネル行列は(i,j)番目の要素に k(x1_i, x2_j) を持つ
def gauss_kernel(x1, x2, ker_var, length_scale):
    # x1.shape=(n,1)
    # x2.shape=(m,1)

    # ヒント: (i,j)番目の要素に (x1_i -x2_j) を持つようなn×m行列を作成してからカーネル行列を計算すると良い
    x_diff = x1 - x2.flatten()  # shape=(n,m)
    r = np.square(x_diff)
    return ker_var * np.exp(-r / (2 * length_scale ** 2))


# (定数倍の単位行列が加わった)カーネル行列の逆行列(K+σI)^{-1} を計算する関数
# コレスキー分解を用いることで高速&数値的に安定した逆行列計算を行っている
def kernel_inv(k_matrix):
    k_dim = k_matrix.shape[0]
    return cho_solve((cho_factor(k_matrix)), np.eye(k_dim))


# 指定したハイパラで構築したGPの予測分布をプロットする関数
# ker_var, lengthscaleはガウスカーネルに関するパラメータ
# noise_var: GPモデル構築時に仮定するノイズ分散
# seedは乱数生成器に与えるシード値 (結果の再現性を保つため, 適当に固定しておく)
def plot_gp(ker_var, length_scale, gp_noise_var, seed):
    # 実験条件に関するパラメータ
    x_domain = (-5, 5)  # 入力xの範囲
    n_grid = 1000  # グリッド数
    true_noise_scale = 1e-2  # 実際の正規ノイズの標準偏差

    # 初期化
    rng = np.random.default_rng(seed)  # 乱数生成器
    x = np.linspace(x_domain[0], x_domain[1], n_grid).reshape(
        n_grid, 1)  # x_domainをグリッドで区切ったもの. x.shape=(n_grid, 1)
    f_true = f(x)  # 各xにおける真の関数値. f_true.shape=(n_grid, 1)

    # 観測データの作成 (最初は初期点のみ)
    train_i = [50, 100, 300, 700, 800]  # 観測データのindex (個数や番号は適当に変更する)
    n_train = len(train_i)

    # 計算の都合上, x_train, y_train 共に shape=(n_init, 1)の2次元の配列にしておく
    # 具体的には, 1次元の配列 shape=(n_init, ) では縦ベクトルや横ベクトルなどが扱えない
    # x=np.zeros(shape=(3, 1)) とすると x.shape=(3, 1), x.T.shape=(1, 3) となる (.T は転置)
    # y=np.zeros(shape=(3,)) だと y.shape=(3), y.T.shape=(3) となってしまう
    x_train = x[train_i].reshape(n_train, 1)
    f_train = f_true[train_i].reshape(n_train, 1)
    # 観測値は真の関数値に正規ノイズが加わって得られる
    y_train = f_train + rng.normal(scale=true_noise_scale, size=(n_train, 1))

    # 課題2: x_trainやy_trainなどから事後分布を計算

    # カーネル行列などの計算 (課題1で作成したgauss_kernelを用いる)
    # k =   # [k(x,x_1), k(x,x_2), ..., k(x,x_t)] のベクトル
    # shape=(n_grid, n_train)
    k = gauss_kernel(x_train, x, ker_var=ker_var, length_scale=length_scale)
    # K =   # K+σI
    K = gauss_kernel(x_train, x_train, ker_var=ker_var, length_scale=length_scale) + \
        gp_noise_var * np.eye(y_train.size)

    K_inv = kernel_inv(K)  # (K+σI)の逆行列

    # 事後平均・分散の計算 (課題.pdf・Hint1の μ_t, σ_t)
    # σ_tの式そのまま当てはめると分散共分散行列が得られる.  (for文などを用いていると違うかも)
    # その対角成分が事後分散となる.

    # mean =   # mean.shape=(n_grid,1)
    mean = k @ K_inv @ y_train
    # cov =   # cov.shape=(n_grid, n_grid)
    cov = gauss_kernel(
        x,
        x,
        ker_var=ker_var,
        length_scale=length_scale) - k.T @ K_inv @ k
    # var = np.diag(cov)  # 共分散行列の対角成分だけ取り出す. var.shape=(n_grid,1)
    var = np.diag(cov)

    # プロット時は1次元の配列の方が扱いやすいのでmean, varを1次元配列に直しておく
    mean = mean.flatten()  # mean.shape=(n_grid, )
    var = var.flatten()  # var.shape=(n_grid, )
    std = np.sqrt(var)  # 事後標準偏差

    # plot
    fig, ax = plt.subplots()

    ax.plot(x, f_true, c='black', ls='--', lw=2, label='true')
    ax.plot(x, mean, c='tab:blue', ls='-', lw=2, label='predictive mean')

    # 95%信用区間のプロット. **正規分布では**この区間は[平均±1.96*標準偏差]で与えられる
    ax.fill_between(x.flatten(), mean + 1.96 * std, mean - 1.96 * std,
                    fc='tab:blue', alpha=0.2, label='95% credible interval')
    ax.scatter(
        x_train,
        y_train,
        marker='o',
        s=50,
        lw=1.3,
        c='gold',
        ec='black',
        zorder=4,
        label='observed')

    # タイトルやファイル名は適宜変更
    ax.set_title(
        f'ker_var={ker_var:.2f}, ls={length_scale:.2f}, noise_var={gp_noise_var:.2f}')
    ax.set_ylabel(r'$f(x)$')
    ax.set_ylim(-6, 4.7)
    ax.legend(
        ncol=2,
        fontsize=11,
        framealpha=0.4,
        loc='lower left',
        borderaxespad=0.1)
    ax.set_xlabel(r'$x$')
    fig.tight_layout()
    fig.show()
    fig.savefig(
        fig_dir /
        f'ker_var{ker_var:.2f}_ls{length_scale:.2f}_noise_var{gp_noise_var:.2f}.pdf')
    plt.close(fig)
    return


def main():
    seed = 0
    # 実行例
    #plot_gp(ker_var=2, length_scale=0.5, gp_noise_var=1e-2, seed=seed)
    import functools as ft
    default_plot = ft.partial(plot_gp,
                              ker_var=2,
                              length_scale=0.5,
                              gp_noise_var=0.01,
                              seed=seed)

    ker_vars = [0.2, 2, 4]
    length_scales = [0.2, 0.5, 2]
    gp_noise_vars = [0.01, 0.1, 1]

    for kv in ker_vars:
        default_plot(ker_var=kv)
    for ls in length_scales:
        default_plot(length_scale=ls)
    for nv in gp_noise_vars:
        default_plot(gp_noise_var=nv)
    return


if __name__ == '__main__':
    project_dir = Path(__file__).parent.resolve()  # このファイルが存在するディレクトリ

    # project_dir 下にプロットを保存するようのディレクトリを作成
    fig_dir = project_dir / 'figure'
    fig_dir.mkdir(exist_ok=True, parents=True)

    plt.rcParams['font.size'] = 14  # 発表用のプロットの場合フォントは大きめに
    main()
