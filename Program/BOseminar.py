from pathlib import Path    # ファイルやディレクトリのパスを扱いやすくするためのライブラリ

import matplotlib.pyplot as plt  # データのプロット
import numpy as np  # 高度な数値計算を行うためのAPIを提供
# matplotlibプロットをpdfファイルに保存するためのクラス
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec  # 複数のプロットをグリッド状に配置するためのクラス
from scipy.linalg import cho_solve, cho_factor  # 行列のCholesky分解を行う関数
from scipy.stats import norm  # 正規分布に関する計算を行うクラス


# 想定する真のblack-box関数
def f(x):
    return np.cos(2 * x) + np.sin(3 * x) - 0.5 * x


# 課題1: ガウスカーネルの定義 (課題.pdfに載っているカーネル関数)
# n個の入力x1, m個の入力x2にに対し, カーネル行列を返す関数
# shape=(n,m)のカーネル行列は(i,j)番目の要素に k(x1_i, x2_j) を持つ
def gauss_kernel(x1, x2):
    # x1.shape = (n, 1)
    # x2.shape = (m, 1)

    # ヒント: (i,j)番目の要素に (x1_i -x2_j) を持つようなn×m行列を作成してからカーネル行列を計算すると良い
    return 2 * np.exp(-np.square(x1 - x2.flatten()) / 0.5)


# (定数倍の単位行列が加わった)カーネル行列の逆行列(K+σI)^{-1} を計算する関数
# コレスキー分解を用いることで高速&数値的に安定した逆行列計算を行っている
def kernel_inv(k_matrix):
    k_dim = k_matrix.shape[0]
    return cho_solve((cho_factor(k_matrix)), np.eye(k_dim))


# ベイズ最適化のループを回す関数
# methodは獲得関数名, seedは乱数のシード値を与える引数
def bo(method: str, seed: int = 0):
    print(f'{method=}, {seed=}')
    # 実験条件に関するパラメータ
    x_domain = (-5, 5)  # 入力xの範囲[-5, 5]
    n_grid = 1000  # グリッド数 xの範囲をn_grid分割した時，それぞれの区間の大きさを表す
    n_init = 1  # 初期点の数
    noise_scale = 1e-2  # 正規ノイズの標準偏差
    noise_var = noise_scale ** 2  # 分散
    max_iteration = 50  # 最大イテレーション数

    # 初期化
    rng = np.random.default_rng(seed)  # 乱数生成器
    x = np.linspace(x_domain[0], x_domain[1], n_grid).reshape(
        n_grid, 1)  # x_domainをグリッドで区切ったもの. x.shape=(n_grid, 1)
    f_true = f(x)  # 各xにおける真の関数値. f_true.shape=(n_grid, 1)

    init_i = rng.choice(
        np.arange(n_grid),
        size=n_init,
        replace=False)  # 初期点のindex

    # 観測データの作成 (最初は初期点のみ)

    # 計算の都合上, x_train, y_train 共に shape=(n_init, 1)の2次元の配列にしておく
    # 具体的には, 1次元の配列 shape=(n_init, ) では縦ベクトルや横ベクトルなどが扱えない
    # x=np.zeros(shape=(3, 1)) とすると x.shape=(3, 1), x.T.shape=(1, 3) となる (.T は転置)
    # y=np.zeros(shape=(3,)) だと y.shape=(3), y.T.shape=(3) となってしまう
    x_train = x[init_i].reshape(n_init, 1)
    f_train = f_true[init_i].reshape(n_init, 1)
    # 観測値は真の関数値に正規ノイズが加わって得られる
    y_train = f_train + rng.normal(scale=noise_scale, size=(n_init, 1))

    # 1つのpdfファイルにプロットを追加していく
    with PdfPages(fig_dir / f'{method}_seed{seed}.pdf', keep_empty=False) as pdf:
        for i in range(1, max_iteration + 1):
            # 課題2: x_trainやy_trainなどから事後分布を計算

            # カーネル行列などの計算 (課題1で作成したgauss_kernelを用いる)
            # [k(x,x_1), k(x,x_2), ..., k(x,x_t)] のベクトル, k.shape=(n_init, n_init)
            k = gauss_kernel(x_train, x)
            # K+σI, K.shape=(n_init, n_init)
            K = gauss_kernel(x_train, x_train) + \
                noise_var * np.eye(y_train.size)
            K_inv = kernel_inv(K)  # (K+σI)の逆行列

            # 事後平均・分散の計算 (課題.pdf・Hint1の μ_t, σ_t)
            # σ_tの式そのまま当てはめると分散共分散行列が得られる.  (for文などを用いていると違うかも)
            # その対角成分が事後分散となる.

            # mean.shape=(n_grid,1)
            mean = k.T @ K_inv @ y_train

            # cov.shape=(n_grid, n_grid)
            cov = gauss_kernel(x, x) - k.T @ K_inv @ k
            var = np.diag(cov)  # 共分散行列の対角成分だけ取り出す. var.shape=(n_grid,1)

            # プロット時は1次元の配列の方が扱いやすいのでmean, varを1次元配列に直しておく
            mean = mean.flatten()  # mean.shape=(n_grid, )
            var = var.flatten()  # var.shape=(n_grid, )
            std = np.sqrt(var)  # 事後標準偏差

            # 課題3: 獲得関数(acquisition function)の計算
            # 標準正規分布の確率密度関数や累積分布関数はそれぞれ norm.pdf(), norm.cdf()で計算できる
            current_best = np.max(y_train)  # カレントベスト(観測データ中の最大値). EIやPIの計算に用いる
            if method == 'US':  # uncertainty sampling
                af = var
            elif method == 'PI':  # probability of improvement
                af = norm.cdf((mean - current_best) / std)
            elif method == 'EI':  # expected improvement
                z = (mean - current_best) / std
                af = std * (norm.pdf(z) + z * norm.cdf(z))
            elif method == 'UCB':  # Gaussian process upper confidence bound
                beta = 3
                af = mean + beta * std
            else:
                import sys
                sys.exit(-1)
            next_i = np.argmax(af)  # 次の観測点は獲得関数の最大化点

            # plot
            fig = plt.figure()  # Figureオブジェクトの作成
            gs = GridSpec(4, 1, fig)  # figを分割
            ax = fig.add_subplot(gs[0:3, 0])  # 上側はモデルのプロット用
            ax_af = fig.add_subplot(gs[3, 0])  # 下側は獲得関数のプロット用

            ax.plot(x, f_true, c='black', ls='--', lw=2, label='true')
            ax.plot(
                x,
                mean,
                c='tab:blue',
                ls='-',
                lw=2,
                label='predictive mean')

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
            ax.axvline(x[next_i], ls=':', lw=1.5, c='tab:green', label='next')

            ax_af.plot(x, af, lw=2, c='tab:blue', label=method)
            ax_af.axvline(x[next_i], ls=':', lw=1.5, c='tab:green')

            ax.set_title(f'{method}: iteration {i}')
            ax.tick_params(labelbottom=False)
            ax.set_ylabel(r'$f(x)$')
            ax.set_ylim(-6, 4.7)
            ax.legend(
                ncol=2,
                fontsize=11,
                framealpha=0.4,
                loc='lower left',
                borderaxespad=0.1)
            ax_af.set_xlim(ax.get_xlim())
            ax_af.set_xlabel(r'$x$')
            ax_af.set_ylabel(r'$\alpha(x)$')
            ax_af.legend(fontsize=11, loc='upper left', borderaxespad=0.1)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # 観測データの追加
            next_x = x[next_i].reshape(1, -1)
            next_f = f_true[next_i]
            next_y = (next_f + rng.normal(scale=noise_scale)
                      ).reshape(1, -1)  # 観測値yは真の関数値fにノイズが加わって得られる
            x_train = np.vstack([x_train, next_x])
            y_train = np.vstack([y_train, next_y])
    return


def main():
    seed = 0
    methods = ['US', 'PI', 'EI', 'UCB']
    for method in methods:
        bo(method=method, seed=seed)
    return


if __name__ == '__main__':
    project_dir = Path(__file__).parent.resolve()  # このファイルが存在するディレクトリ

    # project_dir 下にプロットを保存するようのディレクトリを作成
    fig_dir = project_dir / 'figure'
    fig_dir.mkdir(exist_ok=True, parents=True)

    plt.rcParams['font.size'] = 14  # 発表用のプロットの場合フォントは大きめに
    main()
