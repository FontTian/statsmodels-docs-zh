"""
具有 AR 误差的广义最小二乘

GLSAR 使用人工数据的 6 个示例 
"""

# .. 注意：这些示例主要用于交叉检验结果。它是仍在编写中，并且GLSAR仍在开发中。


import numpy as np
import numpy.testing as npt
from scipy import signal
import statsmodels.api as sm
from statsmodels.regression.linear_model import GLSAR, yule_walker

examples_all = range(10) + ['test_copy']

examples = examples_all  # [5]

if 0 in examples:
    print('\n Example 0')
    X = np.arange(1, 8)
    X = sm.add_constant(X, prepend=False)
    Y = np.array((1, 3, 4, 5, 8, 10, 9))
    rho = 2
    model = GLSAR(Y, X, 2)
    for i in range(6):
        results = model.fit()
        print('AR coefficients:', model.rho)
        rho, sigma = yule_walker(results.resid, order=model.order)
        model = GLSAR(Y, X, rho)

    par0 = results.params
    print('params fit', par0)
    model0if = GLSAR(Y, X, 2)
    res = model0if.iterative_fit(6)
    print('iterativefit beta', res.params)
    results.tvalues   # TODO: 这是正确的吗？它确实等于params / bse
    # 但与AR示例不同（这是错误的）
    print(results.t_test([0, 1]))  # sd 和 t 正确吗? vs
    print(results.f_test(np.eye(2)))


rhotrue = np.array([0.5, 0.2])
nlags = np.size(rhotrue)
beta = np.array([0.1, 2])
noiseratio = 0.5
nsample = 2000
x = np.arange(nsample)
X1 = sm.add_constant(x, prepend=False)

wnoise = noiseratio * np.random.randn(nsample + nlags)
# .. noise = noise[1:] + rhotrue*noise[:-1] # 错，这不是 AR

# .. 查找有关单变量 ARMA 函数的草稿
# generate AR(p)
if np.size(rhotrue) == 1:
    # 替换为 scipy.signal.lfilter, 继续测试
    arnoise = np.zeros(nsample + 1)
    for i in range(1, nsample + 1):
        arnoise[i] = rhotrue * arnoise[i - 1] + wnoise[i]
    noise = arnoise[1:]
    an = signal.lfilter([1], np.hstack((1, -rhotrue)), wnoise[1:])
    print('simulate AR(1) difference', np.max(np.abs(noise - an)))
else:
    noise = signal.lfilter([1], np.hstack((1, -rhotrue)), wnoise)[nlags:]

# 生成带有 AR 噪声的 GLS 模型
y1 = np.dot(X1, beta) + noise

if 1 in examples:
    print('\nExample 1: iterative_fit and repeated calls')
    mod1 = GLSAR(y1, X1, 1)
    res = mod1.iterative_fit()
    print(res.params)
    print(mod1.rho)
    mod1 = GLSAR(y1, X1, 2)
    for i in range(5):
        res1 = mod1.iterative_fit(2)
        print(mod1.rho)
        print(res1.params)

if 2 in examples:
    print('\nExample 2: iterative fitting of first model')
    print('with AR(0)', par0)
    parold = par0
    mod0 = GLSAR(Y, X, 1)
    for i in range(5):
        res0 = mod0.iterative_fit(1)
        print('rho', mod0.rho)
        parnew = res0.params
        print('params', parnew,)
        print('params change in iteration', parnew - parold)
        parold = parnew

# generate pure AR(p) process
Y = noise

# 没有回归变量的示例，结果现在直接具有与 yule-walker 相同的估计 rho

if 3 in examples:
    print('\nExample 3: pure AR(2), GLSAR versus Yule_Walker')
    model3 = GLSAR(Y, rho=2)
    for i in range(5):
        results = model3.fit()
        print("AR coefficients:", model3.rho, results.params)
        rho, sigma = yule_walker(results.resid, order=model3.order)
        model3 = GLSAR(Y, rho=rho)

if 'test_copy' in examples:
    xx = X.copy()
    rhoyw, sigmayw = yule_walker(xx[:, 0], order=2)
    print(rhoyw, sigmayw)
    print((xx == X).all())  # 测试没有变化的数组 (固定的)

    yy = Y.copy()
    rhoyw, sigmayw = yule_walker(yy, order=2)
    print(rhoyw, sigmayw)
    print((yy == Y).all())  # 测试没有变化的数组 (固定的)


if 4 in examples:
    print('\nExample 4: demeaned pure AR(2), GLSAR versus Yule_Walker')
    Ydemeaned = Y - Y.mean()
    model4 = GLSAR(Ydemeaned, rho=2)
    for i in range(5):
        results = model4.fit()
        print("AR coefficients:", model3.rho, results.params)
        rho, sigma = yule_walker(results.resid, order=model4.order)
        model4 = GLSAR(Ydemeaned, rho=rho)

if 5 in examples:
    print('\nExample 5: pure AR(2), GLSAR iterative_fit versus Yule_Walker')
    model3a = GLSAR(Y, rho=1)
    res3a = model3a.iterative_fit(5)
    print(res3a.params)
    print(model3a.rho)
    rhoyw, sigmayw = yule_walker(Y, order=1)
    print(rhoyw, sigmayw)
    npt.assert_array_almost_equal(model3a.rho, rhoyw, 15)
    for i in range(6):
        model3b = GLSAR(Y, rho=0.1)
        print(i, model3b.iterative_fit(i).params, model3b.rho)

    model3b = GLSAR(Y, rho=0.1)
    for i in range(6):
        print(i, model3b.iterative_fit(2).params, model3b.rho)


print(np.array(res.history['params']))
print(np.array(res.history['rho']))
