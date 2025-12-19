import numpy as np
import matplotlib.pyplot as plt

# Global variables
Ts = 0.001
Amax = 100
Vmax = 1


def sat(In, SatLevel=1.0):
    if In > SatLevel:
        return SatLevel
    elif In < -SatLevel:
        return -SatLevel
    else:
        return In


def SecondOrderFilter2(r, dr, x, dx):
    global Ts, Vmax, Amax

    e = (x - r) / Amax
    de = (dx - dr) / Amax

    zd = de / Ts
    z = (e / Ts + de / 2) / Ts

    m = np.floor((1 + np.sqrt(1 + 8 * abs(z))) / 2)
    if m < 1:
        m = 1  # sicurezza numerica

    sigma = zd + z / m + (m - 1) / 2 * np.sign(z)

    ua = -Amax * sat(sigma)

    uV = (Vmax - dx) / Ts
    uv = (-Vmax - dx) / Ts

    u = min(Amax, max(-Amax, min(ua, uV, Amax)))
    u = max(u, uv)

    return u


# =========================
# Main
# =========================

t = np.arange(0, 10 + Ts, Ts)
r = 0.5 * np.sin(3 * t)   # target
dr = 0.0

x0 = 0.0
xtm1 = x0
dxtm1 = 0.0

X = []

for i in range(len(t)):
    u = SecondOrderFilter2(r[i], dr, xtm1, dxtm1)

    dxt = dxtm1 + Ts * u
    xt = xtm1 + Ts / 2 * (dxt + dxtm1)

    xtm1 = xt
    dxtm1 = dxt

    X.append(xt)

X = np.array(X)

# Plot
plt.figure()
plt.plot(t, r, 'r--', label='r (target)')
plt.plot(t, X, 'b', label='x')
plt.legend()
plt.grid(True)
plt.xlabel('Time [s]')
plt.ylabel('Value')
plt.title('Second Order Filter')
plt.show()
