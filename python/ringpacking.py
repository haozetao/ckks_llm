from sage.all import *
from random import *

N = 32
N1 = 8
q = 65537

R1.<x> = PolynomialRing(Zmod(q))
R1 = R1.quo(x^N + 1)

R2.<x> = PolynomialRing(Zmod(q))
R2 = R2.quo(x^N1 + 1)


def ringDown(poly_big_ring, N, N1):
    k = N//N1
    poly_res = [0] * k
    for i in range(k):
        poly_res[i] = [0] * N1
    for global_idx in range(N):
        # poly_res[global_idx // N1][global_idx % N1] = poly_big_ring[global_idx]

        poly_res[global_idx//N1][global_idx%N1] = poly_big_ring[global_idx // N1 + global_idx % N1 * k]
    # for i in range(N//N1):
    #     poly_res.append([])
    #     for j in range(N1):
    #         poly_res[i].append(poly_big_ring[j*k + i])
# global_idx / N1 + global_idx % N1 * k
    return [R2(i) for i in poly_res]

def ringUp(poly_sub_ring, N, N1):
    poly_res = []
    for i in range(N1):
        for j in range(N//N1):
            poly_res.append(poly_sub_ring[j][i])
    return R1(poly_res)

def ring_extract(poly_big_ring, N, N1, i):
    return ringDown(poly_big_ring, N, N1)[i]

def twist_vector(sub_rings):
    temp = [0] * (N//N1)
    k = N//N1
    temp[0] = sub_rings[0]
    for i in range(1, k):
        temp[i] = sub_rings[N//N1 - i] * R2(x)^-1
    return temp


def twist_inv_vector(sub_rings):
    temp = [0] * (N//N1)
    k = N//N1
    temp[0] = sub_rings[0]
    for i in range(1, k):
        this_poly = sub_rings[k - i]
        temp[i] = this_poly * R2(x)^1

        # poly = [0] * N1
        # poly[0] = q - this_poly[N1-1]
        # for idx in range(1, N1):
        #     poly[idx] = this_poly[idx - 1]

        # temp[i] = R2(poly)
        # print(temp[i])
        # print(this_poly * R2(x)^1)
    return temp


def twist(poly_big_ring):
    sub_rings = ringDown(poly_big_ring, N, N1)
    temp = [0] * (N//N1)
    k = N//N1
    temp[0] = sub_rings[0]
    for i in range(1, k):
        temp[i] = sub_rings[N//N1 - i] * R2(x)^-1
    print(temp)
    return ringUp(temp, N, N1)

def twist_inv(poly_big_ring):
    sub_rings = ringDown(poly_big_ring, N, N1)
    temp = [0] * (N//N1)
    k = N//N1
    temp[0] = sub_rings[0]
    for i in range(1, k):
        temp[i] = sub_rings[N//N1 - i] * R2(x)^1
    print(temp)
    return ringUp(temp, N, N1)

def inner_product(poly1_sub_ring, poly2_sub_ring, k):
    temp = 0
    for i in range(k):
        temp += poly1_sub_ring[i] * poly2_sub_ring[i]
        print(poly1_sub_ring[i] * poly2_sub_ring[i])
    return temp

def rlwe_enc(mx_big_ring, sx_big_ring):
    ex = R1([int(gauss(0, 3.2)) for i in range(N)])
    ax = R1([randint(0, q) for i in range(N)])
    scaled_mx = R1([scaler * i for i in mx_big_ring])
    return (ax, -ax*sx_big_ring + scaled_mx + ex)

def mlwe_enc(mx_small_ring, sx_small_ring):
    ex = R2([int(gauss(0, 3.2)) for i in range(N1)])
    ax = []
    scaled_mx = R2([scaler * i for i in mx_small_ring])

    bx = R2(ex + scaled_mx)
    for i in range(N//N1):
        ax.append(R2([randint(0, q) for i in range(N1)]))
        bx -= sx_small_ring[i] * ax[i]
    return (ax, bx)

def rlwe_dec(cipher_rlwe, sx_big_ring):
    mx = cipher_rlwe[0]*sx_big_ring + cipher_rlwe[1]
    output = [0]*N
    for i in range(N):
        output[i] = mx[i] if int(mx[i]) < q//2 else int(mx[i]) - q

    return R1([round(int(output[i]) / scaler) for i in range(N)])

def mlwe_dec(cipher_mlwe, sx_small_ring):
    mx = cipher_mlwe[1]
    for i in range(N//N1):
        mx += cipher_mlwe[0][i] * sx_small_ring[i]

    output = [0] * N1
    for i in range(N1):
        output[i] = mx[i] if int(mx[i]) < q//2 else int(mx[i]) - q
    return R2([round(int(output[i]) / scaler) for i in range(N1)])


def homo_ring_up(poly_small_ring):
    poly_big_ring = [0] * N
    k = N//N1
    for i in range(N1):
        poly_big_ring[i*k] = poly_small_ring[i]
    return R1(poly_big_ring)

# 1 mlwe -> 1 rlwe
def Embeded(mlwe_cipher):
    bx = homo_ring_up(mlwe_cipher[1])
    ax = ringUp(twist_vector(mlwe_cipher[0]), N, N1)
    return (ax, bx)
