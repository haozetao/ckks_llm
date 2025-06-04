# 256*128*256列主
import numpy as np


# matrix_C = matrix_A * matrix_B
data_a = np.random.randint(0, 16, size=(32768,), dtype=int)
# print(data_a)
data_b = np.random.randint(0, 16, size=(32768,), dtype=int)
matrix_A = data_a.reshape(128,256)
matrix_B = data_b.reshape(128,256)
matrix_C = matrix_A.T @ matrix_B
# print(matrix_C)
tmp_data_a = data_a.copy()

# mul
mul_result = []
for i in range(256):
    mul_result.append(np.multiply(tmp_data_a,data_b))
    # 按行循环左移一位
    tmp_data_a_copy = tmp_data_a.copy()
    shift_amount = 1
    tmp_data_a = np.concatenate((tmp_data_a[shift_amount:], tmp_data_a[:shift_amount]))
    for i in range(128):
        tmp_data_a[i*256+255] = 0

    # 处理第一列
    shift_amount = 255
    for i in range(32768):
        if (i%256 != 0 ):
            tmp_data_a_copy[i]=0
    tmp_data_a_copy = np.concatenate((tmp_data_a_copy[-shift_amount:], tmp_data_a_copy[:-shift_amount]))
    tmp_data_a = tmp_data_a + tmp_data_a_copy

# reduce sum
reduce_sum_result = []
for i in range(256):
    shift_amount = 32768 >> 1
    mul_result_data = mul_result[i]
    while (shift_amount>=256):
        rotate_mul_result_data = np.concatenate((mul_result_data[shift_amount:], mul_result_data[:shift_amount]))
        mul_result_data = rotate_mul_result_data + mul_result_data
        shift_amount = shift_amount >> 1
    result_sum_result_data = np.zeros(shape=(32768), dtype=int)
    for j in range(256):
        result_sum_result_data[j] = mul_result_data[j]
    reduce_sum_result.append(result_sum_result_data)

# repeat
repeat_result = []
for i in range(256):
    shift_amount = 256
    repeat_result_data = reduce_sum_result[i]
    while (shift_amount < 32768):
        rotate_repeat_result_data = np.concatenate((repeat_result_data[-shift_amount:], repeat_result_data[:-shift_amount]))
        repeat_result_data = repeat_result_data + rotate_repeat_result_data
        shift_amount = shift_amount << 1
    repeat_result.append(repeat_result_data)
# mask
'''
上下存储
'''
C1_mask_result = []
C2_mask_result = []
for i in range(256):
    # deal with reserve position
    reserve_position_C1_list = []
    reserve_position_C2_list = []
    for k in range(256):
        row = (i+k)%256
        if (row <128 ):
            reserve_position_C1_list.append((row,k))
        else:
            reserve_position_C2_list.append((row-128,k))
    # get mask result
    C1_mask_result_data = np.zeros(shape = (32768), dtype = int)
    C2_mask_result_data = np.zeros(shape = (32768), dtype = int)
    # print(reserve_position_C1_list)
    for m in reserve_position_C1_list:
        row = m[0]
        column = m[1]
        C1_mask_result_data[256*row + column] = repeat_result[i][256*row + column]
    C1_mask_result.append(C1_mask_result_data)
    # print(C1_mask_result_data.tolist())
    for m in reserve_position_C2_list:
        row = m[0]
        column = m[1]
        C2_mask_result_data[256*row + column] = repeat_result[i][256*row + column]
    C2_mask_result.append(C2_mask_result_data)
# sum
C1 = np.zeros(shape = (32768), dtype = int)
C2 = np.zeros(shape = (32768), dtype = int)
#print(123)
#print(C1_mask_result[1][:256])
#print(C1_mask_result[1][256:512])

for i in range(256):
    C1 = C1 + C1_mask_result[i]
    C2 = C2 + C2_mask_result[i]

print(C1.reshape(128,256))
print(C2.reshape(128,256))
print(matrix_C)
print(matrix_C.shape)
print(matrix_C[:128, :])
print("C1 == matrix_C[:128, :] ?", np.array_equal(C1.reshape(128,256), matrix_C[:128, :]))
print("C2 == matrix_C[128:, :] ?", np.array_equal(C2.reshape(128,256), matrix_C[128:, :]))


# 256*256*128左边为行主，右边为列主
data_a1 = C1.copy()
data_a2 = C2.copy()

tmp_data_a1 = data_a1.copy()
tmp_data_a2 = data_a2.copy()

# mul
mul_result = []
for i in range(128):
    shift_amount = 256
    mul_result.append(np.multiply(tmp_data_a1,data_b))
    tmp_data_a1 = np.concatenate((tmp_data_a1[shift_amount:], tmp_data_a1[:shift_amount]))

for i in range(128):
    shift_amount = 256
    mul_result.append(np.multiply(tmp_data_a2,data_b))
    tmp_data_a2 = np.concatenate((tmp_data_a2[shift_amount:], tmp_data_a2[:shift_amount]))

# reduce sum
reduce_sum_result = []
for i in range(256):
    shift_amount = 256 >> 1
    mul_result_data = mul_result[i]
    while (shift_amount>0):
        rotate_mul_result_data = np.concatenate((mul_result_data[shift_amount:], mul_result_data[:shift_amount]))
        mul_result_data = rotate_mul_result_data + mul_result_data
        shift_amount = shift_amount >> 1
    result_sum_result_data = np.zeros(shape=(32768), dtype=int)
    for j in range(128):
        result_sum_result_data[256*j] = mul_result_data[256*j]
    reduce_sum_result.append(result_sum_result_data)

# repeat
repeat_result = []
for i in range(256):
    shift_amount = 1
    repeat_result_data = reduce_sum_result[i]
    while (shift_amount < 256):
        rotate_repeat_result_data = np.concatenate((repeat_result_data[-shift_amount:], repeat_result_data[:-shift_amount]))
        repeat_result_data = repeat_result_data + rotate_repeat_result_data
        shift_amount = shift_amount << 1
    repeat_result.append(repeat_result_data)
# mask
'''
上下存储
'''
C1_mask_result = []
for i in range(256):
    # deal with reserve position
    reserve_position_C1_list = []
    for k in range(128):
        column = (i+k)%128
        if (i>=128):
            column += 128
        reserve_position_C1_list.append((k,column))
    # get mask result
    C1_mask_result_data = np.zeros(shape = (32768), dtype = int)
    # print(reserve_position_C1_list)
    for m in reserve_position_C1_list:
        row = m[0]
        column = m[1]
        C1_mask_result_data[256*row + column] = repeat_result[i][256*row + column]
    C1_mask_result.append(C1_mask_result_data)
    # print(C1_mask_result_data.tolist())
# sum
C1 = np.zeros(shape = (32768), dtype = int)
C2 = np.zeros(shape = (32768), dtype = int)
#print(123)
#print(C1_mask_result[1][:256])
#print(C1_mask_result[1][256:512])

for i in range(256):
    C1 = C1 + C1_mask_result[i]

m1 = C1.reshape(128,256)
m2 = matrix_C@data_b.reshape(128,256).T
print(m1)
print(m2)
flag = 0
for i in range(128):
    for j in range(256):
        if (m1[i][j]!=m2[j][i]):
            flag+=1
print(flag)


'''
# 256*128*256行主
import numpy as np


# matrix_C = matrix_A * matrix_B
data_a = np.random.randint(0, 64, size=(32768,), dtype=int)
# print(data_a)
data_b = np.random.randint(0, 64, size=(32768,), dtype=int)
matrix_A = data_a.reshape(256,128)
matrix_B = data_b.reshape(256,128)
# print(matrix_A.shape)
# print(matrix_B.T.shape)
matrix_C = matrix_A @ matrix_B.T
# print(matrix_C)

# mul
# n rot -- 256
mul_result = []
for i in range(256):
    shift_amount = 128
    mul_result.append(np.multiply(data_a,data_b))
    data_b = np.concatenate((data_b[shift_amount:], data_b[:shift_amount]))
for i in mul_result:
    # print(i)
    pass

# reduce sum
# log(d)*n rot -- log(128)*256
reduce_sum_result = []
for i in range(256):
    shift_amount = 128 >> 1
    mul_result_data = mul_result[i]
    while (shift_amount>0):
        rotate_mul_result_data = np.concatenate((mul_result_data[shift_amount:], mul_result_data[:shift_amount]))
        mul_result_data = rotate_mul_result_data + mul_result_data
        shift_amount = shift_amount >> 1
    result_sum_result_data = np.zeros(shape=(32768), dtype=int)
    for j in range(256):
        result_sum_result_data[128*j] = mul_result_data[128*j]
    reduce_sum_result.append(result_sum_result_data)

# repeat
# log(d)*n rot -- log(128)*256
repeat_result = []
for i in range(256):
    shift_amount = 1
    repeat_result_data = reduce_sum_result[i]
    while (shift_amount < 128):
        rotate_repeat_result_data = np.concatenate((repeat_result_data[-shift_amount:], repeat_result_data[:-shift_amount]))
        repeat_result_data = repeat_result_data + rotate_repeat_result_data
        shift_amount = shift_amount << 1
    repeat_result.append(repeat_result_data)

print(repeat_result[0][:256])
# mask

# 左右存储
# A*B结果记为C,是256*256的矩阵
# 我们用两个32768的密文存储C的结果,可以看做两个256*128的矩阵,分别记为C1,C2
# repeat得到了256个32768的密文,记为C_i,i \in 256
# C_i[j,k] = C[j, (j+i)%256]
# C1[m,n] = C[m,n]
# C2[m,n] = C[m,n+128]

# C_i[j,(j+i)%256]保留,对应{
# C1[j,(j+i)%256] ((j+i)%256 < 128) 
# C2[j,(j+i)%256128] ((j+i)%256 >= 128)
# }
C1_mask_result = []
C2_mask_result = []
for i in range(256):
    # deal with reserve position
    reserve_position_C1_list = []
    reserve_position_C2_list = []
    for j in range(256):
        column = (i+j)%256
        if (column <128 ):
            reserve_position_C1_list.append((j,column))
        else:
            reserve_position_C2_list.append((j,column-128))
    
    # get mask result
    C1_mask_result_data = np.zeros(shape = (32768), dtype = int)
    C2_mask_result_data = np.zeros(shape = (32768), dtype = int)
    for m in reserve_position_C1_list:
        row = m[0]
        column = m[1]
        C1_mask_result_data[128*row + column] = repeat_result[i][128*row + column]
    C1_mask_result.append(C1_mask_result_data)
    # print(C1_mask_result_data.tolist())
    for m in reserve_position_C2_list:
        row = m[0]
        column = m[1]
        C2_mask_result_data[128*row + column] = repeat_result[i][128*row + column]
    C2_mask_result.append(C2_mask_result_data)



# sum
C1 = np.zeros(shape = (32768), dtype = int)
C2 = np.zeros(shape = (32768), dtype = int)
for i in range(256):
    C1 = C1 + C1_mask_result[i]
    C2 = C2 + C2_mask_result[i]

print(C1.reshape(256,128))
print(C2.reshape(256,128))
print(matrix_C)
print("C1 == matrix_C[:, :128] ?", np.array_equal(C1.reshape(256,128), matrix_C[:, :128]))
print("C2 == matrix_C[:, 128:] ?", np.array_equal(C2.reshape(256,128), matrix_C[:, 128:]))'''