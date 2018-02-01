#帮助文档  info(XXX)
from numpy  import *

初始化
np.arange(15).reshape(3, 5)  #等同于 .resize((3,5))
np.linspace(2.0, 3.0, num=5)
a.size  
a.shape
a.dtype   
b = array([(1.5,2,3),(4,5,6)]) #默认创建的数组类型(dtype)都是float64
np.zeros((3,4))  #Return a new array of given shape and type, filled with zeros
np.zeros_like(x)  #Return an array of zeros with the same shape and type as a given array.
np.ones((2,3,4),dtype=int16)                # dtype can also be specified
np.empty((2,3))  #Return a new array of given shape and type, without initializing entries.  #random
np.random.rand(3,2)  #Return a matrix of random values with given shape.
np.random.randn(3,2) #Return a sample (or samples) from the “standard normal” distribution.
np.random.random(3,2)     #Return random floats in the half-open interval [0.0, 1.0).
x = np.eye(3)   np.nonzero(x)  #单位矩阵

def f(x,y):
    return 10*x+y
b = fromfunction(f,(5,4),dtype=int)

a[ : :-1]       # reversed a

逻辑
all：统计数组/数组某一维度中是否都是True
np.all([-1, 4, 5])
np.all([[True,False],[True,True]], axis=0)

any,统计数组/数组某一维度中是否存在True
np.any([[True, False], [False, False]], axis=0)
np.any([-1, 0, 5])

apply_along_axis,Apply a function to 1-D slices along the given axis.
b = np.array([[8,1,7], [4,3,9], [5,2,6]])
def my_func(a):
     """Average first and last element of a 1-D array"""
     return (a[0] + a[-1]) * 0.5
b = np.array([[1,2,3], [4,5,6], [7,8,9]])
np.apply_along_axis(my_func, 0, b)

argmax :Returns the indices of the maximum values along an axis.
a = np.arange(6).reshape(2,3)
np.argmax(a)
np.argmax(a, axis=0)
np.argmax(a, axis=1)
argmin 类似 

sort 
a = np.array([[1,4],[3,1]])
np.sort(a)  
np.sort(a, axis=None) 
np.sort(a, axis=0)

argsort Returns the indices that would sort an array.
x = np.array([3, 1, 2])
np.argsort(x)

lexsort类似于argsort 但是对于元组内相同的元素，会根据第二个向量进行二次排序
a = [1,5,1,4,3,4,4]
b = [9,4,0,4,0,2,1]
ind = np.lexsort((b,a))  # Sort by a, then by b   http://blog.csdn.net/lyrassongs/article/details/78814615

average, Compute the weighted average along the specified axis
data = np.arange(6).reshape((3,2))
np.average(data, axis=1, weights=[1./4, 3./4])

bincount, Count number of occurrences of each value in array of non-negative ints
有bin(根据最大数值确定bin：多少个)然后count
np.bincount(np.array([0, 1, 1, 3, 2, 1, 7]))

w = np.array([0.3, 0.5, 0.2, 0.7, 1., -0.6]) 
x = np.array([0, 1, 1, 2, 2, 2])
np.bincount(x,  weights=w)

clip 例如给定一个区间[0,1]，则小于0的将变成0，大于1则变成1
a = np.arange(10)
np.clip(a, 1, 8)

corrcoef 相关系数矩阵
np.corrcoef(a, b)

cov cov函数来计算协方差
np.cov(a, b)

cross  两个向量的叉积（外积）  是一个向量  
np.cross(x, y)

inner 两个向量的点积，是一个数值
np.inner(a, b)

outer 跟cross不一样 outer是a的第一个元素跟b的每一个元素相乘作为第一行，第二个元素跟b的每一个元素相乘作为第二个元素
np.outer(a, b) #a 1*m, b 1*n 结果为m*n

a.cumprod()累计求积 
a.cumsum()累计求和 

diff diff函数就是执行的是后一个元素减去前一个元素
x = np.array([[1, 3, 6, 10], [0, 5, 6, 8]])
np.diff(x)
np.diff(x, axis=0)

max与maximum区别  min, minimum同理
max([2, 3, 4], [1, 5, 2])
maximum([2, 3, 4], [1, 5, 2])

prod 计算数组元素的连乘积
np.prod([1.,2.])

around   四舍五入
np.around(80.23456, 2) : 80.23 

std标准差
np.std(a)

var 方差
np.var(a)

sum

trace 用trace计算方阵的迹 主对角线（从左上方至右下方的对角线）上各个元素的总和
np.trace(np.eye(3))

transpose 转置 
vdot  返回两向量的点积

挺实用
vectorize  对numpy数组中的每一个元素应用一个函数
count_fun = np.vectorize(lambda x: len(x))
count_fun(a)

allclose或者array_equal  判断是否相等
np.array_equal([1, 2], [1, 2])

flatten 展开成一维的
a = np.array([[1,2], [3,4]])
a.flatten()

a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
np.floor(a)  #向下取整
np.ceil(a)  #向上取整
np.ravel(a) #Returns a 1D version of self, as a view. 平铺
a.transpose()  #转置

合并 还没有总结好
stack() 通过axis=0或1，对value进行堆积 主要用于list
a = [1, 2, 3, 4]
b = [5, 6, 7, 8]
np.stack((a, b),axis=0) 
np.stack((a,b),axis=1)

hstack()按水平方向堆叠数组  vstack()按垂直方向堆叠数组、 主要用于array数组
a = np.array([[1],[2],[3]])
b = np.array([[2],[3],[4]])
np.hstack((a,b))
np.vstack((a,b))

column_stack()
a = np.array([[1],[2],[3]])
b = np.array([[2],[3],[4]])
np.column_stack((a, b))

row_stack()
a = np.array([[1],[2],[3]])
b = np.array([[2],[3],[4]])
np.row_stack((a, b))

concatenate
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
np.concatenate((a, b), axis=0)
np.concatenate((a, b.T), axis=1)

c_

r_


分割
hsplit(a,3)   # Split a into 3
vsplit(a,3)   #换一个方向分割

创建数组
arange, array, copy, empty, empty_like, eye, fromfile, fromfunction, identity, 
linspace, logspace, mgrid, ogrid, ones, ones_like, r , zeros, zeros_like ,pad,diag，

tile
tile(A,(m,n)) 将数组A作为元素构造出m行n列的数组

dict.get(key,x)
从字典中获取key对应的value


转化
astype, atleast 1d, atleast 2d, atleast 3d, mat

操作
array split, , diagonal, dsplit, dstack, hsplit, , item, 
newaxis, ravel, repeat, reshape, resize, squeeze, swapaxes, take, transpose, vsplit, vstack
random.choice   meshgrid

询问
all, any,  where
np.nonzero()  #非0元素的位置索引  只有a中非零元素才会有索引值，那些零值元素没有索引值
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
np.nonzero(a > 3)

排序
argmax, argmin, argsort, max, min, ptp, searchsorted, sort ,copysign

运算
A*B 与 dot(A,B) 区别
np.exp(B) 
sqrt(B) 
choose, compress, cumprod, cumsum, inner, fill, imag, prod, put, putmask, real, sum  np.power(x,3)

基本线性代数 linalg

svd

inv 矩阵的逆

多项式poly1d很有用
一、多项式表示
f(x)= anxn+an-1xn-1 +…+a2x2+a1x +a0 多项式表示
f(x) = x3 -2x + 1表示方法：a[0]是最高次的系数，a[-1]是常数项，注意x2的系数为0
a= np.array([1.0, 0, -2, 1])
f = np.poly1d(a)
使用：f(np.linspace(0,1, 5))

二、多项式计算
f + [-2, 1] # 和 f + np.polyld([-2, 1])相同 poly1d([ 1., 0., -4., 2.])
f*f #两个3次多项式相乘得到一个6次多项式  polyld([ 1., 0., -4., 2., 4., -4.,1.])

三、多项式微分和积分
f.deriv() 微分
f.integ() 积分

四、多项式的根 多项式等于0的解
r = np.roots(f)
np.poly(r) #有了根，可以得到多项式的系数

五、使用多项式去拟合其他的函数
x = np.linspace(-np.pi/2, np.pi/2, 1000) 
y = np.sin(x) 
a = np.polyfit(x, y, deg) #deg是维数，如果deg=3，那么拟合出来的多项式就是3阶
#索引
np.unravel_index(100,(6,7,8))   #考虑一个 (6,7,8) 形状的数组，其第100个元素的索引(x,y,z)是什么



对一个5x5的随机矩阵做归一化
Z = np.random.random((5,5))
Zmax, Zmin = Z.max(), Z.min()
Z = (Z - Zmin)/(Zmax - Zmin)

给定一个一维数组，对其在3到8之间的所有元素取反
Z = np.arange(11)
Z[(3 < Z) & (Z <= 8)] *= -1

找到两个数组中的共同元素?
np.intersect1d(Z1,Z2)


日期 
得到昨天，今天，明天的日期?
yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today     = np.datetime64('today', 'D')
tomorrow  = np.datetime64('today', 'D') + np.timedelta64(1, 'D')

得到所有与2016年7月对应的日期
Z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')

将向量中最大值替换为1 
Z[Z.argmax()] = 0

给定两个数组X和Y，构造Cauchy矩阵C (Cij =1/(xi - yj))
X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)

打印一个数组中的所有数值?
np.set_printoptions(threshold=np.nan)

对于numpy数组，enumerate的等价操作是什么？
Z = np.arange(9).reshape(3,3)
for index, value in np.ndenumerate(Z):
     print (index, value)


减去一个矩阵中的每一行的平均值 
X = np.random.rand(5, 10)
Y = X - X.mean(axis=1, keepdims=True)
或者
Y = X - X.mean(axis=1).reshape(-1, 1)  #如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值
