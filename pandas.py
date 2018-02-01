
（1）
class Vehicle:
    def __init__(self, number_of_wheels, type_of_tank, seating_capacity, maximum_velocity):
        self.number_of_wheels = number_of_wheels
        self.type_of_tank = type_of_tank
        self.seating_capacity = seating_capacity
        self.maximum_velocity = maximum_velocity
self 代表类的实例，self 在定义类的方法时是必须有的，虽然在调用时不必传入相应的参数。
__init__() 方法是一种特殊的方法，被称为类的构造函数或初始化方法，当创建 vehicle 类的实例时就会调用该方法来定义这些属性。


for key, value in dictionary.items()

(2)数据的查看与检查
df.info() # 查看数据框 (DataFrame) 的索引、数据类型及内存信息
df.describe() # 对于数据类型为数值型的列，查询其描述性统计的内容
s.value_counts(dropna=False) # 查询每列独特数据值出现次数统计
df.apply(pd.Series.value_counts(dropna=False)) # 查询数据框 (Data Frame) 中每个列的独特数据值出现次数统计
data.dtypes  #查看数据类型 
s.dtype  #查看元组中元素的数据类型 
apple.index.is_unique #索引唯一性查看
 
(3)数据选取
df[col] # 以数组 Series 的形式返回选取的列
df[[col1, col2]] # 以新的数据框(DataFrame)的形式返回选取的列
s.iloc[0] # 按照位置选取
s.loc['index_one'] # 按照索引选取
df.iloc[0,:] # 选取第一行
df.iloc[0,0] # 选取第一行的第一个元素

(4)数据的清洗
pd.isnull(data).sum() #每列，有多少数据值缺失
pd.notnull()
df.dropna() # 移除数据框 DataFrame 中包含空值的行
df.dropna(axis=1) # 移除数据框 DataFrame 中包含空值的列
df.dropna(axis=1,thresh=n) # 移除数据框df中空值个数不超过n的行
del crime['Total']  #删除名为Total的列
df.fillna(x) # 将数据框 DataFrame 中的所有空值替换为 x
s.fillna(s.mean()) #将所有空值替换为平均值
s.astype(float) # 将数组(Series)的格式转化为浮点数
s.replace(1,'one') # 将数组(Series)中的所有1替换为'one'
s.replace([1,3],['one','three']) # 将数组(Series)中所有的1替换为'one', 所有的3替换为'three'
df.rename(columns=lambda x: x + 2) # 将全体列重命名
df.rename(columns={'old_name': 'new_ name'}) # 将选择的列重命名
df.set_index('column_one') # 改变索引,column_one作为索引
iris = iris.dropna(how='any') # 删除有缺失值的行

def fix_century(x): # 2061年？我们真的有这一年的数据？这一列的数据每个减去100
  year = x.year - 100 if x.year > 1989 else x.year
  return datetime.date(year, x.month, x.day)
# apply the function fix_century on the column and replace the values to the right ones
data['Yr_Mo_Dy'] = data['Yr_Mo_Dy'].apply(fix_century)

dollarizer = lambda x: float(x[1:-1])
chipo.item_price = chipo.item_price.apply(dollarizer) #将item_price（字符串类型）转换为浮点数


(5)数据的过滤(```filter```),排序(```sort```)和分组(```groupby```)
df[df[col] > 0.5] # 选取数据框df中对应行的数值大于0.5的全部行
df[(df[col] > 0.5) & (df[col] < 0.7)] # 选取数据框df中对应行的数值大于0.5，并且小于0.7的全部行
df.sort_values(col1) # 按照数据框的列col1升序(ascending)的方式对数据框df做排序
df.sort_values(col2,ascending=False) # 按照数据框的列col2降序(descending)的方式对数据框df做排序
df.sort_values([col1,col2],ascending=[True,False]) # 按照数据框的列col1升序，col2降序的方式对数据框df做排序
data.query('day == 1')  #'day'这一列取1的行
df.query('a > b') 与 df[df.a > df.b] 等价

df.groupby(col) # 按照某列对数据框df做分组
df.groupby(col1)[col2].mean() #按照列col1对数据框df做分组处理后，返回对应的col2的平均值
df.pivot_table(index=col1,values=[col2,col3],aggfunc=mean) # 做透视表（交叉表），索引为col1,针对的数值列为col2和col3，分组函数为平均值
df.groupby('A').agg(np.mean)
df.apply(np.mean) # 对数据框df的每一列求平均值
df.apply(np.max,axis=1) # 对数据框df的每一行求最大值
chipo.groupby(by=['order_id'])['item_price'].sum().mean()  #每一单(order)对应的平均总价是多少？
chipo.item_name.value_counts()   #每种不同的商品被售出次数
chipo.item_name.nunique() #在item_name这一列中，一共有多少不同种类的商品被下单？
chipo.choice_description.value_counts().head()  #在choice_description中，下单次数最多的商品是什么？
euro12.loc[euro12.Team.isin(['England', 'Italy', 'Russia']), ['Team','Shooting Accuracy']] #找到英格兰(England)、意大利(Italy)和俄罗斯(Russia)的射正率(Shooting Accuracy)

(6)数据的连接(```join```)与组合(```combine```)

all_data = pd.concat([data1, data2])  #将data1和data2两个数据框按照行的维度进行合并  列数应该相等
all_data = pd.concat([data1, data2], axis = 1)  #将data1和data2两个数据框按照列的维度进行合并  行数应该相等

df1.join(df2,on=col1,how='inner') # 对数据框df1和df2做内连接，其中连接的列为col1

(7)数据的统计
df.mean() # 得到数据框df中每一列的平均值
df.corr() # 得到数据框df中每一列与其他列的相关系数
df.count() # 得到数据框df中每一列的非空值个数
df.max() # 得到数据框df中每一列的最大值
df.min() # 得到数据框df中每一列的最小值
df.median() # 得到数据框df中每一列的中位数
df.std() # 得到数据框df中每一列的标准差

s.idxmax()  s.idxmin()   #方法，可以返回数组中最大（小）值的索引值，
argmin()    argmax()     #返回索引位置

(8)画图
扇形图
散点图
直方图

