
# coding: utf-8

# In[15]:


a=10
b=20
c=a+b
print("c", str(c))
a="hi i am mallappa"
print(a[1])
type(c)
type(a)


# In[17]:


for i in range(0,10,2):
    print(i)


# In[21]:


for i in range(10,-1,-2):
    print(i)
    


# In[34]:


sum=0
for i in range(0,5):
    sum=sum+i;
print("sum",sum)


# In[36]:


for i in range(0,5):
    j=i*100
    print(i,j)


# In[47]:


for i in range(6):
    print(i)
    for j in range(2):
        
        print("hello")


# In[42]:


num=20
if num%2==0:
    print(str(num) +"no is even")
else:
    print(str(num) +"no is odd")


# In[1]:


for i in range(0,5):
   
   if i%2==0:
       
       print(str(i) +"no is even")
   else:
       print(str(i) +"no is odd")


# day=1
# if 
# 

# # numpy

# In[50]:


import numpy as np
a=np.array([1,2,3,4])
b=np.array([11,12,13,14])
c=a+b
print(c)


# In[56]:


a=np.array([[1,2],[3,4]])
b=np.array([[11,12],[13,14]])
c=a+b
print(c)
print(len(a))
print(a[0])
s=""
for i in range(0,len(a)):
    for j in range(0,len(a[i])):
        s=s+str(a[i][j])+""
    s=s+"\n"
print(s)
    


# In[68]:


a=np.array([[1,2],[3,4]])
b=np.array([[11,12],[13,14]])
c=a*b
print(c)
f=b/a
print(f)


# In[66]:


a=np.matrix(a)
b=np.matrix(b)
c=a*b
print(c)
e=b/a
print(e)


# In[74]:


aa=np.transpose(a)
aa
bb=np.concatenate((a,b),axis=0)
bb


# # exception handling and formating

# In[86]:


a=10
b=20
c=a+b;
s="the value of addition of {0} and {1} is ={2}".format(a,b,c)
print(s)
ss="the value of addition of {var_a} and {var_b} is ={var_c}".format(var_a=a,var_b=b,var_c=c)
print(ss)
sss="name={std_name:10} age={std_age:10}".format(std_name="mallappa",std_age=30)
print(sss)
ssss="name={std_name:>10} age={std_age:10}".format(std_name="mallappa",std_age=30)
print(ssss)


# In[91]:


a=123455
s="{var_a:e}".format(var_a=a)
print(s)


# In[92]:


a=123455
s="{var_a:e}".format(var_a=a)
print(float(s)+10)


# In[100]:



a="1.2"
b=12.12345555
c=int(float(a))+b
print(c)

cc=np.ceil(float(a))+b
print(cc)

ccc=np.ceil(float(a),3)+b
print(ccc)


# In[104]:


#converting binary and hexa no to intiger
b="0b1101"
i=int(b,2)
print("i={0}".format(i))

h="0x111"
i=int(h,16)
print("i={0}".format(i))


# In[112]:


print(np.round(np.pi,4))

c=np.pi*123456
print(c)
print("{0:.3f}".format(c))


# # exception handling

# In[121]:


a=14
b=2
c=a/b
s="{0}/{1}={2}".format(a,b,c)
print(c)

a=14
b=0
cc=a/b
s="{0}/{1}={2}".format(a,b,cc)
print(cc)


# In[138]:


try:
    a="10"
    b=2
    c=a+b
    print(c)
except Exception as ex:
    print("some error:" +str(ex))


# In[139]:


try:
    a=10
    b=0
    c=a/b
    print(c)
except Exception as ex:
    print("some error:" +str(ex))


# In[144]:


import sys
try:
    a=10
    b=0
    c=a/b
    print(c)
except:
    print("some error:" +str(sys.exc_info()[0]))


# In[ ]:


#


# # normal operations
# 

# In[150]:


def add(a,b):
    c=a+b
    s=a-b
    m=a*b
    d=a/b
    return(c,s,m,d)


# In[151]:


[c,s,m,d]=add(11,20)
print(c)
print(s)
print(m)
print(d)


# In[159]:


global a
a=20
def add():
    global a
    a=a*2
    


# In[161]:


print(a)
add()
print(a)


# # python list

# In[3]:


l=[1,2,3,[1,2,3]]
l
#l=[[3],[3]]
#l


# In[177]:


a=[1,2,3,4]

del a[0]
print("first",a)
a.insert(1,20)
print("afer insert", a)
del a[1]
print("after deleting", a)
a.append(10)
print("after appending", a)
a.append([100,200])
print("after appending list within list", a)
a.extend([111,"hi",222])
print("after extend  commond used", a)


# In[187]:


a=[1,2,3,4,5,6,7,8]
try:
    del a[1]
    print(a)
    a.remove(5)
    print(a)
except Exception as ex:
    print(str(ex))
print(a)
a.pop()
print(a)


# In[188]:


a=[1,2,3,4,5,6,7,8]
try:
    del a[10]
    print(a)
    a.remove(5)
    print(a)
except Exception as ex:
    print(str(ex))
print(a)
a.pop()
print(a)


# In[190]:


a=[1,2,3,4,5,6,7,8]
try:
    del a[1]
    print(a)
    a.remove(a[9])
    print(a)
except Exception as ex:
    print(str(ex))
print(a)
a.pop()
print(a)


# In[206]:


a=[1,2,3,4,5,6]
for i in range(0,len(a),2):
    print("ascending ", a[i])
   # print(_*60)
for i in range(len(a)-1,-1,-2):
    print("reverse", a[i])


# In[200]:


print(a[0:len(a):1])


# In[210]:


import math as m
print("sum",sum(a[2:4]))


# In[4]:


#no matematical operation on list, we have to convert into numpy array
a=[1,2,3,4,5]
a=a+1
print(a)


# In[260]:


import numpy as np
a=np.array(a)
a=a+100
print(a)


# In[273]:


a=[1,2,3,4,5]
#a=a+200
import numpy as np
a=np.array(a)
b=a+100
print(b)
bb=a*100
print(bb)


# In[270]:


b=b.tolist()
print(b)


# In[275]:


l=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,12,2,2,2,2,2,2,2,2,3,3,3,33]


# In[277]:


print(l.count(1))


# In[304]:


a=[11,21,31,41,51,61,1,2,3,4,5,4,3,2,1,4,3,2]
try:
    x=100
    a=a.index(x)
    print(a)
except Exception as ex:
    print(str(ex))


# In[297]:


a
x=2
a=np.array(a)
ind=np.where(a==x)
print(ind)


# In[300]:


a
x=2
a=np.array(a)
ind=np.where(a==x)[0].tolist()
print(ind)


# # python sets

# # # difference betwwen sets and lists

# In[36]:


c={9,6,9,8,2,1,2,3,4,5,6,2,3,4,1,2,3}
print("list of elements in the sets", c)

c=[9,6,9,8,2,1,2,3,4,5,6,2,3,4,1,2,3]
print("list of elements in the list", c)


# In[307]:


print(type(a))


# In[325]:


# difference between lists and sets
a=[8,4,3,6,1,3,1,2,3,4,5,6,7]
print(a)
print(type(a))
a.sort()
print(a[0:7])


# In[329]:


a={6,4,2,1,2,3,2,1,3,4,2}
a
type(a)
print(a)


# In[335]:


#union and intersection are scalar operation

a={1,2,3,4}
b={3,4,5,6,7}
c=a.union(b)
print(c)
d=a.intersection(b)
print(d)
#difference is vector operation
e=b.difference(a)
print(e)
ee=a-b#a+b not working
ee


# In[340]:


l={1,2,3,"i",4,4.0,"mallappa","g"}
l


# In[342]:


l=[1,2,3,4,1,2,3,4,2,2,2,2]
print("list values", l)
s=set(l)
print("set values", s)


# In[343]:


s.add(10)


# In[345]:


s.update({1,2,3,4,5,6,7,8,9})
s


# In[346]:





# In[347]:


s.add(10)


# In[19]:


a=[1,2,3,4,5,6,7]
a
b=set(a)
print("list converted into set", b)
b.remove(6)
print("items removed from sets", b)
b.discard(7)
print("items discared", b)
b.pop()
print("popped item from b", b)


# In[7]:


a={1,2,3,4}
b=list(a)
#print(del b[0])
b

#print(5 in a)


# In[21]:


b=[1,2,3,4,5,6,7]

print("list converted into set", b)
b.remove(6)
print("items removed from sets", b)
# list object have no discarad attrbute
#b.discard(7)
#print("items discared", b)
b.pop()
print("popped item from b", b)


# In[23]:


print(6 in b)


# In[8]:


b=[1,2,3,4,5,6,7]
print("value 2 index from list", b.index(2))
#b={1,2,3,4,5,6,7}
#b.index(2)


# # dictionary

# In[61]:


data={"name":"mallappa","id":1,2:35}
data
data["name"]
print(data)

data[8]=0x101
print(data)

data["colname"]='sjce'
print(data)

data[8]="my lucky no"
print(data)

data.pop(8,None)
print("after poped from data dictionary", data)

data.pop(10)
print("after poped from data dictionary", data)



# In[11]:


print("create empty dictionary")
d=dict()
print(d)

d["name"]='mallappa'
print("added name to dictionary",d)

l=[1,2,3,4,5]
d['list']=l
print("added list to dictionary",d)

days={"sun":1,"mon":2,"tues":3}
d["days"]=days
print("dictionary within a dictionary", d)

print("accesing elements from dictionary")
print(d["days"]["sun"])
print(d["days"])

print("days" in d)
print("num" in d)

print("list of keys of dictionary d")
print(d.keys())
print("convert dict keys  to list", list(d.keys()))

d.keys().add("age")
print(d.keys())


# In[111]:


#conerting into keys and accessing the data trough keys from dictionary

keys=list(d.keys())
print(keys)
print("------------")
print(keys[2])

print(d[keys[2]])


# In[116]:


w1=["hi","hello"]
w2=["boat","ball"]
w3={"a":w1,"b":w2}
print(w3)
print(w3["b"])
print("------------------------")
g="anand"
if g[0]=="a":
    w3["a"].append(g)
else:
    w3["b"].append(g)
print(w3)


# In[117]:


g="fbanand"
if g[0]=="a":
    w3["a"].append(g)
else:
    w3["b"].append(g)
print(w3)


# # touple

# In[1]:


l=[1,2,3,4,4,4,4,4]
s={1,2,3,4,4,4,4,4}
t=(1,2,3,4,4,4,4,4)
print(l,s,t)


# In[2]:


a=1
b=2
c=3
(x,y,z)=(11,22,33)
print(b,x,y,z)


# In[3]:


type(t)


# In[12]:


t=(11,22,33,44,55)
print("given elememnts of tuples", t)
tl=list(t)
print("converting tuple to list", tl)
tl.append(100)
type(tl)
print("appending new element to list", tuple(tl))


# In[13]:


x=True
y=False
print(x and y)


# In[15]:


n1=2
n2=3
print(n1 & n2)
print(n1 | n2)


# In[17]:


s="""jfldasljfadkjlasflljas
asjasfljfaglgadsglfads"""
type(s)


# In[29]:


s="mallappa"
s
print(s.find('p'))
print(s.find('d'))

print(s.replace("ma","ya"))

print(s.split(" , "))


print(s.count("a"))


# In[30]:


t=("mallappa","kumar","yallappa")
print(t+("ramesh","suresh"))


# In[31]:


t


# In[32]:


t=(1,2,3,4,"m","mallappa")
t


# In[33]:


t[1]


# In[34]:


"m" in t


# In[38]:


a={1,2,3}
b={2,3,4}
print(a|b)
print(a&b)
print(a-b)
print(b-a)


# # pandas
# 

# In[11]:


import pandas as pd
import numpy as np
data=pd.Series([0,10,1,3,2,1,.5])
data
data[1:5]


# In[19]:


data=pd.Series([0,10,1,3,2,1,.5],index=["a","a","c","d","e","f","h"])
print(data)
print("----------------------")
print(data["a"])


# In[15]:


data=pd.Series({"name":"mallu",3:420})
data


# In[20]:


data=pd.Series(5,index=[1,2,3,4])
data


# # pandas dataframe

# In[26]:


d=pd.Series({"a":1,"b":2,"c":3,"d":4})
p=pd.Series({"a":2,"b":3,"c":2,"x":7})
df=pd.DataFrame({"d":d,"p":p})
df


# In[27]:


df.index


# In[29]:


df.columns


# In[30]:


df.d


# In[31]:


df.p


# In[32]:


pd.DataFrame(p,columns=["p"])


# In[34]:


data=[{"a":i,"b":3*i} for i in range(3)]
data
pd.DataFrame(data)


# In[35]:


pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])


# In[42]:


pd.DataFrame(np.random.randn(3,2),columns=["a","b"],index=['x',"y","z"])


# # data indexing and seletion

# In[45]:


import pandas as pd
data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=['a', 'b', 'c', 'd'])
data
data.keys()
list(data.items())


# In[5]:


import pandas as pd
df=pd.read_csv("mallappasiragur.csv")
df


# In[2]:


df


# # matplotlib

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


x=[1,2,3,4,5,6,7]
y=[50,51,45,65,44,55,47]
plt.xlabel("temp")
plt.ylabel("day")
plt.title("temp vs days")
plt.plot(x,y,color="red")


# In[21]:


plt.plot(x,y,"y+")
plt.plot(x,y,marker="3")
plt.plot(x,y,"rD")


# In[19]:


plt.plot(x,y,alpha=.2)


# In[29]:


z=[33,21,54,23,44,67,43]
x=[34,23,12,56,78,23,53]
plt.xlabel("temp")
plt.ylabel("day")
plt.title("temp vs days")
plt.plot(x,y,label="a")
plt.plot(x,z,label="b")
plt.plot(y,z,label="c")

plt.legend(loc="lower right",shadow=True)
plt.grid()


# In[41]:



import numpy as np
com=["mm","nn","vv","dd"]
rev=[66,77,33,44]
ypos=np.arange(len(com))
ypos
plt.bar(ypos+.2,rev,label="rev")
plt.bar(ypos-.2,pr,label="pr")
plt.legend()


# In[40]:


import numpy as np
com=["mm","nn","vv","dd"]
rev=[66,77,33,44]
pr=[23,43,34,44]
ypos=np.arange(len(com))
ypos
plt.bar(ypos,rev)
plt.xticks(ypos,com)
plt.bar(ypos+.2,rev,label="rev")
plt.bar(ypos-.2,pr,label="pr")
plt.legend()


# In[42]:


import numpy as np
com=["mm","nn","vv","dd"]
rev=[66,77,33,44]
pr=[23,43,34,44]
ypos=np.arange(len(com))
ypos
plt.bar(ypos,rev)
plt.yticks(ypos,com)
plt.barh(ypos+.2,rev,label="rev")
plt.barh(ypos-.2,pr,label="pr")
plt.legend()


# # histogram

# In[43]:


bds=[111,89,110,100,89,90]
plt.hist(bds)


# In[47]:


bds=[111,89,110,100,89,90]
plt.hist(bds,bins=3,rwidth=.8,color="red")


# In[56]:


bdsm=[111,89,110,100,89,90]
bdsw=[110,78,89,99,100,105]
plt.xlabel("qqqqqqqqqqqqqqqqqqqqqqqqqq")
plt.ylabel("lllllllllllllllllllllllllllll")
plt.title("ooooooooooooooooooooo")
plt.hist([bdsw,bdsm],bins=[80,100,125,150],rwidth=.8,color=("red","green"),label=["bdsw","bdsm"])
plt.legend()


# In[59]:


bdsm=[111,89,110,100,89,90]
bdsw=[110,78,89,99,100,105]
plt.xlabel("qqqqqqqqqqqqqqqqqqqqqqqqqq")
plt.ylabel("lllllllllllllllllllllllllllll")
plt.title("ooooooooooooooooooooo")
plt.hist([bdsw,bdsm],bins=[80,100,125,150],rwidth=.8,color=("red","green"),label=["bdsw","bdsm"],orientation="horizontal")
plt.legend()


# # pie chart

# In[71]:


plt.axis("equal")
x=[23,34,54,11]
yval=["a","b","c","d"]
plt.pie(x,labels=yval,radius=1.5,autopct="%.2f%%",explode=[0,0,0,0],startangle=90)
plt.show()
plt.savefig("piecart.png")


# # seaborn visualization
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


tips=sns.load_dataset("tips")
sns.barplot(x="day",y="tip",data=tips)


# In[13]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.barplot(x="day",y="tip",data=tips,hue="sex", palette="spring")


# In[16]:


sns.barplot(x="total_bill",y="day",data=tips,palette="spring")


# In[19]:


sns.barplot(x="total_bill",y="day",data=tips,palette="spring",ci=97)


# In[25]:


sns.barplot(x="size",y="tip",data=tips,palette="winter",ci=97,capsize=.15)


# In[27]:


sns.barplot(x="size",y="tip",data=tips,palette="winter",ci=97,capsize=.15)


# In[32]:


sns.barplot(x="smoker",y="tip",data=tips,palette="winter_r",ci=34,estimator=median)


# In[42]:


import numpy as np

sns.barplot(x="smoker",y="tip",data=tips,ci=34,palette="winter_r",estimator=median)


# In[39]:


from numpy import median


# In[46]:


sns.barplot(x=[1,2,3],y=[2,3,4],palette="winter",ci=98)


# # density plots

# In[47]:


n1=np.random.randn(150)
n1


# In[49]:


sns.distplot(n1)


# In[54]:


lb=pd.Series(n1,name="variable_x")
sns.distplot(lb)


# In[56]:


sns.distplot(lb,vertical=True,color="red")


# In[57]:


sns.distplot(lb,vertical=True,color="red",hist=False)


# In[58]:


sns.distplot(lb,vertical=True,color="red",rug=True)


# In[59]:


tips.head()


# In[60]:


tips.columns


# # boxplots

# In[61]:


sns.boxplot(x=tips["size"])


# In[63]:


sns.boxplot(x=tips["total_bill"])


# In[67]:


sns.boxplot(x="size",y="total_bill",data=tips)


# In[68]:


sns.boxplot(x="day",y="total_bill",data=tips,hue='sex',palette="spring")


# In[69]:


sns.boxplot(x="day",y="total_bill",data=tips,hue='smoker',palette="winter")


# In[71]:


sns.boxplot(x="day",y="total_bill",data=tips,order=["sun","fri",'thus',"sat"],color="red")


# In[73]:


sns.boxplot(x="sex",y="tip",data=tips,order=["Female","Male"])


# In[74]:


iris=sns.load_dataset("iris")


# In[75]:


iris.head()


# In[77]:


sns.boxplot(data=iris,palette='coolwarm')


# In[80]:


sns.boxplot(data=iris,palette='coolwarm',orient='horizontal')


# In[81]:


sns.boxplot(data=iris,palette='coolwarm',orient='vertical')


# In[84]:


sns.boxplot(x="day",y="total_bill",data=tips,palette="husl")
sns.swarmplot(x="day",y="total_bill",data=tips,color="black")


# In[85]:


sns.boxplot(x="day",y="total_bill",data=tips,palette="husl")
sns.swarmplot(x="day",y="total_bill",data=tips,color=".5")


# # stripplots
# 

# In[86]:


sns.stripplot(x=tips["tip"])


# In[90]:


sns.stripplot(x="day",y="total_bill",data=tips)


# In[91]:


sns.stripplot(x="day",y="total_bill",data=tips,jitter=True)


# In[92]:


sns.stripplot(x="total_bill",y="day",data=tips,jitter=True)


# In[93]:


sns.stripplot(x="total_bill",y="day",data=tips,jitter=True,linewidth=1.2)


# In[95]:


sns.stripplot(x="day",y="total_bill",data=tips,jitter=True,              hue="smoker",linewidth=1.2,split=True)


# In[99]:


tips.head()
sns.stripplot(x="sex",y="tip",data=tips)
sns.stripplot(x="sex",y="tip",data=tips,marker="D",size=15,hue="sex",             edgecolor="green")


# In[100]:


sns.stripplot(x="tip",y="day",data=tips)
sns.boxplot(x="tip",y="day",data=tips)


# In[103]:


sns.stripplot(x="tip",y="day",data=tips,jitter=True)
sns.violinplot(x="tip",y="day",data=tips,color=".8")


# # pair grid plot

# In[106]:


iris=sns.load_dataset("iris")

x=sns.PairGrid(iris)
x=x.map(plt.scatter)


# In[108]:


iris=sns.load_dataset("iris")

x=sns.PairGrid(iris)
x=x.map_diag(plt.hist)


# In[111]:


iris=sns.load_dataset("iris")

x=sns.PairGrid(iris,hue="species")
x=x.map_diag(plt.hist)
x=x.map_offdiag(plt.scatter)
x=x.add_legend()


# # violin plots
# 

# In[112]:


sns.violinplot(x=tips["tip"])


# In[114]:


sns.violinplot(x="day",y='total_bill',data=tips,hue="sex",               palette='winter')


# In[118]:


sns.violinplot(x="day",y='total_bill',data=tips,hue="sex",               palette="coolwarm",split=True,inner="quartile")


# In[119]:


sns.violinplot(x="day",y='total_bill',data=tips,hue="sex",               palette="coolwarm",split=True,inner="quartile",              scale="count")


# In[120]:


sns.violinplot(x="day",y='total_bill',data=tips,hue="sex",               palette="coolwarm",split=True,inner="stick",              scale="count")


# In[126]:


flights=sns.load_dataset"flights")
flights.head()


# In[142]:


fligts=sns.load_dataset("flights")


# In[143]:


flights.head()


# In[146]:


sns.clustermap(flights,cmap="coolwarm",linewidth=1.2)


# In[147]:


flights


# In[152]:


normal=np.random.rand(12,15)
sns.heatmap(normal,annot=True)


# In[153]:


flights=flights.pivot("month",'year',"passengers")
sns.heatmap(flights)


# # facet gris plots

# In[158]:


tips=sns.load_dataset("tips")
sns.FacetGrid(tips,row="smoker",col="time")


# In[159]:


x=sns.FacetGrid(tips,row="smoker",col="time")
x=x.map(plt.hist,"total_bill")


# In[160]:


x=sns.FacetGrid(tips,row="smoker",col="time")
x=x.map(plt.hist,"total_bill",color="green")


# # kde plots

# In[161]:


mean=[0,0]
cov=[[.2,0],[0,3]]
x_axis,y_axis=np.random.multivariate_normal(mean,cov,size=40).T
sns.kdeplot(x_axis)


# In[167]:


sns.kdeplot(x_axis,y_axis,cmap="winter")


# # joint plots

# In[175]:


tips=sns.load_dataset("tips")
sns.jointplot(x="total_bill",y="tip",data=tips)


# In[178]:


tips=sns.load_dataset("tips")
sns.jointplot(x="total_bill",y="tip",data=tips,kind="kde")


# In[177]:


tips=sns.load_dataset("tips")
sns.jointplot(x="total_bill",y="tip",data=tips,kind="reg")


# In[181]:


sns.regplot(x="tip",y="total_bill",data=tips,marker="d",            line_kws={"color":"yellow","linewidth":2.4})


# In[182]:


sns.regplot(x="tip",y="total_bill",data=tips,ci=75)


# In[186]:


sns.regplot(x="size",y="total_bill",data=tips,ci=97,x_jitter=.03)


# In[187]:


sns.regplot(x="size",y="total_bill",data=tips,ci=97,            x_extimator=np.mean)


# # pairplot

# In[188]:


sns.pairplot(iris)


# In[189]:


sns.pairplot(iris,hue="species")


# In[190]:


sns.pairplot(iris,hue="species",palette="husl",             markers=["d","o","s"])


# In[191]:


sns.pairplot(iris,hue="species",palette="husl",             markers=["d","o","s"])


# In[192]:


sns.pairplot(iris,diag_kind="kde",kind="reg",palette="husl"             ,hue="species")


# In[8]:


import pandas as pd
data=pd.DataFrame({"id":[1,2,-10,4,32,1,1,5,6,],"sal":[12,12,3,2,4,53,21,2,-10]})
data


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.boxplot(data["id"])
data.describe()


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


a=np.arange(10)
a
b=np.arange(200)
b
plt.plot(a,"r-",b,"b+")
plot.show()


# # simple linear rgression in python sklearn

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


from sklearn import linear_model
from sklearn.cross_validation import train_test_split


# In[18]:


from sklearn.datasets import load_boston
boston=load_boston()
boston


# In[20]:


df_x=pd.DataFrame(boston.data,columns=boston.feature_names)
df_y=pd.DataFrame(boston.target)


# In[21]:


df_x.describe()


# In[22]:


reg=linear_model.LinearRegression()


# In[24]:


x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=.2,                                              random_state=4)


# In[26]:


reg.fit(x_train,y_train)


# In[29]:


reg.coef_


# In[30]:


a=reg.predict(x_test)


# In[31]:


a[1]


# In[32]:


y_test[0]


# In[34]:


np.mean((a-y_test)**2)


# In[3]:


import numpy as np


# In[4]:


df=np.arange(10).reshape(5,2)


# In[5]:


df


# # list comprehension with examples

# In[2]:


h_letters = [ letter for letter in 'human' ]
print( h_letters)


# In[4]:


y = list(map(lambda x: x, 'human'))
y


# In[8]:


[x for x in range(20) if x%2==0]


# In[10]:


[x for x in range(200) if x%2==0 if x%5==0 if x%50==0]


# In[11]:


[[1,2,3],[4,5,6],[7,8,9]]


# In[13]:


["even" if x%2==0 else "odd" for x in range(10)]


# In[13]:


matrix = [[1, 2],[3,4],[5,6],[7,8]]
#[[row[i] for row in matrix] for i in range(2)]
matrix


# In[19]:


[i*i for i in range(10) if i%2==0]


# In[22]:


address=["mallappa",'satyappa',"siragur"]
[word[0]for word in address]


# In[23]:


string="mallappa123satyappa456siragur"
[x for x in string if x.isdigit()]


# In[24]:


import tensorflow as tf


# # nltk text processing

# In[1]:


import nltk


# In[2]:


import numpy


# In[3]:


import string


# In[4]:


from nltk.corpus import stopwords


# In[10]:


stopwords.words("english")[0:10]


# In[20]:


test="this is first time i am going to belgaum!"
test="hi dude, 5ine, i am good here, 123@"


# In[21]:


x=[char for char in test if char not in string.punctuation]
x


# In[11]:


x="".join(x)
x


# In[12]:


x.split()


# In[14]:


clean_s=[word for word in x.split() if word.lower() not in stopwords.words("english")]


# In[15]:


clean_s


# In[31]:


par="i am mallappa siragur. i am staying in pune.jklfailsgfdsaljdas. kjadllhhjfla;hja. afadfhkfjkasffhkfs. jadgjd,dkjfjhdffs. dahdsfafhjsa"
par="hi i am mallappa siragur from karnataka. i am pursuing data science course at marsian"
par


# In[32]:


from nltk.tokenize import sent_tokenize


# In[33]:


sent_tokenize(par)


# In[34]:


from nltk.tokenize import word_tokenize


# In[35]:


word_tokenize("i deosn't am mallappa siragur.")
xx=word_tokenize("i deosn't am mallappa siragur")
xx[2]


# In[36]:


xx


# In[37]:


from nltk.tokenize import regexp_tokenize


# In[38]:


s="i can't do this. i won't do that."


# In[39]:


from nltk.tokenize import word_tokenize


# In[40]:


word_tokenize(s)


# In[41]:


regexp_tokenize(s,"[\w']+")


# In[42]:


regexp_tokenize(s,"[\w']")


# In[43]:


regexp_tokenize(s,"[\w]+")


# In[35]:


from nltk.corpus import stopwords


# In[46]:


ensw=stopwords.words("english")
ensw


# In[37]:


from nltk.tokenize import word_tokenize


# In[44]:


st="wat are u doing exactly right now? i wanted to go office."
pr=word_tokenize(st)
pr


# In[47]:


fl=[item for item in pr if item not in ensw]
fl


# In[48]:


import nltk
from nltk.corpus import wordnet


# In[49]:


word1='weapon'
word1='love'


# In[50]:


synarray=wordnet.synsets(word1)
x=synarray
x[0]


# In[51]:


x


# In[55]:


x[4].definition()


# In[22]:


x[0].name()


# In[23]:


x[0].hypernyms()


# In[24]:


x[0].hyponyms()


# In[25]:


from nltk.corpus import wordnet
sa=wordnet.synsets("win")
sa


# In[26]:


sa[2]


# In[27]:


s=sa[2]


# In[28]:


s.pos()


# In[30]:


s.definition()


# # lammatization

# In[32]:


s.lemmas()


# In[34]:


s.lemmas()[0].name()


# In[37]:


synArr=[]
antArr=[]
for syn in sa:
    for lem in syn.lemmas():
        synArr.append(lem.name())


# In[38]:


synArr


# In[40]:


set(synArr)


# In[41]:


len(synArr)


# In[42]:


len(set(synArr))


# In[44]:


#antonyms
s.lemmas()[0]


# In[45]:


s.lemmas()[0].antonyms()


# In[47]:


s.lemmas()[0].antonyms()[0].name()


# In[49]:


for syn in sa:
    for lem in syn.lemmas():
        for ant in lem.antonyms():
            antArr.append(ant.name())


# In[50]:


antArr


# In[51]:


set(antArr)


# In[52]:


len(antArr)


# In[53]:


set(synArr)


# In[54]:


#palmer similarity
from nltk.corpus import wordnet


# In[55]:


sarr1=wordnet.synsets("cake")
sarr2=wordnet.synsets("loaf")
sarr3=wordnet.synsets("bread")


# In[58]:


cake=sarr1[0]
cake


# In[60]:


loaf=sarr2[0]
loafb=sarr2[1]
loaf


# In[61]:


loafb


# In[62]:


bread=sarr3[0]
bread


# In[63]:


#hypernym tree
cake.wup_similarity(loaf)


# In[64]:


cake.wup_similarity(loafb)


# In[65]:


loafb.wup_similarity(loaf)


# In[66]:


bread.wup_similarity(loaf)


# In[68]:


bread.wup_similarity(loafb)


# In[69]:


loafb


# In[71]:


loafb.hypernyms()


# In[72]:


loafb.hypernyms()[0]


# In[75]:


ref=loafb.hypernyms()[0]
loafb.shortest_path_distance(ref)


# In[76]:


bread.shortest_path_distance(ref)


# In[77]:


loaf.shortest_path_distance(ref)


# In[78]:


cake.shortest_path_distance(ref)


# In[83]:


#path and lc similarity
from nltk.corpus import wordnet


# In[87]:


catarr=wordnet.synsets('cat')
catarr
dogarr=wordnet.synsets("dog")
dogarr


# In[89]:


coi=catarr[0]
doi=dogarr[0]


# In[90]:


coi


# In[91]:


doi


# In[93]:


doi.wup_similarity(coi)


# In[96]:


doi.path_similarity(coi)


# In[98]:


doi.path_similarity(doi)


# In[101]:


coi.lch_similarity(coi)


# In[56]:


#bigrams
import nltk
nltk.download("webtext")


# In[57]:



from nltk.corpus import webtext


# In[58]:


from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


# In[59]:


textwords=[w.lower() for w in webtext.words("pirates.txt")]


# In[60]:


finder=BigramCollocationFinder.from_words(textwords)


# In[61]:


finder.nbest(BigramAssocMeasures.likelihood_ratio,10)


# In[14]:


from nltk.corpus import stopwords
ignored_words=set(stopwords.words("english"))
ignored_words


# In[16]:


filterstops=lambda w: len(w)<3 or w in ignored_words


# In[17]:


finder.apply_word_filter(filterstops)


# In[18]:


finder.nbest(BigramAssocMeasures.likelihood_ratio,10)


# In[21]:


#trigrams
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import TrigramAssocMeasures
from nltk.corpus import webtext


# In[22]:


textwords=[w.lower() for w in webtext.words("grail.txt")]


# In[24]:


finder=TrigramCollocationFinder.from_words(textwords)


# In[25]:


finder.nbest(TrigramAssocMeasures.likelihood_ratio,10)


# In[26]:


from nltk.corpus import stopwords
ignored_words=set(stopwords.words("english"))


# In[28]:


filterstops=lambda w: len(w) <3 or w in ignored_words
finder.apply_word_filter(filterstops)


# In[29]:


finder.nbest(TrigramAssocMeasures.likelihood_ratio,10)


# In[32]:


finder.apply_freq_filter(3)


# In[34]:


#stemming
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import RegexpStemmer


# In[36]:


pstemmer=PorterStemmer()
pstemmer.stem("dancing")


# In[37]:


pstemmer.stem("dancer")


# In[39]:


pstemmer.stem("cooking")


# In[40]:


pstemmer.stem("cookery")


# In[41]:


pstemmer.stem("morning")


# In[44]:


pstemmer.stem("sharpness")


# In[45]:


#porterstemmer  is the least aggressive


# In[48]:


#lstemmer is te more aggressive
lstemmer=LancasterStemmer()
lstemmer.stem("dancing")


# In[49]:


#regex stemmer
rstemmer=RegexpStemmer("ing")
rstemmer.stem("skiing")


# In[50]:


rstemmer.stem("dancing")


# In[51]:


rstemmer=RegexpStemmer("s")
rstemmer.stem("peoples")


# In[53]:


#lammatization
from nltk.stem import WordNetLemmatizer


# In[54]:


lzr=WordNetLemmatizer()


# In[57]:


lzr.lemmatize("working")


# In[64]:


lzr.lemamatize("working",pos="v")


# In[65]:


lzr.lemmatize("working",pos="a")


# In[67]:


# difference between stemming and lammatization
stm=PorterStemmer()
stm.stem("dancing")


# In[73]:


stm.stem("buses")
stm.stem("believes")     


# In[74]:


lzr.lemmatize("buses")
lzr.lemmatize("believes")
              


# In[62]:


#regular expression replacer
import re;


# In[66]:


regex=re.compile(r"don\'t")
fst="i don't go to school"
sst=regex.sub("do not", fst)
print("\n oroginal res:"+ fst)
print("\n edited res:" +sst)


# In[78]:


regex=re.compile("[0123456789]")
fst="i don't go to school at 12. next day i go to school at 30. i brought 100kg apple"
sst=regex.sub(" ", fst)
print("\n oroginal res:"+ fst)
print("\n edited res:" +sst)


# In[3]:


import numpy as np
np.array(a)


# In[4]:


a=(np.random.randn(10))


# In[5]:


np.array()

