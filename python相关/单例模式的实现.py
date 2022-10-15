
单例模式有以下几种实现方式。
方法一、实现__new__方法，然后将类的一个实例绑定到类变量_instance上；如果cls._instance为None，则说明该类还没有被实例化过，new一个该类的实例，并返回；如果cls._instance不为None，直接返回_instance，代码如下：

class Singleton(object):
  
  def __new__(cls, *args, **kwargs):
    if not hasattr(cls, '_instance'):
      orig = super(Singleton, cls)
      cls._instance = orig.__new__(cls, *args, **kwargs)
    return cls._instance
  
class MyClass(Singleton):
  a = 1
  
one = MyClass()
two = MyClass()
  
#one和two完全相同,可以用id(), ==, is检测
print id(one)  # 29097904
print id(two)  # 29097904
print one == two  # True
print one is two  # True
方法二、本质上是方法一的升级版，使用__metaclass__（元类）的高级python用法，具体代码如下：

class Singleton2(type):
  
  def __init__(cls, name, bases, dict):
    super(Singleton2, cls).__init__(name, bases, dict)
    cls._instance = None
  
  def __call__(cls, *args, **kwargs):
    if cls._instance is None:
      cls._instance = super(Singleton2, cls).__call__(*args, **kwargs)
    return cls._instance
  
class MyClass2(object):
  __metaclass__ = Singleton2
  a = 1
  
one = MyClass2()
two = MyClass2()
  
print id(one)  # 31495472
print id(two)  # 31495472
print one == two  # True
print one is two  # True

方法三、使用Python的装饰器(decorator)实现单例模式，这是一种更Pythonic的方法；单利类本身的代码不是单例的，通装饰器使其单例化，代码如下：

def singleton(cls, *args, **kwargs):
  instances = {}
  def _singleton():
    if cls not in instances:
      instances[cls] = cls(*args, **kwargs)
    return instances[cls]
  return _singleton
  
@singleton
class MyClass3(object):
  a = 1
  
one = MyClass3()
two = MyClass3()
  
print id(one)  # 29660784
print id(two)  # 29660784
print one == two  # True
print one is two  # True
