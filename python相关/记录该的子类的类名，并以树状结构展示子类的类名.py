#!/usr/bin/env python
# encoding: utf-8

class RegisterClasses(type):
    def __init__(cls, name, bases, atts):
        super(RegisterClasses, cls).__init__(name, bases, atts)

        #创建一个集合，这样继承元类，都会有一个childrens的集合
        cls.childrens = set()

        #将把当前的子类保存到父类中去
        for base in bases:
            if hasattr(base, 'childrens'):
                base.childrens.add(cls)

    #classmethod, called on class object
    def __iter__(cls):
        return iter(cls.childrens)

    def __str__(cls):
        if len(cls.childrens) > 0:
            return cls.__name__ + ": " + ", ".join([sc.__name__ for sc in cls])
        else:
            return cls.__name__

class Shape(metaclass=RegisterClasses):
    pass

print ("---------------------")
class Round(Shape): pass
class Square(Shape): pass
class Triangular(Shape): pass
class Boxy(Shape): pass
print (Shape)
print ("---------------------")
class Circle(Round): pass
class Ellipse(Round): pass
print (Shape)
print ("---------------------")
for s in Shape: #Iterator over subclasses (def __str__(cls):)
    print (s)
print ("---------------------")
for cls in Shape.childrens:
    if len(cls.childrens) > 0:
        print (cls.__name__ + ": " + ", ".join([sc.__name__ for sc in cls]))
    else:
        print (cls.__name__)
