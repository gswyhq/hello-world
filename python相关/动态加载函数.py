#!/usr/bin/python3
# coding: utf-8

class A():
    def abc(self, k):
        print(k, 3)

    def acc(self, k):
        print(k, 4)


a = A()
d = getattr(a, 'abc')
d(234)



def main():
    pass


if __name__ == '__main__':
    main()