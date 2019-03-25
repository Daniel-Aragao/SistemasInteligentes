class a:
    def __init__(self):
        self.abcd = "hello"
        self.__ab = "hi"
        print('a')
    def __ppp(self, abc):
            print(abc)


class b(a):
    def __init__(self):
            super().__init__()
            print('b')
    def ccc(self,abc):
            self._a__ppp(abc + self.abcd + self._a__ab)

t = b()
t.ccc('oi')