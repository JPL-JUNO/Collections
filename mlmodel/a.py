
class B:
    def __init__(self):
        self.b = 10
        self.a = 1000

    def change_a(self):
        self.a = self.a + self.b


class A(B):
    def __init__(self):
        super().__init__()


a = A()
b = B()
a.change_a()
print(a.a)
