class FourPeaks():
    def f(self, x, t):
        tail = self.tail('0', x)
        head = self.head('1', x)
        return max(tail, head) + self.R(x, t)

    def tail(self, b, x):
        return (len(x) - len(x.rstrip(b)))

    def head(self, b, x):
        return len(x.split('0', 1)[0])

    def R(self, x, t):
        tail_bool = self.tail('0', x) > t
        head_bool = self.head('1', x) > t
        if head_bool and tail_bool: return len(x)
        else: return 0

