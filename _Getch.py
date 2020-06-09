import msvcrt

class _Getch:
    def __call__(self):
        return msvcrt.getch()
inkey = _Getch()

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

arrow_keys = {
    b'w' : UP,
    b'a' : LEFT,
    b's' : DOWN,
    b'd' : RIGHT
}
