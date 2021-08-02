# ported from https://easings.net/
import math

def linear(x: float) -> float:
    return x

def easeInSine(x: float) -> float:
    return 1 - math.cos((x * math.pi) / 2)


def easeOutSine(x: float) -> float:
    return math.sin((x * math.pi) / 2)


def easeInOutSine(x: float) -> float:
    return -(math.cos(math.pi * x) - 1) / 2


def easeInQuad(x: float) -> float:
    return x * x


def easeOutQuad(x: float) -> float:
    return 1 - (1 - x) * (1 - x)


def easeInOutQuad(x: float) -> float:
    if x < 0.5:
        return 2 * x * x
    return 1 - pow(-2 * x + 2, 2) / 2


def easeInCubic(x: float) -> float:
    return x * x * x


def easeOutCubic(x: float) -> float:
    return 1 - pow(1 - x, 3)


def easeInOutCubic(x: float) -> float:
    if x < 0.5:
        return 4 * x * x * x
    return 1 - pow(-2 * x + 2, 3) / 2


def easeInQuart(x: float) -> float:
    return x * x * x * x


def easeOutQuart(x: float) -> float:
    return 1 - pow(1 - x, 4)


def easeInOutQuart(x: float) -> float:
    if x < 0.5:
        return 8 * x * x * x * x
    return 1 - pow(-2 * x + 2, 4) / 2


def easeInQuint(x: float) -> float:
    return x * x * x * x * x;


def easeOutQuint(x: float) -> float:
    return 1 - pow(1 - x, 5)


def easeInOutQuint(x: float) -> float:
    if x < 0.5:
        return 16 * x * x * x * x * x
    return 1 - pow(-2 * x + 2, 5) / 2


def easeInExpo(x: float) -> float:
    if x == 0:
        return 0
    return pow(2, 10 * x - 10)


def easeOutExpo(x: float) -> float:
    if x == 1:
        return 1
    return 1 - pow(2, -10 * x)


def easeInOutExpo(x: float) -> float:
    if x == 0:
        return 0
    if x == 1:
        return 1
    if x < 0.5:
        return pow(2, 20 * x - 10) / 2
    return (2 - pow(2, -20 * x + 10)) / 2


def easeInCirc(x: float) -> float:
    return 1 - sqrt(1 - pow(x, 2))


def easeOutCirc(x: float) -> float:
    return sqrt(1 - pow(x - 1, 2))


def easeInOutCirc(x: float) -> float:
    if x < 0.5:
        return (1 - sqrt(1 - pow(2 * x, 2))) / 2
    return (sqrt(1 - pow(-2 * x + 2, 2)) + 1) / 2


def easeInBack(x: float) -> float:
    c1 = 1.70158
    c3 = c1 + 1
    return c3 * x * x * x - c1 * x * x


def easeOutBack(x: float) -> float:
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * pow(x - 1, 3) + c1 * pow(x - 1, 2)


def easeInOutBack(x: float) -> float:
    c1 = 1.70158
    c2 = c1 * 1.525
    if x < 0.5:
        return (pow(2 * x, 2) * ((c2 + 1) * 2 * x - c2)) / 2
    return (pow(2 * x - 2, 2) * ((c2 + 1) * (x * 2 - 2) + c2) + 2) / 2


def easeInElastic(x: float) -> float:
    c4 = (2 * math.pi) / 3
    if x == 0:
        return 0
    if x == 1:
        return 1
    return -pow(2, 10 * x - 10) * math.sin((x * 10 - 10.75) * c4)


def easeOutElastic(x: float) -> float:
    c4 = (2 * math.pi) / 3
    if x == 0:
        return 0
    if x == 1:
        return 1
    return pow(2, -10 * x) * math.sin((x * 10 - 0.75) * c4) + 1


def easeInOutElastic(x: float) -> float:
    c5 = (2 * math.pi) / 4.5
    if x == 0:
        return 0
    if x == 1:
        return 1
    if x < 0.5:
        return -(pow(2, 20 * x - 10) * math.sin((20 * x - 11.125) * c5)) / 2
    return (pow(2, -20 * x + 10) * math.sin((20 * x - 11.125) * c5)) / 2 + 1


def easeInBounce(x: float) -> float:
    return 1 - easeOutBounce(1 - x)


def easeOutBounce(x: float) -> float:
    n1 = 7.5625
    d1 = 2.75
    if x < 1 / d1:
        return n1 * x * x
    elif x < 2 / d1:
        x -= 1.5 / d1
        return n1 * (x) * x + 0.75
    elif x < 2.5 / d1:
        x -= 2.25 / d1
        return n1 * (x) * x + 0.9375
    else:
        x -= 2.625 / d1
        return n1 * (x) * x + 0.984375


def easeInOutBounce(x: float) -> float:
    if x < 0.5:
        return (1 - easeOutBounce(1 - 2 * x)) / 2
    return (1 + easeOutBounce(2 * x - 1)) / 2;